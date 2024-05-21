```python
import jax
jax.config.update("jax_enable_x64", True)

import avb
from avb.distributions import (
    PrecisionNormal,
    LinearDynamicalSystem,
    Reshaped,
)
from avb.util import tree_leaves_with_path
from datetime import datetime
import functools
import ifnt
import isoweek
from jax import numpy as jnp
import jaxopt
from localscope import localscope
from matplotlib import pyplot as plt
import numpy as np
import numpyro
from numpyro import distributions as dists
import pandas as pd
from pathlib import Path
from sklearn import metrics, preprocessing
import shutil
import tensorboardX
from tqdm.notebook import tqdm

%reload_ext snakeviz
%reload_ext line_profiler
```

Load the raw data. We only consider case data aggregated by US state from the last three and a half years. We filter out diseases with very low number of records and US territories. Observation weeks start on Sundays (which we verify), and we record the ISO week of the following Monday as an identifier (rather than using the `epi_week` field provided by the dataset). That makes it easier to map onto holidays, for example.

```python
filename = Path("../data/ProjectTycho_Level2_v1.1.0.csv")
if not filename.is_file():
    raise FileNotFoundError(
        f"Data file at `{filename}` is missing. Download it from "
        "https://www.tycho.pitt.edu/data/."
    )

raw = pd.read_csv(
    filename,
    usecols=["country", "loc", "loc_type", " event", "number", "from_date", "disease"],
    parse_dates=["from_date"],
)
raw.rename(columns={" event": "event", "loc": "state"}, inplace=True)
raw = raw[(raw.country == "US") &
    (raw.loc_type == "STATE") &
    (raw.event == "CASES") &
    (raw.from_date >= datetime(2011, 1, 1)) &
    (~raw.disease.isin([
        "EHRLICHIOSIS/ANAPLASMOSIS",
        "BABESIOSIS",
        "ROCKY MOUNTAIN SPOTTED FEVER",
    ])) &
    (~raw.state.isin([
        "AMERICAN SAMOA",
        "NORTHERN MARIANA ISLANDS",
        "GUAM",
        "VIRGIN ISLANDS",
    ]))]
raw.drop(columns=["country", "loc_type", "event"], inplace=True)

# Map parts of New York to the state.
raw.replace({"state": {
    "UPSTATE NEW YORK": "NEW YORK",
    "NEW YORK CITY": "NEW YORK",
}}, inplace=True)

# Fill the temporal interior with zeros.
parts = []
for (state, disease), subset in raw.groupby(["state", "disease"]):
    # There may be overlap in dates because we aggregated New York. We take the sum
    # first so each date appears exactly once.
    subset = subset.groupby("from_date").number.sum()
    if len(subset) < 10:
        continue
    index = pd.date_range(subset.index.min(), subset.index.max(), freq="7D")
    subset = subset.reindex(index, fill_value=0).reset_index().rename(columns={"index": "from_date"})
    subset["state"] = state
    subset["disease"] = disease
    parts.append(subset)
    raw = pd.concat(parts)

# Check that all periods begin on a Sunday.
np.testing.assert_array_equal(raw.from_date.dt.weekday, 6)
# We convert to iso weeks, but we push the week one iteration forwards. That's because
# the tycho weeks start on sundays so they really cover the next week.
raw["isoweek"] = raw.from_date.apply(lambda x: (isoweek.Week.withdate(x) + 1).isoformat())
# Create a list of mondays for plotting.
mondays = np.asarray([isoweek.Week.fromstring(x).monday() for x in np.unique(raw.isoweek)])

# Create lookup tables for the ids so we can recover interpretable labels.
id_columns = ["isoweek", "state", "disease", "from_date"]
lookup = {
    column: pd.Series(np.arange(raw[column].nunique()), np.unique(raw[column]))
    for column in id_columns
}
rlookup = {column: pd.Series(x.index, x.values) for column, x in lookup.items()}

# Create indices that we can use for indexing tensors.
for column in id_columns:
    raw[f"{column}_id"] = lookup[column].loc[raw[column]].values
```

```python
holidays_url = "https://gist.githubusercontent.com/tillahoffmann/809742bfcc835c1968be455216e6eb45/raw/3f00e50fc532b6312ec9aedbaa27afffe0f1047d/historical-federal-holidays.csv"
holidays = pd.read_csv(holidays_url, parse_dates=["date"])
holidays = holidays[(raw.from_date.min() <= holidays.date) & (holidays.date <= raw.from_date.max())]

holiday_weeks = {
    holiday: [isoweek.Week.withdate(date).isoformat() for date in subset.date]
    for holiday, subset in holidays.groupby("holiday")
}
```

```python
# Prepare the data.
raw = raw[raw.number > 0]
n_weeks = raw.isoweek_id.nunique()
n_locs = raw.state_id.nunique()
n_types = raw.disease_id.nunique()
week_id = jnp.asarray(raw.isoweek_id)
loc_id = jnp.asarray(raw.state_id)
type_id = jnp.asarray(raw.disease_id)

y = jnp.log1p(jnp.asarray(raw.number))
y_scaler = preprocessing.StandardScaler()
y = jnp.asarray(y_scaler.fit_transform(y[..., None])[..., 0])

X = jnp.stack([np.in1d(raw.isoweek, values) for values in holiday_weeks.values()]).T.astype(float)
X_scaler = preprocessing.StandardScaler()
X = jnp.asarray(X_scaler.fit_transform(X))

static_data = {
    "n_weeks": n_weeks,
    "n_locs": n_locs,
    "n_types": n_types,
}
dynamic_data = {
    "week_id": week_id,
    "loc_id": loc_id,
    "type_id": type_id,
    "y": y,
    "X": X,
}
data = static_data | dynamic_data
y.size
```

```python
def model(transition_matrix, n_locs, n_types, n_weeks, loc_id, type_id, week_id, X, y):
    # Priors for precision parameters of constant effects.
    precision_prior = avb.DelayedDistribution(dists.Gamma, 1, 1)
    tau_a = numpyro.sample("tau_a", precision_prior)
    tau_b = numpyro.sample("tau_b", precision_prior)
    tau_C = numpyro.sample("tau_C", precision_prior)

    # Constant effects.
    mu = numpyro.sample("mu", avb.DelayedDistribution(PrecisionNormal, 0, 1e-8))
    a = numpyro.sample("a", avb.DelayedDistribution(PrecisionNormal, jnp.zeros(n_locs), tau_a))
    b = numpyro.sample("b", avb.DelayedDistribution(PrecisionNormal, jnp.zeros(n_types), tau_b))
    C = numpyro.sample("C", avb.DelayedDistribution(PrecisionNormal, jnp.zeros((n_locs, n_types)), tau_C))

    # Priors for precision matrices of innovation noise.
    p = transition_matrix.shape[-1]
    tau_z = numpyro.sample("tau_z", avb.DelayedDistribution(dists.Wishart, p, jnp.eye(p)))
    tau_A = numpyro.sample("tau_A", avb.DelayedDistribution(dists.Wishart, p * jnp.ones(n_locs), jnp.eye(p)))
    tau_B = numpyro.sample("tau_B", avb.DelayedDistribution(dists.Wishart, p * jnp.ones(n_types), jnp.eye(p)))

    # Temporal effects.
    z = numpyro.sample("z", avb.DelayedDistribution(LinearDynamicalSystem, transition_matrix, tau_z, n_weeks))
    A = numpyro.sample("A", avb.DelayedDistribution(LinearDynamicalSystem, transition_matrix, tau_A, n_weeks))
    B = numpyro.sample("B", avb.DelayedDistribution(LinearDynamicalSystem, transition_matrix, tau_B, n_weeks))

    # Fixed effects.
    precision_prior = avb.DelayedDistribution(dists.Gamma, jnp.ones(X.shape[-1]), 1)
    tau_coef_loc = numpyro.sample("tau_coef_loc", precision_prior)
    tau_coef_type = numpyro.sample("tau_coef_type", precision_prior)
    coef = numpyro.sample(
        "coef",
        avb.DelayedDistribution(PrecisionNormal, 0, 1e-8 * jnp.ones(X.shape[-1])),
    )
    coef_loc = numpyro.sample(
        "coef_loc",
        avb.DelayedDistribution(PrecisionNormal, jnp.zeros((n_locs, X.shape[-1])), tau_coef_loc),
    )
    coef_type = numpyro.sample(
        "coef_type",
        avb.DelayedDistribution(PrecisionNormal, jnp.zeros((n_types, X.shape[-1])), tau_coef_type),
    )

    # Observations.
    tau_y = numpyro.sample("tau_y", avb.DelayedDistribution(dists.Gamma, jnp.ones((n_locs, n_types)), 1))
    y = numpyro.sample(
        "y",
        avb.DelayedDistribution(
            PrecisionNormal,
            loc=mu + a[loc_id] + b[type_id] + z[week_id, 0] + C[loc_id, type_id]
            + A[loc_id, week_id, 0] + B[type_id, week_id, 0] 
            + (X * (coef + coef_loc[loc_id] + coef_type[type_id])).sum(axis=-1),
            precision=tau_y[loc_id, type_id]
        ),
        obs=y,
    )
```

```python
transition_matrix = jnp.asarray([[1., 1.], [0., 1.]])
p = transition_matrix.shape[-1]
rng = ifnt.random.JaxRandomState(17)

dynamic_data["transition_matrix"] = data["transition_matrix"] = transition_matrix

loc_scale = 0.01
var_scale = 0.1
prec_conc = 10
prec_rate = 10

approximation = {
    "mu": dists.Normal(loc_scale * rng.normal(()), jnp.ones(()) * var_scale),
    "a": dists.Normal(loc_scale * rng.normal((n_locs,)), np.ones(n_locs) * var_scale),
    "b": dists.Normal(loc_scale * rng.normal((n_types,)), jnp.ones(n_types) * var_scale),
    "z": Reshaped(
        dists.LowRankMultivariateNormal(
            loc_scale * rng.normal((n_weeks * p,)),
            loc_scale * rng.normal((n_weeks * p, int(jnp.sqrt(n_weeks * p)))),
            var_scale * jnp.ones(n_weeks * p),
        ),
        event_shape=(n_weeks, p),
    ),
    "A": Reshaped(
        dists.LowRankMultivariateNormal(
            loc_scale * rng.normal((n_locs, n_weeks * p)),
            loc_scale * rng.normal((n_locs, n_weeks * p, int(jnp.sqrt(n_weeks * p)))),
            var_scale * jnp.ones((n_locs, n_weeks * p)),
        ),
        event_shape=(n_weeks, p),
    ),
    "B": Reshaped(
        dists.LowRankMultivariateNormal(
            loc_scale * rng.normal((n_types, n_weeks * p)),
            loc_scale * rng.normal((n_types, n_weeks * p, int(jnp.sqrt(n_weeks * p)))),
            var_scale * jnp.ones((n_types, n_weeks * p)),
        ),
        event_shape=(n_weeks, p),
    ),
    "C": dists.Normal(
        loc_scale * rng.normal((n_locs, n_types)),
        jnp.ones((n_locs, n_types)) * var_scale,
    ),
    "coef": dists.Normal(
        loc_scale * rng.normal((X.shape[-1],)), 
        jnp.ones(X.shape[-1]) * var_scale,
    ),
    "coef_loc": dists.Normal(
        loc_scale * rng.normal((n_locs, X.shape[-1])), 
        jnp.ones((n_locs, X.shape[-1])) * var_scale,
    ),
    "coef_type": dists.Normal(
        loc_scale * rng.normal((n_types, X.shape[-1])), 
        jnp.ones((n_types, X.shape[-1])) * var_scale,
    ),
    "tau_coef_loc": dists.Gamma(prec_conc * jnp.ones(X.shape[-1]), prec_rate * jnp.ones(X.shape[-1])),
    "tau_coef_type": dists.Gamma(prec_conc * jnp.ones(X.shape[-1]), prec_rate * jnp.ones(X.shape[-1])),
    "tau_a": dists.Gamma(prec_conc * jnp.ones(()), prec_rate * jnp.ones(())),
    "tau_b": dists.Gamma(prec_conc * jnp.ones(()), prec_rate * jnp.ones(())),
    "tau_z": dists.Wishart(prec_conc * p, jnp.eye(p)),
    "tau_A": dists.Wishart(prec_conc * jnp.ones(n_locs) * p, jnp.ones((n_locs, 1, 1)) * jnp.eye(p) / prec_rate),
    "tau_B": dists.Wishart(prec_conc * jnp.ones(n_types) * p, jnp.ones((n_types, 1, 1)) * jnp.eye(p) / prec_rate),
    "tau_C": dists.Gamma(prec_conc * jnp.ones(()), prec_rate * jnp.ones(())),
    "tau_y": dists.Gamma(prec_conc * jnp.ones((n_locs, n_types)), prec_rate * jnp.ones((n_locs, n_types))),
}

# Obtain masks for values that have been observed at least once.
masks = {
    "A": jnp.zeros((n_locs, n_weeks)).at[loc_id, week_id].add(1) > 0,
    "B": jnp.zeros((n_types, n_weeks)).at[type_id, week_id].add(1) > 0,
    "C": jnp.zeros((n_locs, n_types)).at[loc_id, type_id].add(1) > 0,
}
```

```python
# Validate the evidence lower bound is consistent with a monte carlo estimate.
avb.infer.validate_elbo(model, approximation, 100)(jax.random.key(13), **data)
# Get unconstrained and auxiliary data for optimization.
unconstrained, aux = avb.approximation_to_unconstrained(approximation)
loss_fn = jax.jit(
    functools.partial(
        avb.infer.elbo_loss_from_unconstrained(model, aux),
        **static_data,
    )
)
loss_fn(unconstrained, **dynamic_data)
```

```python
%snakeviz avb.infer.elbo_loss_from_unconstrained(model, aux)(unconstrained, **static_data, **dynamic_data)
```

```python
# Initialize the optimizer and run one optimization step to compile.
optim = jaxopt.LBFGS(loss_fn)
state = optim.init_state(unconstrained, **dynamic_data)
update = jax.jit(optim.update)
unconstrained, state = update(unconstrained, state, **dynamic_data)
```

```python
logdir = Path("/tmp/tensorboard")
if logdir.exists():
    shutil.rmtree(logdir)
writer = tensorboardX.SummaryWriter(logdir)
```

```python
@functools.partial(jax.jit, static_argnames=["n_steps"])
def batch_update(n_steps, unconstrained, state, *args, **kwargs):
    def _target(carry, _):
        unconstrained, state = optim.update(*carry, *args, **kwargs)
        return (unconstrained, state), state.value
    carry, values = jax.lax.scan(_target, (unconstrained, state), jnp.arange(n_steps))
    return carry, values
```

```python
atol = 1e-3
batch_size = 10
max_iter = 25_000

progress = tqdm(total=max_iter)
progress.n = int(state.iter_num)
print(f"starting from iteration {progress.n}")
with progress:
    while state.iter_num < max_iter:
        (unconstrained, state), values = batch_update(batch_size, unconstrained, state, **dynamic_data)

        writer.add_scalar("optim/value", values[-1], global_step=state.iter_num)
        writer.add_scalar("optim/error", state.error, global_step=state.iter_num)

        # Report the largest gradients for each block.
        max_grads = jax.tree.map(lambda x: max(x.min(), x.max(), key=abs), state.grad)
        max_grads = tree_leaves_with_path(max_grads, sep="/")

        for key, value in max_grads:
            writer.add_scalar(f"{key}/grad", value, global_step=state.iter_num)

        # Report the largest steps for each block.
        max_steps = jax.tree.map(lambda x: max(x.min(), x.max(), key=abs), state.s_history)
        max_steps = tree_leaves_with_path(max_steps, sep="/")
        for key, value in max_steps:
            writer.add_scalar(f"{key}/step", value, global_step=state.iter_num)

        # Report the fraction where the step is negligible but the gradient is not.
        frac = jax.tree.map(
            lambda s, g: jnp.mean((jnp.abs(s).max(axis=0) < 1e-9)[jnp.abs(g) > 1e-9]),
            state.s_history,
            state.grad,
        )
        for key, value in tree_leaves_with_path(frac, sep="/"):
            writer.add_scalar(f"{key}/zero-step-frac", value, global_step=state.iter_num)

        # Check convergence.
        max_grad_key, max_grad = max(max_grads, key=lambda x: abs(x[1]))
        if abs(max_grad) < atol:
            break

        progress.update(batch_size)
        progress.set_description("; ".join([
            f"max(grad)={max_grad:.3g} ({max_grad_key})",
            f"error={state.error:.3g}",
            f"value={values[-1]:.2f}",
        ]))
```

# Results

```python
approximation = avb.approximation_from_unconstrained(unconstrained, aux)
# Revalidate elbo at the optimized parameters.
avb.infer.validate_elbo(model, approximation, 100)(jax.random.key(13), **data)
pass
```

```python
fig, axes = plt.subplots(2, 1, sharex=True)

ax = axes[0]
loc = approximation["z"].mean[..., 0]
scale = jnp.sqrt(approximation["z"].variance[..., 0])
line, = ax.plot(mondays, loc, marker=".", markersize=5)
ax.fill_between(mondays, loc - scale, loc + scale, color=line.get_color(), alpha=0.2)

ax = axes[1]
idx = 9
loc = approximation["B"].mean[idx, ..., 0]
scale = jnp.sqrt(approximation["B"].variance[idx, ..., 0])
line, = ax.plot(mondays, loc)
ax.fill_between(mondays, loc - scale, loc + scale, color=line.get_color(), alpha=0.2)

for ax in axes:
    for key, weeks in holiday_weeks.items():
        for week in weeks:
            ax.axvline(isoweek.Week.fromstring(week).monday(), color="silver", ls=":")
```

```python
fig, ax = plt.subplots()

i = 0
for key in ["a", "b", "C"]:
    x = approximation[key].mean.mean()
    ax.axhline(x, label=f"{key}={x:.3f}", color=f"C{i}")
    i += 1

for key in ["z", "A", "B"]:
    x = approximation[key].mean[..., 0]
    if x.ndim > 1:
        x = x.mean(axis=0)
    color = f"C{i}"
    ax.plot(x, color=color)
    x = x.mean()
    ax.axhline(x, color=color, ls="--", label=f"{key}={x:.3f}")
    i += 1
ax.legend(fontsize="small", ncol=2)
approximation["mu"].mean

```

```python
plt.imshow(approximation["B"].mean[..., 0], aspect="auto")
```

```python
fig, ax = plt.subplots()
prediction = (
    approximation["mu"].mean
    + approximation["a"].mean[loc_id]
    + approximation["b"].mean[type_id]
    + approximation["z"].mean[week_id, 0]
    + approximation["A"].mean[loc_id, week_id, 0]
    + approximation["B"].mean[type_id, week_id, 0]
    + approximation["C"].mean[loc_id, type_id]
    + (X * (
        approximation["coef"].mean 
        + approximation["coef_loc"].mean[loc_id] 
        + approximation["coef_type"].mean[type_id]
    )).sum(axis=-1)
)
step = 10
ax.scatter(y[::step], prediction[::step], marker=".", alpha=0.5)
ax.set_aspect("equal")
mm = y.min(), y.max()
ax.plot(mm, mm, color="silver", ls="--")
metrics.r2_score(y, prediction)
```

```python
residuals = (y - prediction) / approximation["tau_y"].mean[loc_id, type_id]
plt.hist(residuals, density=True, bins=25, range=(-2, 2))
```
