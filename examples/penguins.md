```python
import avb
from avb.distributions import PrecisionNormal
import functools
import jax
from jax import numpy as jnp
import jaxopt
from matplotlib import pyplot as plt
import numpyro
from numpyro.distributions import Gamma, MultivariateNormal
import pandas as pd
from sklearn import preprocessing
```

```python
# Load the data from GitHub and preprocess.
penguins_path = "https://github.com/mwaskom/seaborn-data/raw/master/penguins.csv"
raw = pd.read_csv(penguins_path).dropna()

X = jnp.asarray(
    preprocessing.StandardScaler().fit_transform(
        raw.flipper_length_mm.values[..., None]
    )
)
y = jnp.asarray(
    preprocessing.StandardScaler().fit_transform(
        raw.bill_length_mm.values[..., None]
    ).squeeze()
)
species_id = preprocessing.LabelEncoder().fit_transform(raw.species)
sex_id = preprocessing.LabelEncoder().fit_transform(raw.sex)
n_species = jnp.unique(species_id).size
n_covariates = X.shape[-1]
```

```python
def model(n_species, species_id, sex_id, X, y):
    intercept = numpyro.sample(
        "intercept",
        avb.DelayedDistribution(PrecisionNormal, 0, 1e-4),
    )
    species_effect_precision = numpyro.sample(
        "species_effect_precision",
        avb.DelayedDistribution(Gamma, 1, 1),
    )
    species_effect = numpyro.sample(
        "species_effect",
        avb.DelayedDistribution(
            PrecisionNormal,
            jnp.zeros(n_species),
            species_effect_precision,
        ),
    )
    sex_effect_precision = numpyro.sample(
        "sex_effect_precision",
        avb.DelayedDistribution(Gamma, 1, 1),
    )
    sex_effect = numpyro.sample(
        "sex_effect",
        avb.DelayedDistribution(
            PrecisionNormal,
            jnp.zeros(2),
            sex_effect_precision,
        ),
    )
    coefs = numpyro.sample(
        "coefs",
        avb.DelayedDistribution(
            MultivariateNormal,
            loc=jnp.zeros(X.shape[-1]),
            precision_matrix=1e-4 * jnp.eye(X.shape[-1]),
        ),
    )
    obs_precision = species_effect_precision = numpyro.sample(
        "obs_precision",
        avb.DelayedDistribution(Gamma, 1, 1),
    )
    y_hat = intercept + species_effect[species_id] + sex_effect[sex_id] + X @ coefs
    numpyro.sample(
        "y",
        avb.DelayedDistribution(PrecisionNormal, y_hat, obs_precision),
        obs=y,
    )
```

```python
approximation = {
    "intercept": PrecisionNormal(jnp.zeros(()), jnp.ones(())),
    "species_effect_precision": Gamma(jnp.ones(()), jnp.ones(())),
    "species_effect": PrecisionNormal(jnp.zeros(n_species), jnp.ones(n_species)),
    "sex_effect_precision": Gamma(jnp.ones(()), jnp.ones(())),
    "sex_effect": PrecisionNormal(jnp.zeros(2), jnp.ones(2)),
    "coefs": MultivariateNormal(jnp.zeros(n_covariates), jnp.eye(n_covariates)),
    "obs_precision": Gamma(jnp.ones(()), jnp.ones(())),
}
partial_model = functools.partial(model, n_species, species_id, sex_id, X, y)
avb.infer.validate_elbo(partial_model, approximation)(jax.random.key(7))
unconstrained, aux = avb.approximation_to_unconstrained(approximation)
loss_fn = avb.elbo_loss_from_unconstrained(partial_model, aux)
```

```python
optim = jaxopt.LBFGS(loss_fn, maxiter=10000)
result = optim.run(unconstrained)
result.state.error
```

```python
approximation = avb.approximation_from_unconstrained(result.params, aux)
guide = avb.infer.guide_from_approximation(approximation)
predictive = numpyro.infer.Predictive(avb.nodes.materialize(model), guide=guide, num_samples=100)
samples = predictive(jax.random.key(13), n_species, species_id, sex_id, X, y=None)
```

```python
fig, ax = plt.subplots()
for i in range(n_species):
    for j, (ls, m) in enumerate(zip(["--", "-"], "os")):
        fltr = (species_id == i) & (sex_id == j)
        x = X.squeeze()[fltr]
        mm = jnp.asarray([x.min(), x.max()])

        c = f"C{i}"
        ax.scatter(x, y[fltr], facecolor=c, edgecolor="w", marker=m)
        ax.plot(
            mm,
            approximation["intercept"].mean
            + approximation["species_effect"].mean[i]
            + approximation["coefs"].mean.squeeze() * mm
            + approximation["sex_effect"].mean[j],
            ls=ls,
            color=c,
        )
```
