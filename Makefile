.PHONY : docs doctests lint tests

all : lint tests docs doctests

lint :
	black --check .

docs :
	rm -rf docs/_build
	sphinx-build -nW . docs/_build

doctests :
	sphinx-build -b doctest . docs/_build

tests :
	pytest -v --cov=avb --cov-report=term-missing --cov-report=html
