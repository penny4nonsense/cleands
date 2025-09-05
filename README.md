# cleands

**cleands** is a Python package for statistical modeling and data science that unifies regression, classification, clustering, distribution modeling, and dimension reduction under a single interface.  
It aims to provide a full open-source alternative to packages like Stata, SAS, SPSS, and MATLAB, while remaining extensible and Pythonic.

## Features

- **Formula interface**: Fit models directly from a Patsy-like formula, e.g. `"y ~ x1 + x2 + x1:x2"`.
- **Supervised learning**: Linear regression, logistic regression, Poisson regression, k-nearest neighbors, recursive partitioning trees, ensembles (bagging, random forests), shrinkage methods (lasso, ridge, elastic net), etc.
- **Classification**: Logistic, multinomial, discriminant analysis, kNN classifiers, decision trees, random forests.
- **Unsupervised learning**: k-means clustering and other clustering algorithms (planned).
- **Distributions**: Parametric probability distributions with PDF, CDF, and likelihood-based inference.
- **Utilities**: Cross-validation, bootstrap, gradient descent, Newton’s method, and more.

See the [TODO list](todo.txt) for planned additions such as QDA, SVM, DBSCAN, Gaussian mixtures, quantile regression, splines, and LDV models (Tobit, truncated).

---

## Installation

Install the latest release from PyPI:

```bash
pip install cleands
```

---

## Documentation

Full API documentation, usage guides, and examples are available here:

- [cleands Documentation Index](https://penny4nonsense.github.io/cleands/)  

*(Replace the links above with your actual GitHub Pages or Read the Docs URLs once deployed.)*

---

## Quick Start

Fit a linear regression model using formula notation:

```python
import pandas as pd
from cleands.Prediction import LeastSquaresRegressor

# Example DataFrame
df = pd.DataFrame({
    "y": [1, 2, 3, 4, 5],
    "x1": [1, 2, 3, 4, 5],
    "x2": [2, 1, 2, 1, 2]
})

# Fit model with interaction term
model = LeastSquaresRegressor("y ~ x1 + x2 + x1:x2", data=df)

print(model.tidy)   # Coefficients with std errors, t-stats, and p-values
print(model.glance) # Model summary (R², AIC, BIC, etc.)
```

Logistic and Poisson regression use the same interface:

```python
from cleands.Prediction import LogisticRegressor, PoissonRegressor

logit_model = LogisticRegressor("y ~ x1 + x2", data=df)
pois_model  = PoissonRegressor("y ~ x1 + x2", data=df)
```

k-means clustering (unsupervised):

```python
from cleands.Clustering import kMeans

kmeans = kMeans("~x1+x2", data=df, k=2)
print(kmeans.groups)   # Cluster assignments
print(kmeans.means)    # Cluster centroids
```

---

## Directory Structure

```
cleands/
│
├── base.py              # Abstract base classes (prediction, classification, clustering, distribution)
├── formula.py           # Formula parser for Patsy-like expressions
├── utils.py             # Utility functions (cross-validation, bootstrap, optimizers, etc.)
│
├── Prediction/          # Regression and supervised prediction models
│   ├── glm.py           # Least squares, logistic, Poisson regressors
│   ├── knn.py           # k-nearest neighbors regressors
│   ├── shrinkage.py     # Lasso, ridge, elastic net, etc.
│   ├── tree.py          # Recursive partitioning regressors
│   ├── ensemble.py      # Bagging, random forests
│
├── Classification/      # Classification models
│   ├── glm.py           # Logistic and multinomial classifiers
│   ├── knn.py           # kNN classifiers
│   ├── lda.py           # Linear discriminant analysis
│   ├── tree.py          # Recursive partitioning classifiers
│   ├── ensemble.py      # Bagging and random forest classifiers
│
├── Clustering/          # Unsupervised clustering
│   ├── kmeans.py        # k-means clustering
│
├── DimensionReduction/  # PCA, CCA, etc. (in progress)
├── Distribution/        # Probability distributions and tests
```

---

## Roadmap

- Stepwise model selection
- Support for splines and GAMs
- More clustering methods (DBSCAN, Gaussian mixtures, hierarchical)
- Additional LDV models (Tobit, truncated regression)
- Expanded distribution families
- Neural networks and GLM trees

---

## License

MIT License. See [LICENSE](LICENSE) for details.
