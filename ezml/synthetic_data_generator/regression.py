import numpy as np
from .distributions import generate_distribution

def generate_regression_data(n_samples=100, n_features=1, noise=0.1, dist='normal'):
    """
    Generate synthetic regression dataset.

    Returns:
    - X: feature matrix (n_samples, n_features)
    - y: target vector (n_samples,)
    """
    X = np.column_stack([generate_distribution(dist, n_samples) for _ in range(n_features)])
    coef = np.random.uniform(-10, 10, size=n_features)
    y = X.dot(coef) + np.random.normal(0, noise, size=n_samples)
    return X, y
