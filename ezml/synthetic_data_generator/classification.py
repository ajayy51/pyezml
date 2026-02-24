import numpy as np
from .distributions import generate_distribution
from sklearn.preprocessing import LabelBinarizer

def generate_classification_data(n_samples=100, n_features=2, n_classes=2, dist='normal'):
    """
    Generate synthetic classification dataset.

    Returns:
    - X: feature matrix (n_samples, n_features)
    - y: class labels (n_samples,)
    """
    X = np.column_stack([generate_distribution(dist, n_samples) for _ in range(n_features)])
    
    # Simple linear decision boundary for classes
    coef = np.random.uniform(-5, 5, size=n_features)
    logits = X.dot(coef)
    
    # Assign classes based on thresholds
    thresholds = np.linspace(min(logits), max(logits), n_classes + 1)[1:-1]
    y = np.digitize(logits, thresholds)
    
    return X, y
