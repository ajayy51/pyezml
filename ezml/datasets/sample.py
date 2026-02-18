import pandas as pd
from sklearn.datasets import make_classification, make_regression


def make_classification_data(
    n_samples=500,
    n_features=10,
    n_classes=2,
    random_state=42,
    target_name="target",
):
    """
    Generate a synthetic classification dataset.

    Returns
    -------
    pandas.DataFrame
        Feature columns + target column.
    """

    # Guard: sklearn requires informative features >= log2(classes)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_name] = y

    return df


def make_regression_data(
    n_samples=500,
    n_features=10,
    noise=0.1,
    random_state=42,
    target_name="target",
):
    """
    Generate a synthetic regression dataset.

    Returns
    -------
    pandas.DataFrame
        Feature columns + target column.
    """

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )

    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_name] = y

    return df