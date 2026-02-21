from .automodel import AutoModel
from .datasets import (
    SyntheticDatasetGenerator,
    list_supported_distributions,
    make_classification_data,
    make_distribution_data,
    make_mathematical_synthetic_data,
    make_regression_data,
)
from .helpers import train_model

__all__ = [
    "AutoModel",
    "train_model",
    "make_classification_data",
    "make_regression_data",
    "SyntheticDatasetGenerator",
    "make_distribution_data",
    "make_mathematical_synthetic_data",
    "list_supported_distributions",
]
__version__ = "0.2.0"
