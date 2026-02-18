from .automodel import AutoModel
from .helpers import train_model
from .datasets import make_classification_data, make_regression_data

__all__ = [
    "AutoModel",
    "train_model",
    "make_classification_data",
    "make_regression_data",
]
__version__ = "0.2.0"