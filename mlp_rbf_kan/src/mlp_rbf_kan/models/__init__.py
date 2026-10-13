# payloads/__init__.py

# --- Main classes imports ---
from .mlp_rbfkan import MLP_RBFKAN
from .rbf_kan import RBF_KAN
from .mlp import MLP
from .kan import KAN
from .train import train_classification, train_regression

# --- Module's version ---
__version__ = "0.1.0"


__all__ = [
    "MLP_RBFKAN",
    "RBF_KAN",
    "KAN",
    "MLP",
    "train_regression",
    "train_classification",
]