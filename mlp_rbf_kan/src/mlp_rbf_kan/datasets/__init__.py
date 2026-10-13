# payloads/__init__.py

# --- Main classes imports ---
from .loader import( 
    load_sin_dataset, 
    load_heat_dataset, 
    load_poisson_dataset, 
    load_california_housing_dataset
)


# --- Module's version ---
__version__ = "0.1.0"


__all__ = [
    "load_sin_dataset",
    "load_heat_dataset",
    "load_poisson_dataset",
    "load_california_housing_dataset"
]