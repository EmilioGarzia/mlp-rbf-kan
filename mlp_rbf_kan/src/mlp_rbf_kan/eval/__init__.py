# --- Public API imports ---
from .tools import (
    relative_l2_error,
    linf_error,
    rmse,
    mse,
    measure_training_time,
    error_map,
    extract_time_slice,
    compute_all_metrics,
    plot_rbf_neurons,
    plot_losses,
    plot_eq_solution,
    plot_solutions_grid,
    plot_error,
    n_parameters,
    evaluate_classification,
    plot_confusion_matrix,
    plot_accuracy_trend,
    visualize_rbf_activations
)

from .experiments import ExperimentalRunner, ExperimentCfg

# --- Module version ---
__version__ = "0.1.0"

__all__ = [
    "relative_l2_error",
    "linf_error",
    "rmse",
    "mse",
    "measure_training_time",
    "error_map",
    "extract_time_slice",
    "compute_all_metrics",
    "plot_rbf_neurons",
    "plot_losses",
    "plot_eq_solution",
    "plot_solutions_grid",
    "plot_error",
    "n_parameters",
    "evaluate_classification",
    "plot_confusion_matrix",
    "plot_accuracy_trend",
    "visualize_rbf_activations",
    "ExperimentalRunner",
    "ExperimentCfg"
]
