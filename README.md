# MLP with RBF-KAN library

A small PyTorch library that implements a **Multi-Layer Perceptron whose activation functions are learnable RBF-KAN units** (`MLP_RBFKAN`): every hidden neuron applies a sum of Gaussian radial basis functions with learnable centers, widths and coefficients, optionally added to a scaled SiLU base term. Instead of fixing a nonlinearity such as ReLU, the network learns the activation shape of each neuron, as in a Kolmogorov–Arnold Network, but keeps the cost and simplicity of a standard MLP.

The library also contains the baselines used for comparison (classical MLP, B-spline KAN, pure RBF-KAN), dataset generators for PDE regression problems, training loops, metrics, plotting utilities and a multi-seed experiment runner. The notebooks in this folder reproduce the experiments of the paper:

| Notebook | Task |
| --- | --- |
| `MLP_RBF_KAN_HEAT.ipynb` | Regression of the 1D heat equation solution (parabolic PDE) |
| `MLP_RBF_KAN_POISSON.ipynb` | Regression of the 2D Poisson equation solution (elliptic PDE) |
| `MLP_RBF_KAN_IRIS.ipynb` | Classification on Iris |
| `MLP_RBF_KAN_MNIST.ipynb` | Classification on MNIST |
| `MLP_RBF_KAN_XAI.ipynb` | Explainability on California Housing (learned activations, feature importance, PDP/ICE, SHAP) |

## Paper

This library accompanies the paper:

> P. De Luca, E. Di Nardo, E. Garzia, L. Marcellino, A. Ciaramella. **Intrinsic and Post-hoc Interpretability of Gaussian-Based Kolmogorov–Arnold Networks.** In *Computational Science and Its Applications – ICCSA 2026 Workshops*, Lecture Notes in Computer Science, pp. 173–183. Springer, Cham, 2026. [https://doi.org/10.1007/978-3-032-30530-5_11](https://link.springer.com/chapter/10.1007/978-3-032-30530-5_11)

If you use this code, please cite:

```bibtex
@inproceedings{deluca2026interpretability,
  title     = {Intrinsic and Post-hoc Interpretability of Gaussian-Based Kolmogorov--Arnold Networks},
  author    = {De Luca, Pasquale and Di Nardo, Emanuel and Garzia, Emilio and Marcellino, Livia and Ciaramella, Angelo},
  booktitle = {Computational Science and Its Applications -- ICCSA 2026 Workshops},
  series    = {Lecture Notes in Computer Science},
  pages     = {173--183},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-30530-5_11}
}
```

The MNIST dataset is not included in the repository: `MLP_RBF_KAN_MNIST.ipynb` downloads it automatically into `./data` on first run.

## How to install

```shell
cd mlp_rbf_kan
pip install -e .
```

Then:

```python
from mlp_rbf_kan.models import MLP_RBFKAN, MLP, KAN, RBF_KAN, train_regression, train_classification
from mlp_rbf_kan.datasets import load_heat_dataset, load_poisson_dataset, load_california_housing_dataset
from mlp_rbf_kan.eval import compute_all_metrics, ExperimentCfg, ExperimentalRunner
```

### Quick example

```python
import torch
from mlp_rbf_kan.models import MLP_RBFKAN, train_regression
from mlp_rbf_kan.datasets import load_heat_dataset
from mlp_rbf_kan.eval import compute_all_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_heat_dataset()

model = MLP_RBFKAN(in_dim=2, hidden_dim=64, out_dim=1, num_centers=8, domain=(0, 1))

train_loss, test_loss, _ = train_regression(
    model,
    dataset["train_input"], dataset["train_label"],
    dataset["test_input"], dataset["test_label"],
    epochs=50, lr=0.5, optimizer_type="lbfgs", batch_size=-1, device=device,
)

with torch.no_grad():
    u_pred = model(dataset["test_input"].to(device)).cpu()
print(compute_all_metrics(u_pred, dataset["test_label"]))
```

## Dependencies

The package requires Python >= 3.8. The dependencies are not declared in `pyproject.toml`, so they have to be installed manually:

```shell
pip install torch numpy scikit-learn matplotlib seaborn tqdm
```

| Package | Used for |
| --- | --- |
| `torch` | Models, training, metrics |
| `numpy` | Dataset generation, result aggregation |
| `scikit-learn` | Train/test split, scaling, California Housing, classification metrics |
| `matplotlib`, `seaborn` | Plots and confusion matrices |
| `tqdm` | Training progress bars |

The notebooks additionally use `torchvision` (MNIST download) and `shap` (XAI notebook):

```shell
pip install torchvision shap jupyter
```

## Glossary

The package is organized in three submodules: `models`, `datasets` and `eval`.

### Models

`mlp_rbf_kan.models`

| Name | Description |
| --- | --- |
| `MLP_RBFKAN(in_dim, hidden_dim, out_dim, num_centers=8, domain=(-2.5, 2.5), shared_basis=True, enable_silu=True)` | The proposed model: 3 linear layers with an `RBFKANActivation` after each of the two hidden layers. |
| `RBFKANActivation(num_neurons, num_centers=8, shared_basis=True, domain=(-2.5, 2.5), enable_silu=True)` | Learnable activation `φ(x) = s_base · SiLU(x) + s_rbf · Σ_k a_k · exp(-(x - c_k)² / 2σ_k²)`. With `shared_basis=True` centers `c_k` and widths `σ_k` are shared by all neurons of the layer (only the coefficients `a_k` are per neuron); with `False` every neuron has its own basis. `enable_silu=False` disables the SiLU base term. |
| `MLP(in_dim, hidden_dim, out_dim)` | Classical MLP baseline with two ReLU hidden layers. |
| `KAN(width, num_knots=5, k=3, domain=(0.0, 1.0))` | B-spline KAN baseline (SiLU base + B-spline of degree `k`). `width` is the list of layer sizes, e.g. `[2, 64, 64, 1]`. Supports adaptive grid update via `update_grid(x)`. |
| `RBF_KAN(in_dim=1, hidden_dim=32, out_dim=1, num_knots=8, domain=(-2.5, 2.5))` | Pure RBF-KAN baseline with three `RBFKANLayer`, where Gaussian RBFs replace B-splines on every edge. |
| `train_regression(model, x_train, y_train, x_test, y_test, epochs=100, lr=1e-3, batch_size=128, optimizer_type="adam", ...)` | MSE training loop. `optimizer_type` is `"adam"` or `"lbfgs"`; `batch_size=-1` means full batch. Returns `(train_rmse_history, test_rmse_history, train_predictions)`. |
| `train_classification(model, train_loader, test_loader, epochs=100, lr=1e-3, optimizer_name="adam", device=None)` | Cross-entropy training loop on `DataLoader`s. Returns `(train_losses, test_losses)`. |

### Datasets

`mlp_rbf_kan.datasets`

Every loader returns a dict with `train_input`, `train_label`, `test_input`, `test_label` as `float32` tensors, plus some extra fields.

| Name | Description |
| --- | --- |
| `load_heat_dataset(nx=100, nt=100, test_size=0.2, random_state=42)` | 1D heat equation (parabolic), exact solution `u(x, t) = sin(πx) · exp(-π²t)` on `x ∈ [0, 1]`, `t ∈ [0, 0.1]`. Extra fields: `X_test`, `T_test`, `nx_test`, `nt_test`. |
| `load_poisson_dataset(nx=100, ny=100, test_size=0.2, random_state=42)` | 2D Poisson equation (elliptic) `∇²u = -2π² sin(πx) sin(πy)` on `[0, 1]²`, exact solution `u(x, y) = sin(πx) · sin(πy)`. Extra fields: `X_grid`, `Y_grid`, `nx`, `ny`. |
| `load_california_housing_dataset(test_size=0.2, random_state=42)` | California Housing regression, standardized inputs and target. Extra fields: `feature_names`, `scaler_x`, `scaler_y`. |
| `load_sin_dataset(n=2000)` | `y = sin(x)` on `[-3, 3]`, for debugging. Returns `(x, y)`. |

### Eval

`mlp_rbf_kan.eval`

**Metrics**

| Name | Description |
| --- | --- |
| `relative_l2_error(u_pred, u_true)` | `‖u_pred - u_true‖₂ / ‖u_true‖₂` |
| `linf_error(u_pred, u_true)` | `max |u_pred - u_true|` |
| `rmse(u_pred, u_true)`, `mse(u_pred, u_true)` | Root mean squared error and mean squared error |
| `compute_all_metrics(u_pred, u_true)` | Dict with `relative_L2`, `L_inf`, `RMSE`, `MSE` |
| `error_map(u_pred, u_true)` | Pointwise absolute error |
| `evaluate_classification(model, loader, device=None)` | Dict with `accuracy`, `precision`, `recall`, `f1` (weighted) and `confusion_matrix` |
| `n_parameters(model)` | Number of trainable parameters |
| `measure_training_time(train_fn, *args, **kwargs)` | Returns `(result, seconds)` |

**Plots**

| Name | Description |
| --- | --- |
| `plot_losses(loss_dict, title)` | Loss curves of several models |
| `plot_eq_solution(...)`, `plot_solutions_grid(models_dict, nx, nt, title)` | Predicted PDE solutions |
| `plot_error(U_pred, U_exact, title, nx, nt)` | Error map of a PDE solution |
| `extract_time_slice(U, t_idx)` | Slice `u(x, t_idx)` of a `[nt, nx]` solution |
| `plot_confusion_matrix(cm, classes, model_name)` | Confusion matrix heatmap |
| `plot_accuracy_trend(model, train_loader, test_loader, epochs=50, device=None)` | Train/test accuracy over epochs |
| `plot_rbf_neurons(rbf_layer, neuron_ids=None, x_range=(-3, 3), n=400)` | Learned activation functions of an `RBFKANActivation` (e.g. `model.act1`) |
| `visualize_rbf_activations(activation_layer, feature_name)` | Learned RBF basis for a single feature (XAI) |

**Experiment runner**

`ExperimentCfg` holds the configuration of an experiment (model factories, seeds, hyperparameters, optimizers, device, and either a regression `dataset` dict or classification `train_loader`/`test_loader`). `ExperimentalRunner` trains and evaluates every model on every seed and reports mean ± std.

```python
from mlp_rbf_kan.models import MLP, MLP_RBFKAN
from mlp_rbf_kan.datasets import load_poisson_dataset
from mlp_rbf_kan.eval import ExperimentCfg, ExperimentalRunner

cfg = ExperimentCfg(
    models={
        "MLP": lambda: MLP(2, 64, 1),
        "MLP_RBFKAN": lambda: MLP_RBFKAN(2, 64, 1, num_centers=8, domain=(-1, 1)),
    },
    seeds=[32, 23, 54, 2, 45],
    epochs=20,
    device="cuda",
    dataset=load_poisson_dataset(),
)

runner = ExperimentalRunner(cfg)
runner.run_battery_experiments()   # prints mean ± std for every model and metric
runner.plot_metrics_summary(log_scale=True)
```

Models named `"MLP"` (case insensitive) are trained with `mlp_lr` / `mlp_optimizer` (default Adam, `1e-3`), all the others with `kan_lr` / `kan_optimizer` (default L-BFGS, `0.5`).

## Author

*Emilio Garzia*
