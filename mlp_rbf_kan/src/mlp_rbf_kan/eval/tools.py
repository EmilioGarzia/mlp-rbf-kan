import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def _flatten(u):
    if isinstance(u, torch.Tensor):
        return u.reshape(-1)
    else:
        return torch.tensor(u).reshape(-1)

def relative_l2_error(u_pred, u_true):
    """
    Relative L2 error:
    ||u_pred - u_true||_2 / ||u_true||_2
    """
    u_pred = _flatten(u_pred)
    u_true = _flatten(u_true)

    numerator = torch.norm(u_pred - u_true, p=2)
    denominator = torch.norm(u_true, p=2)

    return (numerator / denominator).item()

def linf_error(u_pred, u_true):
    """
    L-infinity error:
    max |u_pred - u_true|
    """
    u_pred = _flatten(u_pred)
    u_true = _flatten(u_true)

    return torch.max(torch.abs(u_pred - u_true)).item()

def rmse(u_pred, u_true):
    """
    Root Mean Squared Error
    """
    u_pred = _flatten(u_pred)
    u_true = _flatten(u_true)

    return torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()

def mse(u_pred, u_true):
    """
    Mean Squared Error
    """
    u_pred = _flatten(u_pred)
    u_true = _flatten(u_true)

    return torch.mean((u_pred - u_true) ** 2).item()

def measure_training_time(train_fn, *args, **kwargs):
    """
    Measures execution time of a training function
    
    Usage: losses, train_time = measure_training_time(train, model, x, y)
    """
    start = time.time()
    result = train_fn(*args, **kwargs)
    end = time.time()

    return result, end - start

def error_map(u_pred, u_true):
    """
    Pointwise absolute error |u_pred - u_true|
    Shape preserved
    """
    return torch.abs(u_pred - u_true)

def extract_time_slice(U, t_idx):
    """
    Extract u(x, t_idx)
    U shape: [nt, nx]
    Useful for PDE
    """
    return U[t_idx, :]

def compute_all_metrics(u_pred, u_true):
    return {
        "relative_L2": relative_l2_error(u_pred, u_true),
        "L_inf": linf_error(u_pred, u_true),
        "RMSE": rmse(u_pred, u_true),
        "MSE": mse(u_pred, u_true)
    }


def plot_rbf_neurons(rbf_layer, neuron_ids=None, x_range=(-3, 3), n=400):
    device = rbf_layer.centers.device

    if neuron_ids is None:
        neuron_ids = range(min(5, rbf_layer.coeffs.shape[0]))

    x = torch.linspace(x_range[0], x_range[1], n, device=device)

    plt.figure(figsize=(8, 5))

    for j in neuron_ids:
        x_exp = x.unsqueeze(-1)                 # [n, 1]
        c = rbf_layer.centers.view(1, -1)       # [1, C]
        w = rbf_layer.widths.view(1, -1)        # [1, C]
        a = rbf_layer.coeffs[j].view(1, -1)     # [1, C]

        rbf = torch.exp(-(x_exp - c) ** 2 / (2 * w ** 2))
        y = torch.sum(a * rbf, dim=1)

        plt.plot(
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            label=f'Neuron {j}'
        )

    plt.title("RBF-KAN learned activations (per neuron)")
    plt.xlabel("x")
    plt.ylabel("φ(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_rbf_activations(activation_layer, feature_name):
    centers = activation_layer.centers.detach().cpu().numpy()
    coeffs = activation_layer.coeffs.detach().cpu().numpy()
    
    x = np.linspace(centers.min() - 1, centers.max() + 1, 200)
    plt.figure(figsize=(10, 5))
    
    for i in range(len(centers)):
        y = coeffs[0, i] * np.exp(-(x - centers[i])**2 / (2 * 0.5**2))
        plt.plot(x, y, label=f'Center {centers[i]:.2f}', alpha=0.6)
    
    plt.title(f"XAI: Learned RBF basis for the feature: {feature_name}")
    plt.legend(ncol=2, fontsize='small')
    plt.show()

def plot_losses(loss_dict, title="Loss trend"):
    plt.figure(figsize=(8,5))

    for name, losses in loss_dict.items():
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        plt.plot(losses, label=name)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_eq_solution(
    U,
    title,
    nx,
    nt,
    x_domain,
    t_domain,
    cmap="RdBu_r"
):
    if isinstance(U, torch.Tensor):
        U = U.detach().cpu().numpy()

    if U.ndim == 1 or U.shape == (nx * nt, 1):
        U = U.reshape(nt, nx)

    plt.figure(figsize=(6, 4))
    plt.imshow(
        U,
        extent=[x_domain[0], x_domain[1], t_domain[0], t_domain[1]],
        origin="lower",
        aspect="auto",
        cmap=cmap
    )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_solutions_grid(models_dict, nx, nt, title="PDE Solutions"):
    """
    Plot multiple PDE solutions in a grid (max 3 columns per row)
    with a shared color scale.

    Parameters
    ----------
    models_dict : dict
        Dictionary {name: tensor}, where tensor has shape (nt*nx,)
    nx : int
        Number of spatial points
    nt : int
        Number of temporal points
    title : str
        Global figure title
    """

    # Convert tensors to numpy and reshape
    solutions = {
        name: U.detach().cpu().numpy().reshape(nt, nx)
        for name, U in models_dict.items()
    }

    n_models = len(solutions)
    n_cols = min(3, n_models)
    n_rows = math.ceil(n_models / n_cols)

    # Shared color scale
    vmin = min(U.min() for U in solutions.values())
    vmax = max(U.max() for U in solutions.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    # Plot each solution
    for idx, (name, U) in enumerate(solutions.items()):
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]
        im = ax.imshow(
            U,
            extent=[-1, 1, 0, 1],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.axis("on")

    # Single colorbar on the right
    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.02,
        pad=0.04
    )
    cbar.set_label("u(x,t)", fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

def plot_error(U_pred, U_exact, title, nx, nt):
    U_pred = U_pred.detach().cpu().numpy()
    U_exact = U_exact.detach().cpu().numpy()
    error = np.abs(U_pred - U_exact)
    plt.figure(figsize=(6,4))
    plt.imshow(
        error.reshape(nt, nx),
        extent=[-1,1,0,1],
        origin='lower',
        aspect='auto',
        cmap='hot'
    )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.colorbar()
    plt.show()

def evaluate_classification(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.dim() > 2:
                xb = xb.view(xb.size(0), -1)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds)
    }


def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

def plot_accuracy_trend(model, train_loader, test_loader, epochs=50, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_acc = []
    test_acc = []

    for _ in range(epochs):
        model.eval()
        # Train accuracy
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_acc.append(accuracy_score(all_labels, all_preds))
        
        # Test accuracy
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        test_acc.append(accuracy_score(all_labels, all_preds))

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Trend: {model.__class__.__name__}")
    plt.legend()
    plt.grid(True)
    plt.show()

def n_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)