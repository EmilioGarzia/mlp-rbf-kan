import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def load_california_housing_dataset(test_size=0.2, random_state=42):
    """
    Load California housing dataset
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(y_train, dtype=torch.float32),
        'test_input':  torch.tensor(X_test, dtype=torch.float32),
        'test_label':  torch.tensor(y_test, dtype=torch.float32),
        'feature_names': feature_names,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y
    }
    
    print(f"California Housing ready! Features: {len(feature_names)}")
    return dataset

def load_heat_dataset(nx=100, nt=100, test_size=0.2, random_state=42):
    """
    Generate Heat equation (Paraboliy) 1D.
    u(x, t) = sin(pi * x) * exp(-pi^2 * t)
    """
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 0.1, nt)
    X, T = np.meshgrid(x, t)

    U = np.sin(np.pi * X) * np.exp(-(np.pi**2) * T)

    inputs_raw = np.stack([X.ravel(), T.ravel()], axis=-1)
    targets_raw = U.reshape(-1, 1)

    train_input_np, test_input_np, train_label_np, test_label_np = train_test_split(
        inputs_raw, targets_raw, test_size=test_size, random_state=random_state
    )

    dataset = {
        'train_input': torch.tensor(train_input_np, dtype=torch.float32),
        'train_label': torch.tensor(train_label_np, dtype=torch.float32),
        'test_input':  torch.tensor(test_input_np, dtype=torch.float32),
        'test_label':  torch.tensor(test_label_np, dtype=torch.float32),
        'X_test': X, 
        'T_test': T,
        'nx_test': nx,
        'nt_test': nt
    }

    print(f"Heat dataset (Parabolyc) ready!")
    return dataset

def load_poisson_dataset(nx=100, ny=100, test_size=0.2, random_state=42):
    """
    Generate Elliptic equation 2D.
    Equation: nabla^2 u = f(x, y)
    Solution: u(x, y) = sin(pi * x) * sin(pi * y)
    Source term f(x, y) = -2 * pi^2 * sin(pi * x) * sin(pi * y)
    """
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    U = np.sin(np.pi * X) * np.sin(np.pi * Y)

    inputs_raw = np.stack([X.ravel(), Y.ravel()], axis=-1)
    targets_raw = U.reshape(-1, 1)

    train_input_np, test_input_np, train_label_np, test_label_np = train_test_split(
        inputs_raw, targets_raw, test_size=test_size, random_state=random_state
    )

    dataset = {
        'train_input': torch.tensor(train_input_np, dtype=torch.float32),
        'train_label': torch.tensor(train_label_np, dtype=torch.float32),
        'test_input':  torch.tensor(test_input_np, dtype=torch.float32),
        'test_label':  torch.tensor(test_label_np, dtype=torch.float32),
        'X_grid': X, 
        'Y_grid': Y,
        'nx': nx,
        'ny': ny
    }

    print(f"Poisson dataset (Elliptic) ready! Domain: [0,1]x[0,1]")
    return dataset

def load_sin_dataset(n=2000):
    """
    Generate a simple dataset based on sin function (just for DEBUG)
    """
    x = torch.linspace(-3, 3, n).unsqueeze(1)
    y = torch.sin(x)
    return x, y