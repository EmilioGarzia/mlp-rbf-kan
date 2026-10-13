import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP_RBFKAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_centers=8, domain=(-2.5,2.5), shared_basis=True, enable_silu=True):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = RBFKANActivation(hidden_dim, num_centers, shared_basis, domain=domain, enable_silu=enable_silu)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = RBFKANActivation(hidden_dim, num_centers, shared_basis, domain=domain, enable_silu=enable_silu)

        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        
        x = self.fc3(x)
        return x

class RBFKANActivation(nn.Module):
    def __init__(self, num_neurons, num_centers=8, shared_basis=True, domain=(-2.5, 2.5), enable_silu=True):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.num_centers = num_centers
        self.shared_basis = shared_basis

        # Centri e larghezze
        if shared_basis:
            self.centers = nn.Parameter(torch.linspace(domain[0], domain[1], num_centers))
            self.widths = nn.Parameter(torch.ones(num_centers) * (domain[1] - domain[0]) / num_centers)
        else:
            self.centers = nn.Parameter(
                torch.linspace(domain[0], domain[1], num_centers).repeat(num_neurons, 1)
            )
            self.widths = nn.Parameter(torch.ones(num_neurons, num_centers) * (domain[1] - domain[0]) / num_centers)

        # Coefficienti della parte RBF
        self.coeffs = nn.Parameter(torch.randn(num_neurons, num_centers) * 0.01)
        
        if enable_silu:
            self.base_scale = nn.Parameter(torch.ones(num_neurons) * 0.1)
        else:
            self.base_scale = nn.Parameter(torch.zeros(num_neurons))
        
        # MIGLIORIA 2: Scaler per la parte RBF
        self.rbf_scale = nn.Parameter(torch.ones(num_neurons))

    def forward(self, x):
        base = F.silu(x) * self.base_scale

        x_exp = x.unsqueeze(-1)  

        if self.shared_basis:
            c_exp = self.centers.view(1, 1, -1)
            w_exp = self.widths.view(1, 1, -1).abs().clamp(min=1e-4)
        else:
            c_exp = self.centers.unsqueeze(0) 
            w_exp = self.widths.unsqueeze(0).abs().clamp(min=1e-4)

        rbf = torch.exp(-(x_exp - c_exp) ** 2 / (2 * w_exp ** 2))
        rbf_out = torch.sum(rbf * self.coeffs.unsqueeze(0), dim=-1)
        return base + (self.rbf_scale * rbf_out)