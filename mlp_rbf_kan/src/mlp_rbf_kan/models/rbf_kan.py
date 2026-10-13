import torch.nn as nn
import torch
import numpy as np


"""
    An implementation of RBF-KAN using RBFKANLayer
"""
class RBF_KAN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, out_dim=1, num_knots=8, domain=(-2.5,2.5)):
        super().__init__()

        self.kan1 = RBFKANLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_knots=num_knots,
            target_activation=None,
            domain_min=domain[0],
            domain_max=domain[1]
        )

        self.kan2 = RBFKANLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_knots=num_knots,
            target_activation=None,
            domain_min=domain[0],
            domain_max=domain[1]
        )

        self.kan3 = RBFKANLayer(
            in_dim=hidden_dim,
            out_dim=out_dim,
            num_knots=num_knots,
            target_activation=None,
            domain_min=domain[0],
            domain_max=domain[1]
        )

    def forward(self, x):
        x, *_ = self.kan1(x)
        x, *_ = self.kan2(x)
        x, *_ = self.kan3(x)
        return x
    

class RBFKANLayer(nn.Module):
    """
    RBF-based KAN
    """
    def __init__(self, in_dim, out_dim, num_knots=8, init_scale=0.1, adaptive=True,
                 domain_min=-2.5, domain_max=2.5, op='einsum', target_activation=None):
        super(RBFKANLayer, self).__init__()
        self.num_knots = num_knots
        self.init_scale = init_scale
        self.adaptive = adaptive
        self.target_activation = target_activation
        self.op = op
        self.in_dim = in_dim
        self.out_dim = out_dim

        if adaptive:
            self.centers = nn.Parameter(torch.linspace(domain_min, domain_max, num_knots))
        else:
            self.register_buffer('centers', torch.linspace(domain_min, domain_max, num_knots))

        # Widths (σ) for each RBF
        self.widths = nn.Parameter(torch.ones(num_knots) * 0.5)

       # Coefficients (weights) for each RBF
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_knots) * init_scale)
        self.bias = nn.Parameter(torch.zeros(out_dim)) 

        self._initialize_to_target()
        

    def _initialize_to_target(self):
        """Inizializzazione per approssimare funzione target"""
        if self.target_activation is None:
            return

        with torch.no_grad():
            eval_points = self.centers.clone()

            if self.target_activation == 'relu':
                target_vals = torch.clamp(eval_points, min=0.0)
            elif self.target_activation == 'gelu':
                target_vals = eval_points * 0.5 * (1.0 + torch.tanh(
                    np.sqrt(2.0 / np.pi) * (eval_points + 0.044715 * torch.pow(eval_points, 3))
                ))
            elif self.target_activation == 'elu':
                target_vals = torch.where(eval_points >= 0, eval_points, torch.exp(eval_points) - 1)
            else:
                return

            self.coef.data = target_vals

    def forward(self, x):
        orig_shape = x.shape
        preacts = x[:, None, :].clone().expand(orig_shape[0], self.out_dim, self.in_dim)
        x_flat = x.view(-1, 1)  # [N, 1]

        # Compute RBF: exp(-||x - c_i||² / (2σ_i²))
        centers_expanded = self.centers.view(1, -1)  # [1, num_knots]
        dists_sq = (x_flat - centers_expanded).pow(2)  # [N, num_knots]

        widths_sq = self.widths.pow(2).clamp(min=1e-6)
        rbf_values = torch.exp(-dists_sq / (2 * widths_sq.view(1, -1)))
        
        if self.op == 'einsum':
            rbf_values = rbf_values.view(*orig_shape, -1)
            output_flat = torch.einsum('ijk,jlk->ijl', rbf_values, self.coef).sum(dim=1)
        
        else:
            rbf_values = rbf_values.view(*orig_shape, -1).view(orig_shape[0], -1)
            coef = self.coef.permute(0, 2, 1).reshape(-1, self.coef.shape[1])
            output_flat = (rbf_values @ coef) + self.bias
        
        postspline = output_flat.clone()
        postacts = postspline.clone()

        return output_flat, preacts, postacts, postspline

    def get_function_approximation(self, x_range):
        with torch.no_grad():
            return self.forward(x_range).cpu().numpy()

    def get_mathematical_properties(self, x_range):
        with torch.no_grad():
            x_tensor = x_range.to(self.centers.device)
            y_vals = self.forward(x_tensor)

            h = (x_tensor[1] - x_tensor[0]).item()
            dy_dx = torch.gradient(y_vals, spacing=h)[0]
            d2y_dx2 = torch.gradient(dy_dx, spacing=h)[0]
            effective_radius = torch.mean(self.widths).item()
            center_spacing = torch.mean(torch.diff(torch.sort(self.centers)[0])).item()

            return {
                'values': y_vals.cpu().numpy(),
                'derivative': dy_dx.cpu().numpy(),
                'curvature': d2y_dx2.cpu().numpy(),
                'effective_radius': effective_radius,
                'center_spacing': center_spacing,
                'num_active_centers': torch.sum(torch.abs(self.coef) > 1e-3).item()
            }