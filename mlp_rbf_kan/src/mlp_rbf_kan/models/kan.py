import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, width, num_knots=5, k=3, domain=(0.0, 1.0)):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(width) - 1):
            self.layers.append(
                KANLayer(width[i], width[i + 1], num_knots=num_knots, k=k, domain=domain)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def update_grid(self, x):
        """Richiama l'aggiornamento della griglia su tutti i layer"""
        with torch.no_grad():
            for layer in self.layers:
                # Passiamo l'input attraverso il layer per vedere come si distribuisce
                # e aggiorniamo i nodi per il prossimo step
                x = layer.update_grid_from_samples(x)

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_knots=5, k=3, domain=(0.0, 1.0)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.num_knots = num_knots

        # Componente Spline
        self.basis = BSplineBasis(num_knots=num_knots, k=k, domain=domain)
        num_basis = num_knots + k - 1
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, num_basis) * 0.1)

        # Componente Base (SiLU)
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.spline_scaler = nn.Parameter(torch.ones(out_dim, in_dim) * 0.1)

    def forward(self, x):
        # Parte Base
        base_output = F.linear(F.silu(x), self.base_weight)

        # Parte Spline (Vettorizzata)
        x_flat = x.view(-1)
        basis_matrices = self.basis(x_flat)
        basis_matrices = basis_matrices.view(x.shape[0], self.in_dim, -1)

        spline_output = torch.einsum('bin,oin->bo', basis_matrices, self.spline_weight * self.spline_scaler.unsqueeze(-1))
        return base_output + spline_output

    def update_grid_from_samples(self, x):
        """
        Adatta i nodi (knots) alla distribuzione reale degli input 'x'.
        Questo impedisce che le spline finiscano in 'zone morte'.
        """
        batch_size = x.shape[0]
        # Calcoliamo i nuovi nodi basandoci sui percentili degli input correnti
        x_sorted, _ = torch.sort(x, dim=0)
        
        # Campioniamo num_knots punti dalla distribuzione degli input
        indices = torch.linspace(0, batch_size - 1, self.num_knots).long()
        new_knots_base = x_sorted[indices].T # (in_dim, num_knots)
        
        # Aggiorniamo la base spline con i nuovi nodi mediati
        self.basis.update_knots(new_knots_base.mean(dim=0))
        
        # Restituiamo l'output per permettere la ricorsione della funzione update_grid
        return self.forward(x)

class BSplineBasis(nn.Module):
    def __init__(self, num_knots=5, k=3, domain=(0.0, 1.0)):
        super().__init__()
        self.k = k
        self.num_knots = num_knots
        self.domain = domain
        self._generate_initial_knots(domain)

    def _generate_initial_knots(self, domain):
        h = (domain[1] - domain[0]) / (self.num_knots - 1)
        knots = torch.linspace(domain[0] - self.k * h, domain[1] + self.k * h, self.num_knots + 2 * self.k)
        self.register_buffer("knots", knots)

    def update_knots(self, new_knots_base):
        """Aggiorna i buffer dei nodi mantenendo i margini per il grado k"""
        h = (new_knots_base[-1] - new_knots_base[0]) / (self.num_knots - 1)
        # Estendiamo i nodi per gestire il supporto delle spline di grado k
        left_padding = torch.linspace(new_knots_base[0] - self.k * h, new_knots_base[0] - h, self.k)
        right_padding = torch.linspace(new_knots_base[-1] + h, new_knots_base[-1] + self.k * h, self.k)
        
        updated_knots = torch.cat([left_padding.to(new_knots_base.device), 
                                   new_knots_base, 
                                   right_padding.to(new_knots_base.device)])
        self.knots.copy_(updated_knots)

    def forward(self, x):
        x = x.unsqueeze(-1)
        knots = self.knots
        bases = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for j in range(1, self.k + 1):
            left_denom = knots[j:-1] - knots[:-j-1]
            right_denom = knots[j+1:] - knots[1:-j]
            left = (x - knots[:-j-1]) / (left_denom + 1e-8) * bases[:, :-1]
            right = (knots[j+1:] - x) / (right_denom + 1e-8) * bases[:, 1:]
            bases = left + right
        return bases