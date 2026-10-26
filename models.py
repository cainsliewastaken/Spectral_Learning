"""Network architectures for the 1D Poisson operator-learning experiment."""

import torch
import torch.nn as nn


class LinearDense(nn.Module):
    def __init__(self, N, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(N, width, bias=False)]
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width, bias=False))
        layers.append(nn.Linear(width, N, bias=False))
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class GELUDense(nn.Module):
    def __init__(self, N, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(N, width), nn.GELU()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(width, N))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IdealDiagonalSpectralOperator(nn.Module):
    def __init__(self, N, S, K_retained=32):
        super().__init__()
        self.N = N
        self.K_retained = K_retained
        self.register_buffer("S", S.detach().clone())
        self.gains = nn.Parameter(torch.zeros(K_retained, dtype=self.S.dtype))

    def forward(self, x):
        coeffs = x @ self.S
        out_coeffs = torch.zeros_like(coeffs)
        out_coeffs[:, :self.K_retained] = (
            coeffs[:, :self.K_retained] * self.gains.view(1, -1)
        )
        return out_coeffs @ self.S.T


class LinearFNO1D(nn.Module):
    def __init__(self, N, S, width=16, layers=3, K_retained=32):
        super().__init__()
        self.N = N
        self.width = width
        self.layers = layers
        self.K_retained = K_retained
        self.register_buffer("S", S.detach().clone())
        self.lift = nn.Linear(1, width)
        self.local_layers = nn.ModuleList(
            [nn.Linear(width, width, bias=True) for _ in range(layers)]
        )
        self.spec_weights = nn.ParameterList([
            nn.Parameter(0.05 * torch.randn(K_retained, width, width, dtype=self.S.dtype))
            for _ in range(layers)
        ])
        self.proj = nn.Linear(width, 1)

    def forward(self, x):
        h = self.lift(x.unsqueeze(-1))
        for ell in range(self.layers):
            local = self.local_layers[ell](h)
            coeffs = torch.einsum("nm,bmc->bnc", self.S.T, h)
            spec_low = torch.einsum(
                "bkc,kco->bko", coeffs[:, :self.K_retained, :], self.spec_weights[ell]
            )
            coeffs_out = torch.zeros_like(coeffs)
            coeffs_out[:, :self.K_retained, :] = spec_low
            spec = torch.einsum("nm,bmc->bnc", self.S, coeffs_out)
            h = local + spec
        return self.proj(h).squeeze(-1)


class NonlinearFNO1D(nn.Module):
    def __init__(self, N, S, width=16, layers=3, K_retained=32):
        super().__init__()
        self.N = N
        self.width = width
        self.layers = layers
        self.K_retained = K_retained
        self.register_buffer("S", S.detach().clone())
        self.lift = nn.Linear(1, width)
        self.local_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(layers)]
        )
        self.spec_weights = nn.ParameterList([
            nn.Parameter(0.05 * torch.randn(K_retained, width, width, dtype=self.S.dtype))
            for _ in range(layers)
        ])
        self.proj = nn.Linear(width, 1)

    def forward(self, x):
        h = self.lift(x.unsqueeze(-1))
        for ell in range(self.layers):
            local = self.local_layers[ell](h)
            coeffs = torch.einsum("nm,bmc->bnc", self.S.T, h)
            spec_low = torch.einsum(
                "bkc,kco->bko", coeffs[:, :self.K_retained, :], self.spec_weights[ell]
            )
            coeffs_out = torch.zeros_like(coeffs)
            coeffs_out[:, :self.K_retained, :] = spec_low
            spec = torch.einsum("nm,bmc->bnc", self.S, coeffs_out)
            h = torch.nn.functional.gelu(local + spec)
        return self.proj(h).squeeze(-1)
