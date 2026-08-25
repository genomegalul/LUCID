from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticLinear(nn.Module):
    """
    Gaussian-weight linear layer. Can sample weights during forward passes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rho_init: float = -5.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters(rho_init=rho_init)

    def reset_parameters(self, rho_init: float) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, rho_init)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1.0 / math.sqrt(fan_in)

        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, rho_init)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)

            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class RetrievalNet(nn.Module):
    """
    Probabilistic spot-parameter retrieval network.

    For a fixed n_spots model, predicts raw parameters:

        [lat_raw_1, lon_raw_1, radius_raw_1,
         ...
         lat_raw_N, lon_raw_N, radius_raw_N,
         contrast_raw]

    The raw parameters are later squashed to physical values in train.py
    before being passed to the JAX simulator.

    Inputs:
        lc_in: shape (batch, 1, datapoints)
        aux:   shape (batch, 2), [sin(inclination), cos(inclination)]

    Output:
        raw_params: shape (batch, n_spots * 3 + 1)
    """

    def __init__(
        self,
        n_spots: int,
        input_length: int,
        aux_dim: int = 2,
        dropout_p: float = 0.10,
        rho_init: float = -5.0,
    ) -> None:
        super().__init__()

        if n_spots < 1:
            raise ValueError(f"n_spots must be >= 1, got {n_spots}")

        if input_length < 128:
            raise ValueError(f"input_length should be >= 128, got {input_length}")

        self.n_spots = n_spots
        self.input_length = input_length
        self.aux_dim = aux_dim
        self.dropout_p = dropout_p
        self.output_dim = n_spots * 3 + 1

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 384, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 384),
            nn.ReLU(inplace=True),

            nn.Conv1d(384, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(8),
        )

        conv_out = 512 * 8

        self.fc1 = StochasticLinear(conv_out + aux_dim, 2048, rho_init=rho_init)
        self.fc2 = StochasticLinear(2048, 1024, rho_init=rho_init)
        self.fc3 = StochasticLinear(1024, 512, rho_init=rho_init)
        self.fc4 = StochasticLinear(512, self.output_dim, rho_init=rho_init)

    def forward(
        self,
        lc_in: torch.Tensor,
        aux: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        if lc_in.ndim != 3:
            raise ValueError(f"Expected lc_in shape (batch, 1, T), got {tuple(lc_in.shape)}")

        if lc_in.shape[-1] != self.input_length:
            raise ValueError(
                f"Expected input_length={self.input_length}, got {lc_in.shape[-1]}"
            )

        if aux.ndim != 2 or aux.shape[1] != self.aux_dim:
            raise ValueError(f"Expected aux shape (batch, {self.aux_dim}), got {tuple(aux.shape)}")

        z = self.conv(lc_in)
        z = z.flatten(1)

        z = torch.cat([z, aux], dim=1)

        z = F.relu(self.fc1(z, sample=sample))
        z = F.dropout(z, p=self.dropout_p, training=sample)

        z = F.relu(self.fc2(z, sample=sample))
        z = F.dropout(z, p=self.dropout_p, training=sample)

        z = F.relu(self.fc3(z, sample=sample))
        z = F.dropout(z, p=self.dropout_p, training=sample)

        raw = self.fc4(z, sample=sample)
        return raw


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)