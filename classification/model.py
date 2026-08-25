from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpotCountCNN(nn.Module):
    """
    Variable-length light-curve classifier with an auxiliary count-regression head.

    Works with power-of-2 input lengths such as:
        128, 256, 512, 1024, ...

    Inputs:
        x:   shape (batch, input_length) or (batch, 1, input_length)
        inc: shape (batch,) or (batch, 1), in degrees

    Main output:
        logits: shape (batch, 5)

    Auxiliary output:
        count_pred: shape (batch,), continuous predicted spot count in [1, 5]
    """

    def __init__(
        self,
        input_length: int = 256,
        n_classes: int = 5,
        fft_bins: int | None = None,
    ):
        super().__init__()

        if input_length < 128:
            raise ValueError(f"input_length should be at least 128, got {input_length}")

        if input_length & (input_length - 1) != 0:
            raise ValueError(f"input_length must be a power of 2, got {input_length}")

        self.input_length = input_length
        self.n_classes = n_classes

        max_fft_bins = input_length // 2
        if fft_bins is None:
            # Keep useful low-frequency structure, but avoid feeding the full FFT
            # for long curves.
            fft_bins = min(128, max(32, input_length // 8))

        if fft_bins > max_fft_bins:
            raise ValueError(
                f"fft_bins={fft_bins} is too large for input_length={input_length}; "
                f"maximum is {max_fft_bins}"
            )

        self.fft_bins = fft_bins

        # Number of stride-2 downsampling stages.
        #
        # 128  -> 2 stages
        # 256  -> 3 stages
        # 512  -> 4 stages
        # 1024 -> 5 stages
        n_downsample = max(2, int(math.log2(input_length)) - 5)

        channels = [1, 32, 64, 96, 128, 128, 128]
        kernels = [15, 11, 7, 5, 3, 3]

        time_layers = []
        in_ch = channels[0]

        for i in range(n_downsample):
            out_ch = channels[i + 1]
            kernel = kernels[i]
            padding = kernel // 2

            time_layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                    ),
                    self._make_group_norm(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch

        # Final non-strided refinement layer.
        time_layers.extend(
            [
                nn.Conv1d(in_ch, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(inplace=True),
            ]
        )

        self.time_features = nn.Sequential(*time_layers)

        self.time_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.fft_features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        self.fft_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Feature sizes:
        # time branch: 128
        # FFT branch:   64
        # inclination:   1
        # stats:         2   -> std + peak-to-peak
        feature_dim = 128 + 64 + 1 + 2

        self.shared_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
        )

        self.class_head = nn.Linear(128, n_classes)

        # Raw scalar. We squash it to [1, 5] in forward_with_count().
        self.count_head = nn.Linear(128, 1)

    def _make_group_norm(self, channels: int) -> nn.GroupNorm:
        if channels % 8 == 0:
            return nn.GroupNorm(8, channels)
        if channels % 4 == 0:
            return nn.GroupNorm(4, channels)
        return nn.GroupNorm(1, channels)

    def _normalize_inclination(self, inc: torch.Tensor) -> torch.Tensor:
        return (inc - 50.0) / 40.0

    def _curve_stats(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=-1)
        ptp = x.max(dim=-1).values - x.min(dim=-1).values
        return torch.cat([std, ptp], dim=1)

    def _compute_fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=-1, keepdim=True)
        fft_complex = torch.fft.rfft(x_centered, dim=-1)
        fft_mag = torch.abs(fft_complex)
        fft_mag = torch.log1p(fft_mag)

        # Drop DC bin and keep only low-frequency structure.
        fft_mag = fft_mag[:, :, 1 : self.fft_bins + 1]
        return fft_mag

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        inc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(
                f"Expected input with shape (batch, {self.input_length}) or "
                f"(batch, 1, {self.input_length}), got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {x.shape[-1]}"
            )

        if inc.ndim == 1:
            inc = inc.unsqueeze(1)
        elif inc.ndim != 2 or inc.shape[1] != 1:
            raise ValueError(
                f"Expected inclination with shape (batch,) or (batch, 1), got {tuple(inc.shape)}"
            )

        return x, inc

    def extract_features(self, x: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
        x, inc = self._prepare_inputs(x, inc)

        time_feat = self.time_pool(self.time_features(x))

        fft_mag = self._compute_fft_magnitude(x)
        fft_feat = self.fft_pool(self.fft_features(fft_mag))

        inc = self._normalize_inclination(inc)
        stats = self._curve_stats(x)

        features = torch.cat([time_feat, fft_feat, inc, stats], dim=1)
        return features

    def forward_with_count(
        self,
        x: torch.Tensor,
        inc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            logits:     shape (batch, 5)
            count_pred: shape (batch,), continuous prediction in [1, 5]
        """
        features = self.extract_features(x, inc)
        shared = self.shared_head(features)

        logits = self.class_head(shared)

        # Constrain count prediction to the physically valid range [1, 5].
        count_raw = self.count_head(shared).squeeze(1)
        count_pred = 1.0 + 4.0 * torch.sigmoid(count_raw)

        return logits, count_pred

    def forward(self, x: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
        """
        Return classification logits only.

        This preserves compatibility with evaluation scripts that call:
            logits = model(x, inc)
        """
        logits, _ = self.forward_with_count(x, inc)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, inc)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x, inc)
        return torch.argmax(probs, dim=1)

    @torch.no_grad()
    def predict_count(self, x: torch.Tensor, inc: torch.Tensor) -> torch.Tensor:
        _, count_pred = self.forward_with_count(x, inc)
        return count_pred