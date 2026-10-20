"""Five-block temporal CNN for windowed biosignals (Atzori et al., 2016).

Input tensors have shape ``(batch, channels, time)``. They are treated as
single-map images with time as height and channels as width. The first block
collapses the channel axis; the final block produces one logit per class and
time step, which are averaged into ``(batch, classes)`` logits.
"""

import torch
from torch import nn


class TCNN(nn.Module):
    """Configurable five-block temporal convolutional classifier."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        time_kernel: int = 5,
        pool_stride: int = 1,
        n_filters: tuple[int, int, int, int] = (32, 32, 64, 64),
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_classes <= 0:
            raise ValueError("n_channels and n_classes must be positive")
        if time_kernel <= 0 or time_kernel % 2 == 0:
            raise ValueError("time_kernel must be a positive odd integer")
        if pool_stride <= 0:
            raise ValueError("pool_stride must be positive")
        if len(n_filters) != 4 or any(value <= 0 for value in n_filters):
            raise ValueError("n_filters must contain four positive integers")
        f1, f2, f3, f4 = n_filters

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, n_channels)),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=3,
                stride=pool_stride,
                padding=1,
                count_include_pad=False,
            ),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=3,
                stride=pool_stride,
                padding=1,
                count_include_pad=False,
            ),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(
                f3,
                f4,
                kernel_size=(time_kernel, 1),
                padding=(time_kernel // 2, 0),
            ),
            nn.ReLU(),
        )
        self.block5 = nn.Conv2d(f4, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for ``(batch, channels, time)`` input."""

        x = x.transpose(1, 2).unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.mean(dim=(2, 3))
