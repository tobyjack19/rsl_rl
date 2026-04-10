# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from torch import nn as nn

from rsl_rl.utils import get_param, resolve_nn_activation


class CNN3D(nn.Sequential):
    """3D Convolutional Neural Network (CNN3D).

    The CNN3D network is a sequence of 3D convolutional layers, optional normalization layers, optional activation
    functions, and optional pooling. Designed for encoding short temporal stacks of images (B, C, D, H, W) into a
    flat feature vector. The final output can be flattened.
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int],
        input_channels: int,
        output_channels: tuple[int] | list[int],
        kernel_size: int | tuple[int, ...] | list,
        stride: int | tuple[int, ...] | list = 1,
        dilation: int | tuple[int, ...] | list = 1,
        padding: str = "none",
        norm: str | tuple[str] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        """Initialize the CNN3D.

        Args:
            input_dim: Depth, height, and width of the input (D, H, W).
            input_channels: Number of input channels.
            output_channels: List of output channels for each convolutional layer.
            kernel_size: Kernel size(s) for each layer. Can be an int (applied uniformly to D, H, W), a 3-tuple
                (d, h, w) for a single layer, or a list of ints/tuples per layer.
            stride: Stride(s) for each layer. Same format as kernel_size.
            dilation: Dilation(s) for each layer. Same format as kernel_size.
            padding: Padding type. Either 'none', 'zeros', 'reflect', 'replicate', or 'circular'.
            norm: Normalization type(s) per layer. Either 'none', 'batch', or 'layer'.
            activation: Activation function to use.
            max_pool: Whether to apply max pooling after each layer.
            global_pool: Global pooling type at the end. Either 'none', 'max', or 'avg'.
            flatten: Whether to flatten the output tensor.
        """
        super().__init__()

        # Resolve activation function
        activation_function = resolve_nn_activation(activation)

        # Create layers sequentially
        layers = []
        last_channels = input_channels
        last_dim = input_dim  # (D, H, W)
        for idx in range(len(output_channels)):
            # Get parameters for the current layer
            k = _to_3tuple(get_param(kernel_size, idx))
            s = _to_3tuple(get_param(stride, idx))
            d = _to_3tuple(get_param(dilation, idx))
            p = (
                _compute_padding_3d(last_dim, k, s, d)
                if padding in ["zeros", "reflect", "replicate", "circular"]
                else (0, 0, 0)
            )

            # Append convolutional layer
            layers.append(
                nn.Conv3d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
                )
            )

            # Append normalization layer if specified
            n = get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm3d(output_channels[idx]))
            elif n == "layer":
                norm_input_dim = _compute_output_dim_3d(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0], norm_input_dim[1], norm_input_dim[2]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )

            # Append activation function
            layers.append(activation_function)

            # Apply max pooling if specified
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

            # Update last channels and dimensions
            last_channels = output_channels[idx]
            last_dim = _compute_output_dim_3d(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))

        # Apply global pooling if specified
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool3d((1, 1, 1)))
            last_dim = (1, 1, 1)
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
            last_dim = (1, 1, 1)
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )

        # Apply flattening if specified
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        # Store final output dimension
        self._output_channels = last_channels if not flatten else None
        self._output_dim = last_dim if not flatten else last_channels * last_dim[0] * last_dim[1] * last_dim[2]

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels or None if output is flattened."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int, int] | int:
        """Get the output (D, H, W) or total output dimension if output is flattened."""
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize the weights of the CNN3D with Kaiming initialization."""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN3D."""
        for layer in self:
            x = layer(x)
        return x


def _to_3tuple(val) -> tuple[int, int, int]:
    """Convert an int or tuple to a 3-tuple (D, H, W)."""
    if isinstance(val, int):
        return (val, val, val)
    if isinstance(val, (tuple, list)):
        if len(val) == 3:
            return tuple(val)
        elif len(val) == 2:
            # Interpret as (spatial, spatial) and use 1 for depth
            return (1, val[0], val[1])
    raise ValueError(f"Cannot convert {val} to a 3-tuple (D, H, W). Provide an int or a 3-tuple.")


def _compute_padding_3d(
    input_dhw: tuple[int, int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Compute the optimal padding for the current 3D layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    """
    result = []
    for i in range(3):
        p = math.ceil(
            (stride[i] * math.floor(input_dhw[i] / stride[i]) - input_dhw[i] - stride[i] + dilation[i] * (kernel[i] - 1) + 1) / 2
        )
        result.append(p)
    return tuple(result)


def _compute_output_dim_3d(
    input_dhw: tuple[int, int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    padding: tuple[int, int, int],
    is_max_pool: bool = False,
) -> tuple[int, int, int]:
    """Compute the output depth, height, and width of the current 3D layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    """
    result = []
    for i in range(3):
        dim = math.floor((input_dhw[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) / stride[i] + 1)
        if is_max_pool:
            dim = math.ceil(dim / 2)
        result.append(dim)
    return tuple(result)
