# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn as nn

from .cnn import CNN


class TemporalShift(nn.Module):
    """Unidirectional (look-back only) temporal shift via a per-layer feature cache.

    At each step the first ``n_shift`` channels of the incoming activation are
    replaced with the cached values from the *previous* step, and the displaced
    channels are saved as the new cache.  The Conv2d that follows therefore sees
    past-step spatial features in its first ``n_shift`` channels with no
    look-ahead, making the operation strictly causal.

    This matches the online inference design in the original TSM repo
    (``online_demo/mobilenet_v2_tsm.py``, Lin et al. 2019 §C).
    """

    def __init__(self, num_channels: int, shift_fraction: float = 0.125):
        super().__init__()
        n = max(1, int(num_channels * shift_fraction))
        n = min(n, num_channels - 1)  # keep at least one non-shifted channel
        self.n_shift = n if num_channels >= 2 else 0

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the look-back shift.

        Args:
            x: Current activation ``(B, C, H, W)``.
            cache: Cached first-n channels from the previous step ``(B, n, H, W)``.

        Returns:
            Tuple of ``(shifted_x, new_cache)``.  ``shifted_x`` has the same
            shape as ``x``; ``new_cache`` should replace the stored cache.
        """
        if self.n_shift == 0:
            return x, cache
        n = self.n_shift
        new_cache = x[:, :n].clone()
        return torch.cat([cache, x[:, n:]], dim=1), new_cache


class CNNTSM(CNN):
    """CNN with online, unidirectional Temporal Shift Module (TSM).

    Processes **one frame per step** ``(B, C, H, W)``.  Temporal context is
    maintained through per-layer activation caches that persist across steps,
    analogous to a recurrent hidden state.  Before each Conv2d a fraction of
    the current activation channels are replaced with cached values from the
    previous step (look-back only — strictly causal, no look-ahead).

    Compared to the offline frame-stack approach this design:

    * Uses 1/T the compute (one frame processed per step).
    * Carries temporal context indefinitely rather than being capped at T frames.
    * Avoids bidirectional mixing that inflates past-frame representations with
      future information before the mean-pool.
    * Requires no raw frame buffer in the environment.

    **Cache lifecycle:**

    * Caches are initialised lazily to zeros on the first forward call.
    * Call :meth:`reset_cache` for environments that have just reset so that
      stale context from the previous episode does not bleed into the new one.
      In practice this should be called by the runner when ``dones > 0``.

    Args:
        input_dim: ``(H, W)`` spatial size of each input frame.
        input_channels: Number of input channels ``C``.
        output_channels: Output channels per conv layer.
        kernel_size: Spatial kernel size(s).
        frames_per_step: Always ``1``; TSM processes one frame per step and
            carries temporal context via internal activation caches, not a raw
            frame stack. This parameter exists only so that config dicts can
            distinguish CNNTSM from CNN3D (which uses ``frames_per_step > 1``).
        shift_fraction: Fraction of channels used as the temporal cache
            (default ``0.125``, matching the original paper).
        **kwargs: Forwarded verbatim to :class:`CNN`
            (``stride``, ``dilation``, ``padding``, ``norm``, ``max_pool``,
            ``global_pool``, ``flatten``, …).
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int] | list[int],
        kernel_size,
        frames_per_step: int = 1,
        shift_fraction: float = 0.125,
        **kwargs,
    ):
        assert frames_per_step == 1, (
            f"CNNTSM (online mode) requires frames_per_step=1, got {frames_per_step}. "
            "Temporal context is carried by per-layer activation caches, not a raw frame stack."
        )
        super().__init__(
            input_dim=input_dim,
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

        # One TemporalShift per Conv2d; channel counts at each conv's input.
        channel_sizes = [input_channels] + list(output_channels[:-1])
        self._tsm_shifts = nn.ModuleList(
            [TemporalShift(c, shift_fraction) for c in channel_sizes]
        )
        # Per-layer activation caches, lazily initialised on the first forward.
        # Stored as a plain list (not registered buffers) so they are not part
        # of state_dict() — caches are ephemeral runtime state, not weights.
        self._caches: list[torch.Tensor | None] = [None] * len(self._tsm_shifts)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the temporal cache for the specified environments.

        Args:
            env_ids: 1-D integer tensor of environment indices to reset.
                Pass ``None`` to reset all environments (e.g. at training start).
        """
        for cache in self._caches:
            if cache is None:
                continue
            if env_ids is None:
                cache.zero_()
            else:
                cache[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, 1, H, W) from envs that keep a singleton frame dim.
        if x.dim() == 5:
            x = x.squeeze(2)

        shift_idx = 0
        for name, layer in self._modules.items():
            if name == "_tsm_shifts":
                continue
            if isinstance(layer, nn.Conv2d):
                tsm = self._tsm_shifts[shift_idx]
                if tsm.n_shift > 0:
                    # Create or reset the cache when the batch size changes (e.g. PPO
                    # update passes num_steps*num_envs rather than num_envs).
                    if self._caches[shift_idx] is None or self._caches[shift_idx].shape[0] != x.shape[0]:
                        self._caches[shift_idx] = torch.zeros(
                            x.shape[0], tsm.n_shift, *x.shape[2:],
                            device=x.device, dtype=x.dtype,
                        )
                    x, new_cache = tsm(x, self._caches[shift_idx])
                    self._caches[shift_idx] = new_cache.detach()
                shift_idx += 1
            x = layer(x)

        return x
