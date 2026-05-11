# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import CNN, CNN3D, CNNTSM, MLP, EmpiricalNormalization

from .actor_critic import ActorCritic
from ipdb import set_trace

class ActorCriticCNN(ActorCritic):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_cnn_cfg: dict[str, dict] | dict | None = None,
        critic_cnn_cfg: dict[str, dict] | dict | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super(ActorCritic, self).__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        actor_in_dims_nd = []
        actor_in_channels_nd = []
        self.actor_obs_groups_nd = []
        self.actor_obs_ndim = {}  # obs_group -> ndim (4 or 5)
        for obs_group in obs_groups["policy"]:
            ndim = len(obs[obs_group].shape)
            if ndim == 5:  # B, C, D, H, W
                self.actor_obs_groups_nd.append(obs_group)
                actor_in_dims_nd.append(obs[obs_group].shape[2:5])  # D, H, W
                actor_in_channels_nd.append(obs[obs_group].shape[1])  # C
                self.actor_obs_ndim[obs_group] = 5
            elif ndim == 4:  # B, C, H, W
                self.actor_obs_groups_nd.append(obs_group)
                actor_in_dims_nd.append(obs[obs_group].shape[2:4])  # H, W
                actor_in_channels_nd.append(obs[obs_group].shape[1])  # C
                self.actor_obs_ndim[obs_group] = 4
            elif ndim == 2:  # B, C
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")
        num_critic_obs_1d = 0
        self.critic_obs_groups_1d = []
        critic_in_dims_nd = []
        critic_in_channels_nd = []
        self.critic_obs_groups_nd = []
        self.critic_obs_ndim = {}
        for obs_group in obs_groups["critic"]:
            ndim = len(obs[obs_group].shape)
            if ndim == 5:  # B, C, D, H, W
                self.critic_obs_groups_nd.append(obs_group)
                critic_in_dims_nd.append(obs[obs_group].shape[2:5])
                critic_in_channels_nd.append(obs[obs_group].shape[1])
                self.critic_obs_ndim[obs_group] = 5
            elif ndim == 4:  # B, C, H, W
                self.critic_obs_groups_nd.append(obs_group)
                critic_in_dims_nd.append(obs[obs_group].shape[2:4])
                critic_in_channels_nd.append(obs[obs_group].shape[1])
                self.critic_obs_ndim[obs_group] = 4
            elif ndim == 2:  # B, C
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        # Assert that there are image observations (2D or 3D)
        assert self.actor_obs_groups_nd or self.critic_obs_groups_nd, (
            "No image observations (4D or 5D) are provided. If this is intentional, use the ActorCritic module instead."
        )

        # Actor CNN
        if self.actor_obs_groups_nd:
            # Resolve the actor CNN configuration
            assert actor_cnn_cfg is not None, "An actor CNN configuration is required for image actor observations."
            # If a single configuration dictionary is provided, create a dictionary for each image observation group
            if not all(isinstance(v, dict) for v in actor_cnn_cfg.values()):
                actor_cnn_cfg = {group: actor_cnn_cfg for group in self.actor_obs_groups_nd}
            # Check that the number of configs matches the number of observation groups
            assert len(actor_cnn_cfg) == len(self.actor_obs_groups_nd), (
                "The number of CNN configurations must match the number of image actor observations."
            )

            # Create CNNs for each image actor observation
            self.actor_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.actor_obs_groups_nd):
                cfg_dict = actor_cnn_cfg[obs_group]
                ndim = self.actor_obs_ndim[obs_group]
                if "frames_per_step" in cfg_dict:
                    # CNNTSM works from both 4D (C*T,H,W) and 5D (C,T,H,W) inputs.
                    # In both cases it needs input_dim=(H,W) and input_channels=C*T.
                    cnn_class = CNNTSM
                    if ndim == 5:
                        # actor_in_dims_nd[idx] is (T,H,W); actor_in_channels_nd[idx] is C
                        T_dim = actor_in_dims_nd[idx][0]
                        cnn_input_dim = actor_in_dims_nd[idx][1:]
                        cnn_input_channels = actor_in_channels_nd[idx] * T_dim
                    else:
                        cnn_input_dim = actor_in_dims_nd[idx]
                        cnn_input_channels = actor_in_channels_nd[idx]
                elif ndim == 5:
                    cnn_class = CNN3D
                    cnn_input_dim = actor_in_dims_nd[idx]
                    cnn_input_channels = actor_in_channels_nd[idx]
                else:
                    cnn_class = CNN
                    cnn_input_dim = actor_in_dims_nd[idx]
                    cnn_input_channels = actor_in_channels_nd[idx]
                self.actor_cnns[obs_group] = cnn_class(
                    input_dim=cnn_input_dim,
                    input_channels=cnn_input_channels,
                    **cfg_dict,
                )
                print(f"Actor {cnn_class.__name__} for {obs_group}: {self.actor_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.actor_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.actor_cnns[obs_group].output_dim)  # type: ignore
                else:
                    raise ValueError("The output of the actor CNN must be flattened before passing it to the MLP.")
        else:
            self.actor_cnns = None
            encoding_dim = 0

        # Actor MLP
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs_1d + encoding_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs_1d + encoding_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization (only for 1D actor observations)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic CNN
        if self.critic_obs_groups_nd:
            # Resolve the critic CNN configuration
            assert critic_cnn_cfg is not None, "A critic CNN configuration is required for image critic observations."
            # If a single configuration dictionary is provided, create a dictionary for each image observation group
            if not all(isinstance(v, dict) for v in critic_cnn_cfg.values()):
                critic_cnn_cfg = {group: critic_cnn_cfg for group in self.critic_obs_groups_nd}
            # Check that the number of configs matches the number of observation groups
            assert len(critic_cnn_cfg) == len(self.critic_obs_groups_nd), (
                "The number of CNN configurations must match the number of image critic observations."
            )

            # Create CNNs for each image critic observation
            self.critic_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.critic_obs_groups_nd):
                cfg_dict = critic_cnn_cfg[obs_group]
                ndim = self.critic_obs_ndim[obs_group]
                if "frames_per_step" in cfg_dict:
                    cnn_class = CNNTSM
                    if ndim == 5:
                        T_dim = critic_in_dims_nd[idx][0]
                        cnn_input_dim = critic_in_dims_nd[idx][1:]
                        cnn_input_channels = critic_in_channels_nd[idx] * T_dim
                    else:
                        cnn_input_dim = critic_in_dims_nd[idx]
                        cnn_input_channels = critic_in_channels_nd[idx]
                elif ndim == 5:
                    cnn_class = CNN3D
                    cnn_input_dim = critic_in_dims_nd[idx]
                    cnn_input_channels = critic_in_channels_nd[idx]
                else:
                    cnn_class = CNN
                    cnn_input_dim = critic_in_dims_nd[idx]
                    cnn_input_channels = critic_in_channels_nd[idx]
                self.critic_cnns[obs_group] = cnn_class(
                    input_dim=cnn_input_dim,
                    input_channels=cnn_input_channels,
                    **cfg_dict,
                )
                print(f"Critic {cnn_class.__name__} for {obs_group}: {self.critic_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.critic_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.critic_cnns[obs_group].output_dim)  # type: ignore
                else:
                    raise ValueError("The output of the critic CNN must be flattened before passing it to the MLP.")
        else:
            self.critic_cnns = None
            encoding_dim = 0

        # Critic MLP
        self.critic = MLP(num_critic_obs_1d + encoding_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization (only for 1D critic observations)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        if self.actor_cnns is not None:
            # Encode the image actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_nd]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        super()._update_distribution(mlp_obs)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, cnn_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_cnns is not None:
            # Encode the image actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_nd]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)
        # set_trace()
        if self.state_dependent_std:
            return self.actor(mlp_obs)[..., 0, :]
        else:
            return self.actor(mlp_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_cnns is not None:
            # Encode the image critic observations
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_groups_nd]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        return self.critic(mlp_obs)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_groups_1d]
        obs_dict_nd = {}
        for obs_group in self.actor_obs_groups_nd:
            obs_dict_nd[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_nd

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_groups_1d]
        obs_dict_nd = {}
        for obs_group in self.critic_obs_groups_nd:
            obs_dict_nd[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_nd

    def reset_cache(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset temporal caches in all CNNTSM modules for the given environments."""
        for cnn_dict in (self.actor_cnns, self.critic_cnns):
            if cnn_dict is None:
                continue
            for cnn in cnn_dict.values():
                if hasattr(cnn, "reset_cache"):
                    cnn.reset_cache(env_ids)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
