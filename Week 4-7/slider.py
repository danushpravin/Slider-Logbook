# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot



class Slider(LeggedRobot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nx = 43
        self.base_height_target = 0.65
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)


    def lipm_step(self, x0, xdot0, z0, t_step):
        omega = torch.sqrt((9.81 / z0).clamp(min=1e-3))  # avoid NaNs if z0 is tiny

        x = x0 * torch.cosh(omega * t_step) + xdot0 / omega * torch.sinh(omega * t_step)
        xdot = x0 * omega * torch.sinh(omega * t_step) + xdot0 * torch.cosh(omega * t_step)
        return x, xdot
    
    def _reward_lipm_foot_target(self):
        # Predict CoM and foot placement using LIPM
        com_pos = self.root_states[:, 0]  # x position
        com_vel = self.base_lin_vel[:, 0]
        z0 = self.root_states[:, 2].clamp(min=0.2)  # avoid division by 0
        t_step = 0.3  # assume step duration

        pred_com, _ = self.lipm_step(com_pos, com_vel, z0, t_step)
        desired_foot = pred_com  # Place foot ahead of CoM
    
        # Actual foot
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states = gymtorch.wrap_tensor(body_states)
        bodies_per_env = body_states.shape[0] // self.num_envs
        foot_states = body_states.view(self.num_envs, bodies_per_env, -1)
        foot_pos = foot_states[:, self.feet_indices, 0]  # X pos

        # Assume swing leg is right (index 1)
        foot_error = torch.square(foot_pos[:, 1] - desired_foot)
        reward = torch.exp(-10.0 * foot_error)
        return reward
    def _reward_com_lipm(self):
        com_pos = self.root_states[:, 0]
        com_vel = self.base_lin_vel[:, 0]
        z0 = self.root_states[:, 2].clamp(min=0.2)
        t_step = 0.3

        pred_com, _ = self.lipm_step(com_pos, com_vel, z0, t_step)

        error = torch.square(com_pos - pred_com)
        return torch.exp(-5.0 * error)


    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    def _reward_standing_height(self):
        height = self.root_states[:, 2]
        error = height - self.base_height_target
        reward = torch.exp(-5.0 * (error ** 2))  # peak at target, drops off quickly
        return reward
    def _reward_tracking_lin_vel_x(self):
        target_vel_x = self.commands[:, 0]
        vel_error = self.base_lin_vel[:, 0] - target_vel_x
        return torch.exp(-2.0 * vel_error**2)

    def _reward_tracking_lin_vel_y(self):
        target_vel_y = self.commands[:, 1]
        vel_error = self.base_lin_vel[:, 1] - target_vel_y
        return torch.exp(-2.0 * vel_error**2)


    def _reward_one_foot_contact(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 10.0
        left_contact = contacts[:, 0].float()
        right_contact = contacts[:, 1].float()

        total_contacts = left_contact + right_contact
        reward = (total_contacts == 1).float()
        return reward

    def _reward_upright_torso(self):
        quat = self.root_states[:, 3:7]
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (np.pi/2), torch.asin(sinp))
        penalty = roll**2 * 10 + pitch**2 * 5
        return torch.exp(-penalty)
    def _reward_com_centered(self):
        com_xy = self.root_states[:, :2]  # X, Y of base
        foot_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        foot_states = gymtorch.wrap_tensor(foot_states)
        bodies_per_env = foot_states.shape[0] // self.num_envs
        foot_states = foot_states.view(self.num_envs, bodies_per_env, -1)
        foot_xy = foot_states[:, self.feet_indices, :2]  # (N, 2, 2)
        foot_center = torch.mean(foot_xy, dim=1)
        offset = com_xy - foot_center
        return torch.exp(-10.0 * torch.sum(offset**2, dim=1))
    def _reward_action_smoothness(self):
        if not hasattr(self, "prev_actions"):
            self.prev_actions = self.actions.clone()
            return torch.ones(self.num_envs, device=self.device)

        delta = self.actions - self.prev_actions
        self.prev_actions = self.actions.clone()

        # You can tune the scale factor
        return torch.exp(-10.0 * torch.sum(delta**2, dim=1))

    



   
    








    












