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
import random
from isaacgym.torch_utils import *
from isaacgym.torch_utils import torch_rand_float

from isaacgym import gymtorch, gymapi, gymutil
import math
import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Slider(LeggedRobot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.nx = 43
        self.base_height_target = 0.65
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), dtype=torch.bool, device=self.device)
  

    def compute_reward(self):
        
        self.gym.render_all_camera_sensors(self.sim)
        self.body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, self.num_bodies, -1)

        dof_states = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        ).view(self.num_envs, self.num_dofs,2)

        self.joint_positions = dof_states[:, :, 0]
        #print("Observation shape:", self.obs_buf.shape)
        return super().compute_reward()
    


    #-----------------------------Reward Functions-----------------------------

    #-----------Posture Rewards-----------
    
    def _reward_standing_height(self):
        height = self.root_states[:, 2]
        error = height - self.base_height_target
        sigma = 0.1  # tolerance (10cm)
        reward = torch.exp(- (error ** 2) / (2 * sigma ** 2))
        return reward
    
    def _reward_upright_torso(self):
        quat = self.root_states[:, 3:7]
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (np.pi/2), torch.asin(sinp))
        penalty = roll**2  + pitch**2 
        return torch.exp(-penalty)
    
    #-----------Velocity Functions-----------
    
    def _reward_tracking_lin_vel_x(self):
        vel_error_x = self.base_lin_vel[:, 0] - self.commands[:, 0]
        return torch.exp(-self.cfg.rewards.tracking_sigma * vel_error_x**2)

    def _reward_tracking_lin_vel_y(self):
        vel_error_y = self.base_lin_vel[:, 1] - self.commands[:, 1]
        return torch.exp(-self.cfg.rewards.tracking_sigma * vel_error_y**2)

    def _reward_tracking_yaw_rate(self):
        yaw_error = self.base_ang_vel[:, 2] - self.commands[:, 2]
        return torch.exp(-self.cfg.rewards.tracking_sigma * yaw_error**2)
    

    
    def _reward_alternating_and_symmetric(self):
        # --- Alternating Foot Contact Reward ---
        current_contacts = (self.contact_forces[:, self.feet_indices, 2] > 0.1)  # bool tensor [envs, feet]
        foot_just_contact = current_contacts & (~self.last_contacts)  # foot just touched down this step
        alternating_reward = foot_just_contact.sum(dim=1).float()  # reward per env
        self.last_contacts = current_contacts.clone()

        # --- Pose Symmetry Reward ---
        left_joints = self.joint_positions[:, [0,1,2,3,4]]
        right_joints = self.joint_positions[:, [5,6,7,8,9]]

        # Mirror signs for joints - adjust based on your robot's joint directions
        mirror_signs = torch.tensor([1, 1, -1, 1, 1], device=self.device)

        mirrored_right = right_joints * mirror_signs

        diff = torch.abs(left_joints - mirrored_right)
        symmetry_reward = torch.exp(-torch.sum(diff, dim=1))

        # --- Combine rewards ---
        combined_reward = alternating_reward + 0.8*symmetry_reward  # You can weight them if needed

        return combined_reward

    
    def _reward_foot_slip(self):
        # Check which feet are in contact
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 1.0  # bool [envs, feet]

        # Get XY velocities for each foot
        foot_lin_vel = self.body_states[:, self.feet_indices, 7:9]  # vx, vy [envs, feet, 2]
    
        # Slip magnitude per foot
        slip_mag = torch.norm(foot_lin_vel, dim=-1)  # [envs, feet]

        # Average slip only for feet in contact
        slip_penalty = (slip_mag * contact_mask).mean(dim=1)  # [envs]
        
        # Return as reward (lower slip = higher reward)
        return torch.exp(-slip_penalty)
    
    def _reward_foot_clearance(self):
        # Detect swing phase (feet not in contact)
        swing_mask = self.contact_forces[:, self.feet_indices, 2] < 1.0  # bool [envs, feet]
    
        # Get Z position of each foot
        foot_heights = self.body_states[:, self.feet_indices, 2]  # [envs, feet]

        # Target clearance (you can tune in config)
        clearance_target = 0.1  # 8 cm

        # Error only for swing feet
        clearance_error = torch.abs(foot_heights - clearance_target) * swing_mask.float()

        # Avoid zero division (if no feet are in swing)
        avg_error = (clearance_error.sum(dim=1) / (swing_mask.sum(dim=1) + 1e-6))
        
        return torch.exp(-avg_error)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        has_contact = torch.sum(1.0*contacts, dim=1) > 0
        return 1.0 * has_contact
    
   