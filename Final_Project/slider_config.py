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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
class SliderCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 229
        num_actions = 10


    
    class terrain(LeggedRobotCfg.terrain):
        #mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True
        randomize_friction = True
        friction_range = [0.6, 1.2]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        max_init_terrain_level = 1  # Start easier

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 2.5  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.2, 2.0]   # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5] # min max [rad/s]

        

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.64] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'Roll_L_joint': 0.0,
            'Yaw_L_joint': 0.0,
            'Pitch_L_joint': 0.0,
            'Slide_L_joint': 0.0,
            'Foot_L_Bot_joint': 0.0,
            
            'Roll_R_joint': 0.0,
            'Yaw_R_joint': 0.0,
            'Pitch_R_joint': 0.0,
            'Slide_R_joint': 0.0,
            'Foot_R_Bot_joint': 0.0
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'Roll_L_joint': 100.0, 'Yaw_L_joint': 100.0,
                        'Pitch_L_joint': 200.0,'Slide_L_joint':55.0, 'Foot_L_Bot_joint': 60.0, 
                        'Roll_R_joint': 100.0, 'Yaw_R_joint': 100.0,
                        'Pitch_R_joint': 200.0, 'Slide_R_joint':55.0,'Foot_R_Bot_joint': 60.0}  # [N*m/rad]
        damping = {  'Roll_L_joint': 3.0, 'Yaw_L_joint': 3.0,
                        'Pitch_L_joint': 6.0,'Slide_L_joint':2.0, 'Foot_L_Bot_joint': 1.0, 
                        'Roll_R_joint': 3.0, 'Yaw_R_joint': 3.0,
                        'Pitch_R_joint': 6.0, 'Slide_R_joint':2.0,'Foot_R_Bot_joint': 1.0}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '/home/danush/Documents/Slider_Trial_6/urdf/Slider_Trial_6.urdf'
        name = "slider"
        foot_name = "Foot"


        terminate_after_contacts_on = ['base_link','base_joint','Roll_R_link','Roll_L_link',
                                       'Yaw_R_link','Yaw_L_link','Pitch_R_link','Pitch_L_link','Slide_R_link','Slide_L_link']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter



    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        tracking_sigma = 1
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200
            tracking_ang_vel = 0.0
            torques =  -5.e-6  # 5x stronger penalty on torque
            dof_acc =  -1e-7 # 10x stronger penalty on acceleration

            lin_vel_z = -0.2
            feet_air_time = 0.0
            dof_pos_limits = -1
            dof_vel = -5e-4
            ang_vel_xy = 0.0
            feet_contact_forces = -0.05
            #Custom Rewards
            #gait

            standing_height = 3.0
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            tracking_yaw_rate = 1.5
            upright_torso = 1.0
            alternating_and_symmetric = 1.25
            foot_slip = 0.25
            foot_clearance = 0.25
            no_fly = 0.0
            
            
class SliderCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'Goal_Oriented_Walkerv2.0'            # SAME as the folder used in first training run
        experiment_name = 'rough_slider'          # New task/environment name
        load_run = 'Aug04_13-55-28_Goal_Oriented_Walkerv2.0'            # Load weights from this run
        load_checkpoint = 5000                    # The checkpoint to load (e.g. 1000.pt)
        resume = True                             # Important: allows training to resume from this checkpoint
        max_iterations = 5000                     # Total iterations now (1000 already done)

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        #learning_rate = 1e-3
       # num_learning_epochs = 5
        #gamma = 0.99
       # lam = 0.95
        #num_mini_batches = 4






  