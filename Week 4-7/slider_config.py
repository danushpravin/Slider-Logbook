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

class SliderCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12

    
    class terrain( LeggedRobotCfg.terrain):
        #measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        #measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        mesh_type = "plane" 
        measure_heights = False
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.64] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'Roll_L_joint': 0.0,
            'Yaw_L_joint': 0.0,
            'Pitch_L_joint': 0.0,
            'Slide_L_joint': 0.0,
            'Foot_L_Bot_joint': 0.0,
            'Foot_L_Top_joint': 0.0,
            
            'Roll_R_joint': 0.0,
            'Yaw_R_joint': 0.0,
            'Pitch_R_joint': 0.0,
            'Slide_R_joint': 0.0,
            'Foot_R_Bot_joint': 0.0,
            'Foot_R_Top_joint': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'Roll_L_joint': 100.0, 'Yaw_L_joint': 100.0,
                        'Pitch_L_joint': 200.0,'Slide_L_joint':55.0, 'Foot_L_Bot_joint': 60.0, 'Foot_L_Top_joint': 60.0,
                        'Roll_R_joint': 100.0, 'Yaw_R_joint': 100.0,
                        'Pitch_R_joint': 200.0, 'Slide_R_joint':55.0,'Foot_R_Bot_joint': 60.0, 'Foot_R_Top_joint': 60.0}  # [N*m/rad]
        damping = {  'Roll_L_joint': 3.0, 'Yaw_L_joint': 3.0,
                        'Pitch_L_joint': 6.0,'Slide_L_joint':2.0, 'Foot_L_Bot_joint': 1.0, 'Foot_L_Top_joint': 1.0,
                        'Roll_R_joint': 3.0, 'Yaw_R_joint': 3.0,
                        'Pitch_R_joint': 6.0, 'Slide_R_joint':2.0,'Foot_R_Bot_joint': 1.0, 'Foot_R_Top_joint': 1.0}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '/home/danush/Documents/Slider_Trial_6/urdf/Slider_Trial_6.urdf'
        name = "slider"
        foot_name = "_Bot_link"


        terminate_after_contacts_on = ['base_link','base_joint','Roll_R_link','Roll_L_link',
                                       'Yaw_R_link','Yaw_L_link','Pitch_R_link','Pitch_L_link',
                                       'Foot_R_Top_link','Foot_L_Top_link']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter



    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200
            tracking_ang_vel = 0.0
            torques =  -5.e-6  # 5x stronger penalty on torque
            dof_acc =  0 # 10x stronger penalty on acceleration

            lin_vel_z = -0.0
            feet_air_time = 0.0
            dof_pos_limits = -1
            no_fly = 0.0
            dof_vel = -0.0
            ang_vel_xy = 0.0
            feet_contact_forces = 0.0
            #Custom Rewards
            #gait
            lipm_foot_target = 2.5
            com_lipm = 2.0

            standing_height = 5.0
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            one_foot_contact = 1.0
            upright_torso = 3.0
            com_centered = 0.0
            action_smoothness = -0.0

            

class SliderCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'Walking_V1.2'
        experiment_name = 'rough_slider'
        max_iterations = 1000
    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  
