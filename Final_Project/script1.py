from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch
def walk_straight():
    # Hardcoded task and model checkpoint paths
    task_name = "rough_slider"  # Your bot's env/task name
    run_name = "Aug04_15-55-15_Goal_Oriented_Walkerv2.0"  # Your training run folder name
    checkpoint = 10000  # Model checkpoint number

    # Paths
    run_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/rough_slider", run_name)
    model_path = os.path.join(run_path, f"model_{checkpoint}.pt")

    # Load env and train config
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)

    # Set smaller env for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Create environment
    env, _ = task_registry.make_env(name=task_name, args=None, env_cfg=env_cfg)

    # Create PPO runner and load trained model weights
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=task_name, args=None, train_cfg=train_cfg)
    ppo_runner.load(model_path)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.reset()
    total_steps = 1000  # Run for 1000 steps

    # Commands: walk straight forward at 1.5 m/s with heading 0 radians (face forward)
    forward_speed = 1.5
    lateral_speed = 0.0
    yaw_rate = 0.0  # absolute heading angle in radians, e.g. 0 = facing x+

    for step in range(total_steps):
        env.commands[:, 0] = forward_speed   # vx
        env.commands[:, 1] = lateral_speed   # vy
                    # yaw rate (ignored when heading_command=True)
        env.commands[:, 2] = yaw_rate   # heading (absolute angle)

        obs_tensor = obs[0].detach() if isinstance(obs, (tuple, list)) else obs.detach()


        actions = policy(obs_tensor.detach())

        if actions.dim() == 3:
            actions = actions.squeeze(-1)
        elif actions.dim() == 1:
            actions = actions.view(env.num_envs, -1)

        obs, _, reward, done, info = env.step(actions)
        env.render()
        quat = env.root_states[:, 3:7]
        w, x, y, z = quat[0,0], quat[0,1], quat[0,2], quat[0,3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp.cpu(), cosy_cosp.cpu())
        #print(f"Step {step}, Current Yaw: {yaw:.2f}, Target Heading: {heading_angle}")

    print("Finished walking straight.")

if __name__ == "__main__":
    walk_straight()
