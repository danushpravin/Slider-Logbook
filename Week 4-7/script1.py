from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch
def walk_straight():
    # Hardcoded task and model checkpoint paths
    task_name = "rough_slider"  # Your bot's env/task name
    run_name = "Jul06_15-53-28_Walking_V1.1"  # Your training run folder name
    checkpoint = 950  # Model checkpoint number

    # Paths
    run_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/rough_slider", run_name)
    model_path = os.path.join(run_path, f"model_{checkpoint}.pt")

    # Load env and train config
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)

    # Set smaller env for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 9)
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

    # Commands: walk straight forward at 0.5 m/s
    forward_speed = 1.0
    lateral_speed = 0.0
    yaw_rate = 0.0

    for step in range(total_steps):
        env.commands[:, 0] = forward_speed  # vx
        env.commands[:, 1] = lateral_speed  # vy
        env.commands[:, 2] = yaw_rate        # yaw rate

        obs_tensor = obs[0] if isinstance(obs, (tuple, list)) else obs
        
        actions = policy(obs_tensor.detach()) 
        if actions.dim() == 3:
            # If actions shape is (num_envs, num_actions, 1) or similar, squeeze last dim
            actions = actions.squeeze(-1)
        elif actions.dim() == 1:
            # if actions is flat, reshape to (num_envs, num_actions)
            actions = actions.view(env.num_envs, -1)

        obs, _, reward, done, info = env.step(actions)


        env.render()


    print("Finished walking straight.")



if __name__ == "__main__":
    walk_straight()
