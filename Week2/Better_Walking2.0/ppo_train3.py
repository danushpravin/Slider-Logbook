from bipedal_env3 import BipedalEnv
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import numpy as np

# Import your environment class code here (or from a separate file)
  # replace 'your_env_file' with the actual filename without .py

def make_env():
    return BipedalEnv(max_episode_steps=500)

if __name__ == "__main__":
    # Create vectorized environment for stable baselines (required)
    env = DummyVecEnv([make_env])

    # Define PPO model with hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(log_std_init=-2),
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_biped_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=1_000_000)

    # Save the trained model
    model.save("ppo_biped_isaac")

    # Close environment to free resources
    env.close()