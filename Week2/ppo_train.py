from bipedal_env import BipedalEnv
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#import gymnasium as gym
#from gymnasium.wrappers import Monitor




# Environment factory function with render_mode and Monitor wrapper
def make_env():
    def _init():
        env = BipedalEnv(max_episode_steps=1000, render_mode="human")  # Enable rendering
        #env = Monitor(env)  # Wrap environment with Monitor
        return env
    return _init

# Create vectorized environment
vec_env = DummyVecEnv([make_env()])

# Instantiate PPO agent with default MLP policy
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_bipedal_tensorboard/")

# -------------------
# Training block (commented out because model is already trained)
# total_timesteps = 1_000_000  # Adjust as needed
# model.learn(total_timesteps=total_timesteps)
# model.save("ppo_bipedal_slider")
# -------------------

# Load the trained model (replace path if different)
model = PPO.load("ppo_bipedal_slider", env=vec_env)

# Evaluate the trained policy
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Run inference / test with rendering
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(action)
    vec_env.render()  # Calls your env's render method

    if any(dones):
        obs = vec_env.reset()
