from bipedal_env import BipedalEnv
from stable_baselines3 import PPO

# Create a single instance of your env with render_mode='human' to enable rendering
env = BipedalEnv(max_episode_steps=1000)

# Load the trained model
model = PPO.load("ppo_bipedal_slider", env=env)

obs = env.reset()
for _ in range(1000):  # Run for 1000 steps (adjust as needed)
    action, _states = model.predict(obs, deterministic=True)  # deterministic=True for consistent behavior
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()  # Close the env/viewer properly after done
