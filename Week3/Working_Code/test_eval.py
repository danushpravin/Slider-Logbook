from bipedal_env import BipedalEnv
from stable_baselines3 import PPO

# Create environment
env = BipedalEnv(max_episode_steps=1000)

# Load trained model
model = PPO.load("ppo_bipedal_slider", env=env)

# Ensure episode tracking lists exist


# Run for multiple episodes
n_episodes = 10

for _ in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        step_count += 1

# Check if data is non-empty


env.close()
