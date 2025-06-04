from bipedal_env import BipedalEnv
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# ===== Custom Callback to Track Episode Rewards =====
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # vec_env stores info dicts per env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
        return True

# ===== Environment factory function =====
def make_env():
    def _init():
        env = BipedalEnv(max_episode_steps=1000)
        return env
    return _init

# ===== Create vectorized environment =====
vec_env = DummyVecEnv([make_env()])

# ===== Instantiate PPO model =====
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_bipedal_tensorboard/")

# ===== Training block with callback =====
total_timesteps = 500000
reward_callback = RewardTrackingCallback()
model.learn(total_timesteps=total_timesteps, callback=reward_callback)
model.save("ppo_bipedal_slider")

# ===== Plot training rewards =====
plt.plot(reward_callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Reward per Episode')
plt.grid()
plt.show()
