import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("models/ppo_lunar_lander")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.02)

    if terminated or truncated:
        obs, info = env.reset()