import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("LunarLander-v3")

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train the model
model.learn(total_timesteps=200000)

# Save the trained model
model.save("models/ppo_lunar_lander")

env.close()