import gymnasium as gym
import torch
from model import DQN

env = gym.make("CartPole-v1", render_mode="human")

model = DQN(4, 2)
model.load_state_dict(torch.load("models/dqn.pth"))
model.eval()

state, info = env.reset()

done = False

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()