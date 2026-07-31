import gymnasium as gym
import torch

from replay_buffer import ReplayBuffer
from agent import Agent

env = gym.make("CartPole-v1")

agent = Agent()

buffer = ReplayBuffer(10000)

episodes = 100
batch_size = 64

for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0

    done = False

    while not done:

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            agent.train(batch)

    print(
        f"Episode {episode+1} | Reward: {total_reward:.0f} | Epsilon: {agent.epsilon:.3f}"
    )

torch.save(agent.model.state_dict(), "models/dqn.pth")

env.close()