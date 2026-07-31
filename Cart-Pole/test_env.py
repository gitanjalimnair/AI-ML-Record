import gymnasium as gym

env = gym.make("CartPole-v1")

state, info = env.reset()

print(state)
print(state.shape)

env.close()