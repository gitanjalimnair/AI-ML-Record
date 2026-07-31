import random
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


class Agent:
    def __init__(self):
        self.model = DQN(4, 2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, 1)

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def train(self, batch):

        for state, action, reward, next_state, done in batch:

            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward

            if not done:
                target += self.gamma * torch.max(
                    self.model(next_state)
                ).item()

            output = self.model(state)[action]

            loss = self.criterion(
                output,
                torch.tensor(target)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay