from statistics import mean
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random


class DQN(object):

    def __init__(self, architecture, hparams=dict()):
        self.exploration_rate = hparams.get('exploration_max', 1.0)
        self.exploration_min = hparams.get('exploration_min', 0.01)
        self.exploration_decay = hparams.get('exploration_decay', 0.995)

        self.gamma = hparams.get('gamma', 0.95)

        self.batch_size = hparams.get('batch_size', 20)
        self.memory_size = hparams.get('memory_size', 1000000)

        self.action_space = architecture.layers[-1].output_shape[1]
        self.observation_space = architecture.input_shape[1]

        self.memory = deque(maxlen=self.memory_size)

        self.model = architecture


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)


