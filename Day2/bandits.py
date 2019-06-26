import time
import numpy as np


class ContextualBandit(object):

    def __init__(self, n, contexts, rewards):
        assert rewards.shape[1] == n
        self.n = n
        self.contexts = contexts
        self.rewards = rewards
        self.context_dim = contexts.shape[1]

    def best_action(self, context):
        # Best action in hindsight
        idx = np.where(self.contexts == context)[0][0]
        return np.argmax(self.rewards[idx])

    def best_reward(self, context):
        # Best action in hindsight
        idc = np.where(self.contexts == context)[0][0]
        return np.max(self.rewards[idc])

    def generate_reward(self, context, action):
        # Reward of playing action while in context
        idx = np.where(self.contexts == context)[0][0]
        return self.rewards[idx][action]
    
    def generate_context(self, num=1):
        idx = np.random.choice(range(self.contexts.shape[0]), num)
        return self.contexts[idx]

    

