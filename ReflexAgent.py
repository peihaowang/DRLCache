import random
import numpy as np

class RandomAgent:
    def __init__(self, n_actions):
        self.actions = np.arange(n_actions)

    def choose_action(self, observation):
        return random.choice(self.actions)
    def store_transition(self, s, a, r, s_):
        pass
    def learn(self):
        pass


class LRUAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, observation):
        used_times = np.array(observation['last_used_times'])
        min_idx = np.argmin(used_times)
        if min_idx < 0 or min_idx > self.n_actions:
            raise ValueError("LruAgent: Error index %d" % min_idx)
        return min_idx
    def store_transition(self, s, a, r, s_):
        pass
    def learn(self):
        pass

class MRUAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, observation):
        used_times = np.array(observation['last_used_times'])
        max_idx = np.argmax(used_times)
        if max_idx < 0 or max_idx > self.n_actions:
            raise ValueError("MruAgent: Error index %d" % max_idx)
        return max_idx
    def store_transition(self, s, a, r, s_):
        pass
    def learn(self):
        pass


class LFUAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, observation):
        used_times = np.array(observation['last_used_times'])
        max_idx = np.argmax(used_times)
        if max_idx < 0 or max_idx > self.n_actions:
            raise ValueError("LfuAgent: Error index %d" % max_idx)
        return max_idx
    def store_transition(self, s, a, r, s_):
        pass
    def learn(self):
        pass
