import random
import numpy as np
from CacheAgent import ReflexAgent

class RandomAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(n_actions):
        return random.randint(0, n_actions - 1)

    def choose_action(self, observation):
        return RandomAgent._choose_action(self.n_actions)

class LRUAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(observation):
        used_times = np.array(observation['last_used_times'])
        min_idx = np.argmin(used_times)
        return min_idx

    def choose_action(self, observation):
        min_idx = LRUAgent._choose_action(observation)
        if min_idx < 0 or min_idx > self.n_actions:
            raise ValueError("LRUAgent: Error index %d" % min_idx)
        return min_idx

class MRUAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(observation):
        used_times = np.array(observation['last_used_times'])
        max_idx = np.argmax(used_times)
        return max_idx

    def choose_action(self, observation):
        max_idx = MRUAgent._choose_action(observation)
        if max_idx < 0 or max_idx > self.n_actions:
            raise ValueError("MRUAgent: Error index %d" % max_idx)
        return max_idx

class LFUAgent(ReflexAgent):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        
    @staticmethod
    def _choose_action(observation):
        freq = observation['total_use_frequency']
        min_idx = np.argmin(freq)
        return min_idx

    def choose_action(self, observation):
        min_idx = LFUAgent._choose_action(observation)
        if min_idx < 0 or min_idx > self.n_actions:
            raise ValueError("LFUAgent: Error index %d" % max_idx)
        return min_idx
