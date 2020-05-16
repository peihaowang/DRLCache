import random
import numpy as np

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @staticmethod
    def _choose_action(n_actions):
        return random.randint(0, n_actions - 1)

    def choose_action(self, observation):
        return RandomAgent._choose_action(self.n_actions)
    
    def store_transition(self, s, a, r, s_):
        pass
    
    def learn(self):
        pass


class LRUAgent:
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

    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass

class MRUAgent:
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
    
    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass


class LFUAgent:
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
    
    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass
