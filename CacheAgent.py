import sys

# Abstract class
class CacheAgent(object):
    def __init__(self, n_actions): pass
    def choose_action(self, observation): pass
    def store_transition(self, s, a, r, s_): pass
    
class ReflexAgent(CacheAgent):
    def __init__(self, n_actions): pass

    @staticmethod
    def _choose_action(n_actions): pass

class LearnerAgent(CacheAgent):
    def __init__(self, n_actions): pass
    def learn(self): pass