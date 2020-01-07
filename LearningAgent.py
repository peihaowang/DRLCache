'''
Base class of learning based agents.
'''

class LearningAgent:
    def __init__(self, n_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

    def choose_action(self, observation):
        '''
        Gives back a decision based on observation.
        '''
        pass

    def learn(self, *args):
        '''
        Learn from rewards after performing an action.
        '''
        pass
