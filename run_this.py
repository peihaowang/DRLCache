from Cache import Cache
from DQN import DeepQNetwork

import random
import numpy as np

class RandomAgent:
    def __init__(self, n_actions):
        self.actions = list(range(n_actions))

    def choose_action(self, observation):
        return random.choice(self.actions)
    def store_transition(self, s, a, r, s_):
        pass
    def learn(self):
        pass


class LruAgent:
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

if __name__ == "__main__":
    # maze game
    # "data2.0/filesys/base/syn-read.csv", "data2.0/filesys/extended/dir-vine.csv"
    # env = Cache(["data2.0/filesys/base/syn-read.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-mk-tree.csv"], 5, allow_skip=False)
    env = Cache(["data2.0/filesys/extended/dir-vine.csv"], 10, allow_skip=False)
    RL = DeepQNetwork(env.n_actions, env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        # output_graph=True
    )
    # RL = RandomAgent(env.n_actions)
    # RL = LruAgent(env.n_actions)
    step = 0
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            # env.display()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward = env.step(action)

            # break while loop when end of this episode
            if env.hasDone():
                break

            RL.store_transition(observation, action, reward, observation_)

            if (step > 20) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            step += 1
        print(env.miss_rate())

    RL.plot_cost()