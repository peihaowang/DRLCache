from Cache import Cache
from PrioritizedDQN import DQNPrioritizedReplay
from ReflexAgent import RandomAgent, LRUAgent
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def train(env, RL):
    total_steps = 0
    miss_rates = []
    episodes = []
    for i_episode in range(20):
        observation = env.reset()
        while True:
            # env.render()
            observation = observation

            action = RL.choose_action(observation)

            observation_, reward = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if env.hasDone():
                print('episode %d finished.' % i_episode)
                mr = env.miss_rate()
                miss_rates.append(mr)
                episodes.append(i_episode)
                break

            if total_steps > 200:
                RL.learn()

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, miss_rates))

if __name__ == "__main__":
    # env = Cache(["data2.0/filesys/base/syn-read.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-mk-tree.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-vine.csv"], 50, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/grow-create-persistence.csv"], 5, allow_skip=False)
    env = Cache(["data2.0/zipf.csv"], 25, allow_skip=False)

    MEMORY_SIZE = 10000

    sess = tf.Session()
    with tf.variable_scope('natural_DQN'):
        RL_natural = DQNPrioritizedReplay(
            n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=False,
        )

    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = DQNPrioritizedReplay(
            n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
        )
    sess.run(tf.global_variables_initializer())

    his_natural = train(env, RL_natural)
    his_prio = train(env, RL_prio)

    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()