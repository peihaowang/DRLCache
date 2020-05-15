from Cache import Cache
from DQN import DeepQNetwork
from ReflexAgent import RandomAgent, LRUAgent

if __name__ == "__main__":
    # cache
    env = Cache(["data2.0/zipf2.csv"], 50, allow_skip=False)
    # "data2.0/filesys/base/syn-read.csv", "data2.0/filesys/extended/dir-vine.csv"
    # env = Cache(["data2.0/filesys/base/syn-read.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-mk-tree.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-vine.csv"], 50, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/grow-create-persistence.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/zipf.csv"], 5, allow_skip=False)
    RL = DeepQNetwork(env.n_actions, env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.8,
        replace_target_iter=1,
        memory_size=10000,
        batch_size=128
        # output_graph=True
    )
#     RL = RandomAgent(env.n_actions)
    RL = LRUAgent(env.n_actions)
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

            if (step > 20) and (step % 1 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            
            if step % 1000 == 0:
                mr = env.miss_rate()

            step += 1
        print(env.miss_rate())

    RL.plot_cost()