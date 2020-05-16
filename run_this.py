from Cache import Cache
from DQN import DeepQNetwork
from ReflexAgent import RandomAgent, LRUAgent

if __name__ == "__main__":
    # cache
#     env = Cache(["data2.0/zipf2.csv"], 50, allow_skip=False)
    # "data2.0/filesys/base/syn-read.csv", "data2.0/filesys/extended/dir-vine.csv"
    env = Cache(["data2.0/filesys/base/syn-read.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-mk-tree.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/dir-vine.csv"], 50, allow_skip=False)
    # env = Cache(["data2.0/filesys/extended/grow-create-persistence.csv"], 5, allow_skip=False)
    # env = Cache(["data2.0/zipf.csv"], 5, allow_skip=False)
    RL = DeepQNetwork(env.n_actions, env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        
        # Epsilon greedy
        e_greedy_min=(0.0, 0.1),
        e_greedy_max=(0.5, 0.5),
        e_greedy_init=(0.2, 0.8),
        e_greedy_increment=(0.005, 0.01),
        e_greedy_decrement=(0.005, 0.001),
        e_greedy_threshold=50,
        explore_mentor = 'LRU',

        replace_target_iter=100,
        memory_size=10000,
        history_size=50,
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
            
            if step % 100 == 0:
                mr = env.miss_rate()
                print("Episode=%d, Step=%d: NumAccesses=%d, NumHits=%d, MissRate=%f"
                    % (episode, step, env.total_count, env.miss_count, mr)
                )

            step += 1

    RL.plot_cost()