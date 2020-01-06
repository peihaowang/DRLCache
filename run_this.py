from Cache import Cache
from DQN import DeepQNetwork


def run_maze():


    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Cache("data2.0/filesys/base/syn-read.csv", 64, allow_skip=False)
    RL = DeepQNetwork(env.n_actions, env.n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000,
        # output_graph=True
    )
    step = 0
    for episode in range(10):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            # env.display()

            # RL choose action based on observation
            action = RL.choose_action(observation["features"])

            # RL take action and get next observation and reward
            observation_, reward = env.step(action)

            # break while loop when end of this episode
            if env.hasDone():
                break

            RL.store_transition(observation["features"], action, reward, observation_["features"])

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            step += 1
        print(env.miss_rate())

    RL.plot_cost()