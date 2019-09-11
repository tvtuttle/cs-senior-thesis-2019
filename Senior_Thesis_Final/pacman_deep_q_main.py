# A main class, which when run creates a deep_q_agent and uses it to learn pacman
from deep_q_agent import DeepQAgent
import gym

def main():
    env = gym.make('MsPacman-v0')
    agent = DeepQAgent(env, trials=500, eps_start=0.01, eps_min=0.01,
                       in_model="pacman_test_2.h5", in_scores="pickled_scores_test_2.m",
                       in_mems="pickled_memories_test_2.m", out_model="pacman_test_3.h5",
                       out_scores="pickled_scores_test_3.m", out_mems="pickled_memories_test_3.m",
                       model="dq3") # to test different parameters, submit them as arguments to this constructor
    # for example, here i lowered the target frequency in order to save data more often for testing, specified existing
    # input models and outputs for where the more trained models should be, the model to be used, the number of trials and
    # epsilon values to use here

    trials = agent.trials
    step = 0
    update_target = agent.target_freq  # how long between target network updates
    for trial in range(trials):
        print(trial)
        rough_state = env.reset()
        cur_state = agent.preprocess(rough_state)
        done = False
        cum_reward = 0
        while not done:
            step += 1
            action = agent.act(cur_state)
            env.render()
            new_rough_state, reward, done, info = env.step(action)
            # change reward: eating a pellet is worth 10 pts, we'll make any positive reward worth that amount
            if reward > 10:
                reward = 10
            new_state = agent.preprocess(new_rough_state)

            agent.remember(cur_state, action, reward, new_state, done)

            cur_state = new_state
            cum_reward += reward
            if (step > agent.learning_start):
                agent.replay(agent.batch_size)
                if (step % update_target) == 0:
                    print("target updated, epsilon=" + str(agent.epsilon))
                    agent.target_train()
                    agent.save_model()
        agent.scores.append(cum_reward)
    agent.save_model() # at the end of all trials


if __name__ == '__main__':
    main()