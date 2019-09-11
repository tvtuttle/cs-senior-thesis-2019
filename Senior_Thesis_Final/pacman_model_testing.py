# Used for testing saved models in the "models" folder for a given number of trials
# To test a purely random model, choose any model and set "eps_min" and "eps_start" to 1.0
# To test a model without any random moves, set those same values to 0 instead
# At end of runtime, average score of runs is printed as output
from deep_q_agent import DeepQAgent
import gym

def main():
    env = gym.make("MsPacman-v0")
    agent = DeepQAgent(env, trials=100, eps_min=0, eps_start=0, in_model="pacman_dq3_norm_cont.h5") # specify model tested here as well
    episodes = agent.trials
    rewards = list()

    for episode in range(episodes):
        done = False
        rough_state = env.reset()
        cur_state = agent.preprocess(img=rough_state)
        step = 0
        cum_reward = 0
        while not done:
            action = agent.act(cur_state)
            env.render()
            new_rough_state, reward, done, info = env.step(action)
            new_state = agent.preprocess(new_rough_state)
            cum_reward += reward
            cur_state = new_state
        rewards.append(cum_reward)
    print(rewards)
    print(sum(rewards) / len(rewards))


if __name__ == "__main__":
    main()