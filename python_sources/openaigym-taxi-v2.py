import numpy as np
import gym
import random
import os
import time

env = gym.make("Taxi-v2")

env.render()

action_size = env.action_space.n
state_size = env.observation_space.n
print("Acion size: ", action_size)
print("State size: ", state_size)
print("\n")


qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 50000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01


for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:  # then do exploitation
            action = np.argmax(qtable[state, :])
        else:  # exploration
            action = env.action_space.sample()

        # do the action
        new_state, reward, done, info = env.step(action)

        qtable[state, action] += learning_rate * \
            (reward+gamma*np.max(qtable[new_state, :] - qtable[state, action]))

        state = new_state

        if done == True:
            break

    episode += 1
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
        np.exp(-decay_rate * epsilon)


# play

env.reset()
rewards = []

for episodes in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("*********************")
    print("Episode ", episode)

    for step in range(max_steps):
        env.render()
        time.sleep(0.5)
        os.system("cls")
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if(done):
            rewards.append(total_rewards)
            print("Episode Over: Score: ", total_rewards)
            break
        state = new_state

env.close()
print("Score overtime: ", str(sum(rewards)/total_test_episodes))
