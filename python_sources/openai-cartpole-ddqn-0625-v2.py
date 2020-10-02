# -*- coding: utf-8 -*-
# This is a modified implementation of the DDQN algorithm posted at https://github.com/keon/deep-q-learning

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.4   # discount rate
        self.gamma_max = 0.9
        self.gamma_inc = 1.005
        self.epsilon = 0.99 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.993
        self.learning_rate = 0.0025
        self.learning_rate_min = 0.001
        self.learning_rate_decay = 0.9992
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(3, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay
        if self.gamma < self.gamma_max:
            self.gamma *= self.gamma_inc

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
# agent.load("cartpole-ddqn.h5")
done = False
batch_size =  30
maxstep=0
    
EPISODES = 300
STEPS = 300
Solved = False
solvecount=0
    
for e in range(EPISODES):
    state = env.reset()
    print("episode: {}/{}: {}"
         .format(e, EPISODES, state))
    state = np.reshape(state, [1, state_size])
    for step in range(STEPS):
        #if Solved:
        #    env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward# if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size and Solved==False:
            agent.replay(batch_size)
        if done or step==STEPS-1:
            if step>=maxstep:
                maxstep=step+1
            if step==STEPS-1:
                solvecount+=1
                if solvecount==2:
                    Solved=True
                    agent.model.set_weights(agent.target_model.get_weights())
                    #agent.save("cartpole-ddqn-3.h5")
            print("score: {}, max: {}, e: {:.2}, gamma:{:.2}, lr:{:.2}, memo:{}, solve:{}"
                 .format(step+1, maxstep, agent.epsilon, agent.gamma, agent.learning_rate,len(agent.memory),Solved))
            if Solved==False:
                agent.update_target_model()
            break