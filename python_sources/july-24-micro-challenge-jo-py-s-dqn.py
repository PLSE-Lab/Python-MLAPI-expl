#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# In this micro-challenge by Kaggle, I solve the blackjack optimal strategy using Deep Reinforcement Learning. Even though this might be like cracking a nut with a sledgehammer it is neverthless interesting to see if it works. Therefore, I slightly adjust the Blackjack simulator kindly provided by the Kaggle Team to be suitable for DQN learning. Finally, the simulator will test my program by playing 50,000 hands of blackjack. You'll see how frequently my program won.

# # Blackjack Rules [Kaggle]
# 
# We'll use a slightly simplified version of blackjack (aka twenty-one). In this version, there is one player (who you'll control) and a dealer. Play proceeds as follows:
# 
# - The player is dealt two face-up cards. The dealer is dealt one face-up card.
# - The player may ask to be dealt another card ('hit') as many times as they wish. If the sum of their cards exceeds 21, they lose the round immediately.
# - The dealer then deals additional cards to himself until either:
#     - The sum of the dealer's cards exceeds 21, in which case the player wins the round, or
#     - The sum of the dealer's cards is greater than or equal to 17. If the player's total is greater than the dealer's, the player wins. Otherwise, the dealer wins (even in case of a tie).
# 
# When calculating the sum of cards, Jack, Queen, and King count for 10. Aces can count as 1 or 11 (when referring to a player's "total" above, we mean the largest total that can be made without exceeding 21. So e.g. A+8 = 19, A+8+8 = 17)
# 
# 

# # The Blackjack Simulator
# 
# I use the simulator environment provided by Kaggle for this challenge:

# In[ ]:


# SETUP. You don't need to worry for now about what this code does or how it works. 
# If you're curious about the code, it's available under an open source license at https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack
print('Setup complete.')


# It simulates games between my player agent and their own dealer agent by calling the function `should_hit`

# # The Blackjack Player
# Kaggle suggests a simple strategy as an example: Always stay after the first round.

# In[ ]:


def should_hit(player_total, dealer_total , player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return False
blackjack.simulate(n_games=50000)


# Another strategy proposed by [Chris Mattler](https://www.kaggle.com/cmattler) achieves better results than the naive one above.

# In[ ]:


def should_hit(player_total, dealer_total , player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    if  player_total <= 11:
        return True
    elif player_total == 12 and (dealer_total < 4 or dealer_total > 6):
        return True
    elif player_total <= 16 and (dealer_total > 6):
        return True
    elif player_total == 17 and (dealer_total == 1):
        return True
    return False
blackjack.simulate(n_games=50000)


# Finally [Kevin Mader](https://www.kaggle.com/kmader/july-24-micro-challenge?utm_medium=social&utm_source=linkedin.com&utm_campaign=micro%20challenge%20july%2024) achieves 42.2% success rate using a Decision-Tree approach. Let's see what DQN can achieve ...

# # My Turn
# 
# I write my own `should_hit` function using Deep Reinforcement Learning. Let's first define the DQN agent class.

# In[ ]:


# Implementation of DQN largely based on the code from https://keon.io/deep-q-learning/.
import random
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

class DQNAgent:
    def __init__(self, state_size, action_size,is_eval=False,target_updateC = 0,
                 gamma=0.95,epsilon_min=0.01,epsilon_decay=0.995):
        # Game hyperparams
        self.state_size = state_size
        self.action_size = action_size
        
        # Reinforcement learning hyperparams
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = epsilon_min # base exploration rate to keep forever
        self.epsilon_decay = epsilon_decay # exploration rate decay after each experienced replay
        self.memory = deque(maxlen=2000) # Max Number of frames to remember
        self.is_eval = is_eval
        self.target_updateC = target_updateC # Threshold for updating target model
        self.C = 0 # Counting replay calls target model upate
        
        # Neural network hyperparams
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _huber_loss(self, target, prediction):
        # Error cliping between -1 and 1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
                
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse",#self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self,state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
        # The agent acts randomly
            return random.randrange(self.action_size)
        
        # Predict the reward value based on the given state
        act_values = self.model.predict(state)
        # Pick the action based on the predicted reward
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Count number of replay calls/ gradient updates
        self.C += 1
        # Every C threshold update target model.
        if self.C > self.target_updateC:
            self.C = 0
            self.update_target_model()
            
        # Experienced replay based on past memory and observations
        # Randomly sample batch from past experiences
        minibatch = random.sample(self.memory, batch_size)
        # Create training data
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            # Predict Q for unchoosen action.
            target = self.model.predict(state)
            # if the game has finished
            if done:
                target[0][action] = reward
            else:
                # use separate network for generating the discounted future reward
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            # Aggregate training data   
            states.append(state)
            targets.append(target)
        
        # Reshape to numpy array of with dim (batchsize,.)
        states=np.vstack(states)
        targets=np.vstack(targets)
       
        # Retrain the network with full batch
        self.model.fit(states, targets, epochs=5, verbose=0)

        # After every experienced replay decrease the exploration rate a little
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Next, I edit the [Blackjack simulation environment](https://github.com/Kaggle/learntools/blob/master/learntools/python/blackjack.py) provided by Kaggle to be suitable for DQN learning. More specifically, I re-write the original simulation code a little bit to observe intermediate states and rewards during each BlackJack game.

# In[ ]:


# Using the Blackjack simulation environment provided by Kaggle
from learntools.python.blackjack import BlackJack

class BlackJackEnv:
    def __init__(self):
        self.player_cards = []
        self.dealer_cards = []
    
    @property
    def player_total(self):
        return BlackJack.card_total(self.player_cards)
    @property
    def dealer_total(self):
        return BlackJack.card_total(self.dealer_cards)
    
    def _getState(self):
        # State is player_total, dealer_card_val, player_aces 
        state = [self.player_total,
                 self.dealer_total,
                 self.player_cards.count('A')] 
        return(state)
    
    def reset(self):
        # Begin game by dealing cards
        p1, p2 = BlackJack.deal(), BlackJack.deal()
        self.player_cards = [p1, p2]
        d1 = BlackJack.deal()
        self.dealer_cards = [d1]
        return(self._getState())
    
    def _play(self):
        # If player stays in game: reward == 1; else -1
        c = BlackJack.deal()
        self.player_cards.append(c)
        if self.player_total > 21:
            #Player busts! Dealer wins.
            return self._getState(),-10,True
        else:
            #Player still in the game
            return self._getState(),1,False
        
    def step(self,action):
        # action: 1 == hit, 0 == stay
        # Function returns: next_state, reward, done
        if action==1:
            # Play next card
            return(self._play())
        else:
            # Dealers turn
            while True:
                c = BlackJack.deal()
                self.dealer_cards.append(c)
                if self.dealer_total > 21:
                    #'Dealer busts! Player wins.
                    return (self._getState(),1,True)
                # Stand on 17
                elif self.dealer_total >= 17:
                    #'Dealer stands.
                    if self.dealer_total >= self.player_total:
                        #Dealer wins--> Reward is -10
                        return (self._getState(),-10,True)
                    else:
                        #Player wins.
                        return (self._getState(),1,True)


# Finally, I define the [DQN algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) which allows to simulate/play many episodes of BlackJack games during which the DQN agent learns from experience the best BlackJack strategy.

# In[ ]:


def playBlackJack(episodes,replay_batch_size,train=True,
                 is_eval=False,name="BlackJack-dqn"):
    # episodes = a number of games we want the agent to play
    # replay_batch_size = how many randomly selected experiences are used to train agent
    
    # initialize the agent
    state_size = 3 # player_total, dealer_card_val, player_aces 
    action_size = 2 #hit or stay
    env = BlackJackEnv()
    agent = DQNAgent(state_size, action_size,is_eval=is_eval)
    if is_eval:
        agent.load(name+".h5")
    done = False
    # Init empty scores list
    scores=[]
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game and return initial state
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done=False
        while not done:
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every card survived
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state .
            state = next_state
            # done becomes True when the game ends ex) The agent drops the pole
            # train the agent with the experience every x games
            if (len(agent.memory) > replay_batch_size) and train and e%10 == 0:
                agent.replay(replay_batch_size)
        # Save  scores of all games
        scores.append(reward)
        
        # Every now and then save the model and print current game performance
        if e%100 == 0:
            print("episode: {}/{}, score: {}, exploration: {:0.2f}"
                  .format(e+1, episodes, reward ,agent.epsilon))
            agent.save(name+".h5")
    #Final model saved & close
    agent.save(name+".h5")
    return(scores)


# # Let's learn & evaluate my agent

# In[ ]:


scores = playBlackJack(2000,64,train=True,
                 is_eval=False,name="BlackJack-dqn")


# ![](http://)Let's see how the DQN agent is doing using the original simulation environment from Kaggle ...

# In[ ]:


# Game params
state_size=3
action_size=2
# Init agent
agent = DQNAgent(state_size, action_size,is_eval=True)
agent.load("BlackJack-dqn.h5")
    
def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    # Reshape state
    state = np.array([player_total,dealer_card_val, player_aces])
    state = np.reshape(state, [1, state_size])
    # Decide action
    action = agent.act(state)
    return action==1

blackjack.simulate(n_games=1_000_000)


# Nice, the DQN agent also achieves about 42.4% success rate after playing only 2,000 games of Blackjack.

# # Discuss Your Results
# 
# How high can you get your win rate? We have a [discussion thread](https://www.kaggle.com/learn-forum/58735#latest-348767) to discuss your results. Or if you think you've done well, reply to our [Challenge tweet](https://twitter.com/kaggle) to let us know.

# ---
# This exercise is from the **[Python Course](https://www.kaggle.com/Learn/python)** on Kaggle Learn.
# 
# Check out **[Kaggle Learn](https://www.kaggle.com/Learn)**  for more instruction and fun exercises.
