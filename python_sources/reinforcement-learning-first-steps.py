#!/usr/bin/env python
# coding: utf-8

# ### Goal
# 
# Reinforcement learning is a very interesting area which studies creation of self-educating agents, which can solve different tasks in different environments. In this kernel I'm going to solve classic OpenAI Gym CartPole-v0 environment using cross-entropy method. So let's start.

# In[ ]:


import gym
import numpy as np
import matplotlib.pyplot as plt
import requests

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import he_normal

from PIL import Image
from io import BytesIO


# The goal of the environment is to hold pole in vertical position moving cart left or right. The game is completed if the pole is balanced for 200 episodes. The environment looks like this:

# In[ ]:


r = requests.get('https://gym.openai.com/videos/2019-10-21--mqt8Qj1mwo/CartPole-v1/poster.jpg')

plt.figure(figsize = (10, 6))
plt.imshow(Image.open(BytesIO(r.content)))
plt.axis('off')
plt.show()


# Let's create an environment and look at it in details.

# In[ ]:


env = gym.make('CartPole-v0') # Create an environment

# To reset the environment we can use reset() function which returns an array with 4 values
# This 4 values is an observation, which tells us a position of pole. We don't need to know what these values mean
# this is a job for our ANN.
observation = env.reset()
print(f'Observation, returned by reset() function: {observation}')

# To see action space we can use action_space attribute
# Discrete(2) means that actions can be 0 or 1 which can be left or right 
print('Action space: ', env.action_space)

# To make step we need to use step(action) function, it returns 4 values:
# Obseravtion - current observation after action
# Reward - recieved reward after action
# Done - whether game over or not
# Debug data which we don't need
observation, reward, done, debug = env.step(env.action_space.sample()) # Doing random action
print(f'Observation after action: {observation}')
print(f'Reward for the action: {reward}')
print(f'Game is over: {done}')
print(f'Debug data: {debug}')


# ### Algorithm
# 
# To solve this environment a cross-entropy algorithm have been used. The algorithm can be written in several steps:
# 1. Play N numbers of games using random actions or actions predicted by model to collect raw data.
# 2. Collect a total reward for each game and calculate threshold - 70 percentile of all total rewards.
# 3. Select games from raw data which have total reward more than threshold.
# 4. Train model on selected data where observations are an input to the model and actions are targets.
# 5. Repeat from step 1 untill good results.
# 
# In this kernel I want to train model with two steps - first, on data, generated using random actions, second, on data, generated using actions, predicted by the model.

# ### Algorithm implementation
# 
# Let's start coding. The main piece of code - is a function that will generate and preprocess data.

# In[ ]:


def generate_data(env = gym.make('CartPole-v0'), n_games = 1000, model = None, percentile = 70):
    '''
       env - an environment to solve
       n_games - number of games to play to collect raw data
       model - if None, the random actions will be taken to collect data, to predict actions a model must be passed
       percentile - (100% - percentile%) of the best games will be selected as training data
    '''
    observation = env.reset() # Resetting the environment to get our first observation

    train_data = [] # List to store raw data
    rewards = [] # List to store total rewards of each game
    
    print(f'Playing {n_games} games...')
    
    # Step 1 of the algorithm - Play N numbers of games using random actions or actions predicted by model to collect raw data.
    for i in range(n_games):
        temp_reward = 0 # Counts a current game total reward
        temp_data = [] # Stores (observation, action) tuples for each step
        
        # Playing a current game until done
        while True:
            # Use model to predict actions if passed, otherwise take random actions
            if model:
                action = model.predict(observation.reshape((-1, 4)))
                action = int(np.rint(action))
            else:
                action = env.action_space.sample()
            
            temp_data.append((list(observation), action)) # Appending (observation, action) tuple to temp_data list

            observation, reward, done, _ = env.step(action) # Making action

            temp_reward += reward # Counting reward
            
            # If game over - reset environment and break while loop
            if done:
                observation = env.reset()
                break
        
        # Append data of last game to train_data list and total reward of last game to rewards list
        train_data.append(temp_data)
        
        # Step 2 of the algorithm - Collect a total reward for each game and calculate threshold - 70 percentile of all total rewards.
        rewards.append(temp_reward)
        
    print('Done playing games\n')
    
    # Calculating threshold value using rewards list an np.percentile function
    thresh = int(np.percentile(rewards, percentile))
    print(f'Score threshold value: {thresh}')
    
    print(f'Selecting games according to threshold...')
    # Step 3 of the algorithm - Select games from raw data which have total reward more than threshold.
    train_data = [episode for (i, episode) in enumerate(train_data) if rewards[i] >= thresh]
    
    # Now train_data list contains lists of tuples: [[(observation, action), ...], [(observation, action), ...], ...]
    # The next string flattens train_data list: [(observation, action), (observation, action), ...]
    train_data = [observation for episode in train_data for observation in episode]
    
    # Creating labels array
    labels = np.array([observation[1] for observation in train_data])
    
    # Storing only observations in train_data array
    train_data = np.array([observation[0] for observation in train_data])
    print(f'Total observations: {train_data.shape[0]}' )
    
    return train_data, labels


# In[ ]:


# Generating first training data
train_data, labels = generate_data(n_games = 2000)


# ### Model creation
# This is step 4 of our algorithm - **train model on selected data where observations are an input to the model and actions are targets**.
# 
# As a model I'll use simple ANN with two hidden layers (64, and 128 neurons).

# In[ ]:


# Weights initializer
init = he_normal(seed = 666)

model = Sequential()

# We are using observations from environment as input data, so input shape of our ANN is (4, )
model.add(Dense(64, input_shape = (4,), activation = 'relu', kernel_initializer = init))
model.add(Dense(128, activation = 'relu', kernel_initializer = init))

# Because our action can be only 0 or 1, I'll use Dense layer with one neuron and sigmoid activation function
model.add(Dense(1, activation = 'sigmoid'))

# Compile model using SGD and binary_crossentropy
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy')


# ### Model training - sample actions data
# 
# First - I'll train model using data, generated on random actions. To plot loss I'll also create a plot_loss function:

# In[ ]:


def plot_loss():    
    H = model.history.history
    
    plt.figure(figsize = (15, 5))
    plt.plot(H['loss'], label = 'loss')
    plt.plot(H['val_loss'], label = 'val_loss')
    plt.grid()
    plt.legend()
    plt.show()


# In[ ]:


# Model training
model.fit(train_data, labels, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)
plot_loss()


# Our task here is to minimize a loss. If model is tend to overfit - then additional regularization must be added, but here I'll leave it as is, to keep things simple.
# 
# To see what your model is doing, you can uncomment and run next piece of code which will play 3 games:

# In[ ]:


# env = gym.make('CartPole-v0')
# observation = env.reset()

# for i in range(3):
#     temp_reward = 0
#     while True:
#         env.render()

#         action = model.predict(observation.reshape((-1, 4)))
#         action = int(np.rint(action))       

#         observation, reward, done, _ = env.step(action)

#         temp_reward += reward
        
#         if done:
#             print(temp_reward)
#             observation = env.reset()
#             break

# env.close()


# ### Round 2 - model predicted actions
# 
# Now I want to return to step 1 of our algorithm, but now I want to use predictions of our model as actions, when generating data.

# In[ ]:


# Generating data using actions, predicted by the model
train_data, labels = generate_data(model = model)


# Now our score threshold is much higher and we are taking only those games, which have total reward equal or higher than threshold.
# 
# Next - we train our model on new data.

# In[ ]:


# Train model on new data
model.fit(train_data, labels, epochs = 30, batch_size = 32, validation_split = 0.2, verbose = 0)
plot_loss()


# ### Results
# 
# To show results of training, I made a video which you can watch on YouTube:
# 
# https://www.youtube.com/watch?v=YygkJM13UTM&t=41s
# 
# Every 10 epochs of training, I played 3 games and recorded results, so we can see how our agent improves during training process.

# ### Conclusion
# 
# And this is all here. The cross-entropy method is very simple, I'd rather say it's primitive, but it works very well for simple tasks like CartPole. 
# 
# Of course the code here is just a baseline and can be easly improved or written as a convenient class, but my main goal was to show you the algorithm and implement it step by step, keeping things as simple as possible, so I hope it will help anybody who doing first steps in deep reinforcement learning area.

# In[ ]:




