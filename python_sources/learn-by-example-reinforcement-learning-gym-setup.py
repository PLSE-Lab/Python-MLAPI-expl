#!/usr/bin/env python
# coding: utf-8

# # Setup Gym with Atari games on Kaggle
# Welcome to my notebook on Kaggle. I did record my notes so it might help others in their journey to understand Neural Networks by examples (in this case Reinforcement Learning with Gym for Atari games from OpenAI). 
# 
# This notebook is to show how to set-up gym including the gym[atari] environments in Kaggle so you can get to work and try to beat all those Atari 2600 games like Pong, Pacman, Spaceinvaders, etc
# 
# If you are interested in Reinforcement Learning you might be interested in some of my other notebooks:  
# https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym  
# If you are new to Neural Networks you might want to have a look at my notebooks below:    
# https://www.kaggle.com/charel/learn-neural-networks-by-example-mnist-digits  
# https://www.kaggle.com/charel/learn-by-example-rnn-lstm-gru-time-series  

# # Gym  
# In 2014 Google DeepMind published a paper titled "Playing Atari with Deep Reinforcement Learning" that can play Atari 2600 games at expert human levels. This was the first breakthrough in applying deep neural networks for reinforcement learning.
# ![alt text](https://cdn-images-1.medium.com/max/800/1*3ZgGbUpEyAZb9POWijRq4Q.png)
# 
# 
# Gym is released by Open AI in 2016 (http://gym.openai.com/docs/). It is a toolkit for developing and comparing reinforcement learning algorithms.
# <img src="https://i.imgur.com/ria9HOm.jpg%20" width=800>
# 
# Source: [OpenAI](https://openai.com/)
# 
# In 2018 Gym-retro was released as its successor: https://blog.openai.com/gym-retro/
# 
# There are many many games made available. and you need to install the gym[atari] environment to make the Atari games available. In Kaggle you have to enable the Internet-beta feature available to make it internet connected (see to the left of the screen in the settings) and to be able to install the required packages
# ![alt text](https://cdn-images-1.medium.com/max/800/1*vUMIoHkl-PuIjbTqbtn8dA.png)
# 
# 

# In[ ]:


get_ipython().system('pip install gym ')
get_ipython().system("pip install 'gym[box2d]'")
get_ipython().system('pip install atari_py')


# In[ ]:


import gym
from gym import wrappers
from gym import envs
import numpy as np 
#import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import time
import os


# ## Show game screens
# Let's show some game screens of these iconic games

# ## Breakout
# 
# ![alt text](http://www.atarimania.com/2600/boxes/hi_res/breakout_color_cart_4.jpg)
# 
# 
# Source: [Atari mania](http://www.atarimania.com/game-atari-2600-vcs-breakout_18135.html)
# 

# In[ ]:


env = gym.make("BreakoutNoFrameskip-v4")
plt.imshow(env.render('rgb_array'))
plt.grid(False)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# ## Pacman
# 
# ![alt text](http://www.atarimania.com/2600/boxes/hi_res/ms_pac_man_silver_1987_cart_7.jpg)
# 
# 
# Source: [Atari mania](http://www.atarimania.com/game-atari-2600-vcs-ms-pac-man_7391.html)
# 

# In[ ]:


env = gym.make("MsPacmanNoFrameskip-v4")
plt.imshow(env.render('rgb_array'))
plt.grid(False)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# ## Spaceinvaders
# 
# ![alt text](http://www.atarimania.com/2600/boxes/hi_res/space_invaders_silver_1986_cart_2.jpg)
# 
# 
# Source: [Atari mania](http://www.atarimania.com/game-atari-2600-vcs-space-invaders_8102.html/)
# 

# In[ ]:


env = gym.make("SpaceInvadersNoFrameskip-v4")
plt.imshow(env.render('rgb_array'))
plt.grid(False)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# ## Capture a video
# Below the code to set-up a virtual monitor and capture a movie

# In[ ]:


# Set-up the virtual display environment
get_ipython().system('apt-get update')
get_ipython().system('apt-get install python-opengl -y')
get_ipython().system('apt install xvfb -y')
get_ipython().system('pip install pyvirtualdisplay')
get_ipython().system('pip install piglet')
get_ipython().system('apt-get install ffmpeg -y')


# In[ ]:


# Start the virtual monitor
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


# In[ ]:


# play a random game and create video
env = gym.make("MsPacmanNoFrameskip-v4")
monitor_dir = os.getcwd()

#Setup a wrapper to be able to record a video of the game
record_video = True
should_record = lambda i: record_video
env = wrappers.Monitor(env, monitor_dir, video_callable=should_record, force=True)

#Play a random game
state = env.reset()
done = False
while not done:
  action = env.action_space.sample() #random action, replace by the prediction of the model
  state, reward, done, _ = env.step(action)

record_video = False
env.close() 

# download videos
#from google.colab import files
#import glob
os.chdir(monitor_dir) # change directory to get the files
get_ipython().system('pwd #show file path')
get_ipython().system('ls # show directory content')


# ## Play video
# 
# The MP4 file is already generated and is part of the output files. Go to your navigation bar at the top/left of the screen to the tab Output and click on the .mp4 file. 
# 

# ## Reinforcement learning training
# After the first publication of DQN many deeplearning Reinforcement Learning algorithms have been invented/tried, Some main ones in chronological order: DQN, Double DQN, Duelling DQN, Deep Deterministic Policy Gradient, Continuous DQN (CDQN or NAF) , A2C/A3C, Proximal Policy Optimization Algorithms, ARS, etc, etc. 
# 
# For the actual (long, millions of game screens) training I would like to recommend [colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) solutions, since Kaggle is owned by Alphabet I presume they don't mind I refer to Google colab solutions. Will keep posting some links over here:
# 
# Bipedal walker  with Augmented Random Search: 
# * Colab: https://colab.research.google.com/drive/1NxvslFQ6RDLBirlOYH4b77zabSW_x5DM   
# * Video: 	https://www.youtube.com/watch?v=NJBgeOF5CnM  
# 
# Ms Pacman with Proximal Policy Optimization:   
# * Colab: https://colab.research.google.com/drive/1aSoqbO_wysvciYfDv4hJbL6ZS7lXFnMN   
# * video after training 20 Million game screens:  https://youtu.be/um2XX5bktMA  )  
# 
# 
# Hoped you liked my notebook (upvote top right), my way to conribute back to this fantastic Kaggle platform and community.
