#!/usr/bin/env python
# coding: utf-8

# reated by: Sangwook Cheon
# 
# Date: Dec 24, 2018
# 
# This is step-by-step guide to Reinforcement Learning using scikit-learn, which I created for reference. I added some useful notes along the way to clarify things. This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of Reinforcement Learning.
# 
# ## Content:
# ### 1. Upper Confidence Bound (UCB)
# ### 2. Thomson Sampling

# # Upper Confidence Bound (UCB)
# 
# ## The multi-armed Bandit Problem
# Trying to find the optimal option among many possible options as quickly as possible, with minimal exploration of other options which have high disadvantage in money (resources) and time. For example, if an Advertising company has 5 advertisement options, they need to find the best one to publish. In order to do this, they might to AB testing. But if they do too much AB testing, then it is just as same as utilizing all 5 options which is not ideal. Therefore, through reinforcement learning, the company needs to find the optimal option quickly. 
# **Steps of solving Multi-armed Bandit problem**
# ![](https://i.imgur.com/Dn3n0ri.png)
# 
# **What's happening behind the scene**
# ![](https://i.imgur.com/n22tQ3o.png)
# The algorithm starts with an initial expecte value. Then, select a column and exploit it one time. The expected value might go down or up as new observation is taken. As there are more observations, the confidence bound gets smaller as the algorithm is more confident in the result. Then, select the column with the hiest upper confidence bound and exploit it once, which also makes the bound gets smaller and shifts the observation. Repeat these steps until the algorithm consistently exploits the same column, which is an indicator that this column is the optimal model. 
# 
# ### This is different from simple Supervised Learning, as this model starts with no data. When we start experimenting to collect data, reinforcement learning determines what to do with that existing data. 
# #### This dataset is for simulation purposes only. In real world, we cannot expect behaviors of each customer, and there is our job to start experimental strategically using UCB.
# 
# **Algorithm:**
# ![](https://i.imgur.com/NNIwROs.png)

# In[ ]:


#implementing Upper Confidence Bound (UCB)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('../input/Ads_CTR_Optimisation.csv')
dataset.head(10)


# In[ ]:


#UCB needs to be implemented from scratch without using any package, as there is no easy library to use.

N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            # 3 lines below is the algorithm shown above
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 #makes this large so that the first round gives every category a chance 
        if upper_bound > max_upper_bound:
            ad = i
            max_upper_bound = upper_bound
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualizing the result
plt.hist(ads_selected) #histogram of number of times each ad is clicked
plt.title('Histogram of ads_selected')
plt.xlabel('Ad No')
plt.ylabel('Number of times each add is selected')
plt.show()

#to view the reward
print(total_reward)
            


# As can be noticed on the graph, 4 is chosen the most times which shows that it is the best option. As it was the best option, the algorithm converged towards 4 as the number of iterations increased. The first iteration went over all the ads in order: 0, 1, 2, 3, --- ,9 so that the algorithm has at least one observation to work with.
# 
# The aim of Reinforcement learning is to maximize the total reward. Let's compare this model to Thomson Sampling.

# # Thomson Sampling
# ![](https://i.imgur.com/h63JGov.png)
# First of all, in Thomson Sampling, we are trying to guess the expected value for each distribution (which would be a bandit). Therefore, this is a probabilistic algorithm. According to initial distributions created, we then generate three random points from each distribution (creating a bandit configuration, which is sampling)
# 
# ![](https://i.imgur.com/fntToWs.png)
# Then, pick the best point which is the point that is on the far right side of the plot, as it has the highest return. Calculate it on the existing distribution and adjust the expected value and refine the distribution according to this. After iterating these steps many times, the graph will look like this:
# 
# ![](https://i.imgur.com/gXLQqBm.png)
# The distributions are refined because of more observations present. Mathematics behind this is shown below when working with code.
# 
# ## Comparison with Upper Confidence Bound (UCB)
# * While UCB is deterministic, Thomson Sampling is probabilistic
# * In UCB, the result needs to be updated every time. On the other hand, many sanples can be chosen at once, and update the model later, which is computationally more efficient. 
# * Thomson Sampling has better empirical evidence. 
# 
# 
# ## Mathematical details
# 
# 

# In[ ]:


import random

N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d #number of 0 rewards for each ad
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0 #maximum random draw
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            ad = i
            max_random = random_beta
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

plt.hist(ads_selected) #histogram of number of times each ad is clicked
plt.title('Histogram of ads_selected')
plt.xlabel('Ad No')
plt.ylabel('Number of times each add is selected')
plt.show()

#to view the reward
print(total_reward)


# Thomson Sampling is clearly better than UCB. It gave the same result: Ad number 4 is the optimal option.
