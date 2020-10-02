#!/usr/bin/env python
# coding: utf-8

# Are there any inherent, and possibly surprising advantages that we can detect in the UFC data?
# 
# Exploration of the basic fighter features importance (height. weight, etc.)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
plt.style.use('fivethirtyeight')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = "../input/"
filename = 'finalout.csv'
df = pd.read_csv(path + filename)
df = df.fillna(0)


# In[ ]:



df = df[df.winby != 0]
df['blue_win_flag'] = df.winner.apply(lambda x: 1 if x == 'blue' else 0)
df['winner_height'] = df.B_Height*df.blue_win_flag + (1 - df.blue_win_flag)*df.R_Height
df['loser_height'] = df.B_Height*(1 - df.blue_win_flag) + df.blue_win_flag*df.R_Height
df['height_diff'] = df.winner_height - df.loser_height


df['winner_weight'] = df.B_Weight*df.blue_win_flag + (1 - df.blue_win_flag)*df.R_Weight
df['loser_weight'] = df.B_Weight*(1 - df.blue_win_flag) + df.blue_win_flag*df.R_Weight
df['weight_diff'] = df.winner_weight - df.loser_weight


df['winner_age'] = df.B_Age*df.blue_win_flag + (1 - df.blue_win_flag)*df.R_Age
df['loser_age'] = df.B_Age*(1 - df.blue_win_flag) + df.blue_win_flag*df.R_Age
df['age_diff'] = df.winner_age - df.loser_age


# ## Red vs Blue

# Karmanya Aggarwal showed in his [script][1] that the red corner fighter is more likely to win. Is this statistically significant, and if so, what might be the root cause?
# 
# I will estimate the significant similarly to what I used in this [script][2] - by simulating the equivalent amount of coin tosses, and assess how much of an outlier our result is.
# 
# 
#   [1]: https://www.kaggle.com/calmdownkarm/ufc-predictor-and-notes
#   [2]: https://www.kaggle.com/drgilermo/right-or-left-which-is-better

# In[ ]:


rand = np.random.binomial(len(df), 0.5, 100000)
plt.hist(rand, bins = 300)
plt.plot(len(df[df.winner == 'red']),0,'o', markersize = 20, color = 'r')
plt.xlabel('Number of wins for the red gloves fighter')
plt.ylabel('Number of simulations')
plt.legend(['# of actual red fighter wins','Binomial Distribution'], loc = 2)


# ## The red corner is more likely to win - but this is not surprising ##
# 
# The blue corner fighter will be introduced first, and only then the red gloves fighter enters the arena to the sound of the crowd cheering. The red fighter is therefore usually the one who is either ranked higher or has more hype around him.
# 
# and indeed this result as statistically very significant

# In[ ]:


print(np.true_divide(len(df[df.winner =='red']),len(df)))


# ## Height advantage
# 
# Taller fighters have what is called the "reach advantage", their ability to strike (usually by kicking) their opponent from distance, as well as denying take-down attempts. This definitely helped [Conor McGrego][1] knocking down the short Eddie Alvarez after a failing punching attempt by the former)
# 
# But is the effect strong enough to correlate with higher winning chances?
# 
#   [1]: https://www.youtube.com/watch?v=hLAWQcqt1wU

# In[ ]:


height_df = df[df.height_diff!= 0]
rand = np.random.binomial(len(height_df[height_df.height_diff!= 0]), 0.5, 1000000)

plt.hist(rand, bins = 300)

plt.plot(len(height_df[height_df.height_diff > 0]),0,'o', markersize = 20, color = 'r')
plt.legend(['# of actual taller fighter wins','Binomial Distribution'], loc = 2)
plt.xlabel('Number of wins for the taller fighter')
plt.ylabel('Number of simulations')
plt.show()

print(np.true_divide(len(height_df[height_df.height_diff > 0]),len(height_df)))
print(len(height_df))


# Not really, the tiny effect is obviously a random one. This is interesting, but I think that the answer is related to the next feature

# ## Weight Advatnage

# In[ ]:


weight_df = df[df.weight_diff!= 0]
rand = np.random.binomial(len(weight_df), 0.5, 1000000)
plt.hist(rand, bins = 300)
plt.xlabel('Number of wins for the heavier fighter')
plt.ylabel('Number of simulations')
plt.plot(len(weight_df[weight_df.weight_diff > 0]),0,'o', markersize = 20, color = 'r')
plt.legend(['# of actual heavier fighter wins','Binomial Distribution'], loc = 2)
plt.show()

print(np.true_divide(len(weight_df[weight_df.weight_diff > 0]),len(weight_df)))


# We see that the heavier fighter has indeed an advantage which looks fairly real. but we also see that the number of fights in our histogram has diminshed dramatically? why is that?
# 
# The reason can be seen in the next code blocks/plots: usually the fighters are equally heavy - they take as much kg as the weight class allows them. only in the Heavy Weight category, where the weight is unlimited, we see some deviations (and an advantage for the heavier fighters)

# In[ ]:


plt.hist(df.weight_diff, bins = np.arange(-10,10,1))
plt.xlabel('Weight Difference between winner and loser')
plt.show()


# Let's see how the weight distributes around the weight classes limits:

# In[ ]:


plt.hist(df.R_Weight, bins = np.arange(50,120,1))
weights = [52.2,56.7,61.2,65.8,70.3,77.1,83.9,93]
plt.plot(weights,np.zeros(len(weights)),'o', markersize = 20)
plt.legend(['Weight classes','Weight distribution'])
plt.xlabel('Weight')
plt.show()


# Most fighters actually weigh more than their class limit, and then by dehydration reach to the required weight just before the fight.
# 
# This might explain why height is not an advantage even though having all equal, being taller gives the fighter the reach advantage. Since almost always the fighters have the same weight, for every  additional centimeter, they need to compensate with a slimmer body. and since in most weight classes the fighters have relatively low fat reserves, this means less muscle.

# ## Age
# 
# Last but no the least - age. it sounds reasonable that being relatively young is better. however, older fighters have more experience, they can be stronger, and they have the survival bias - if a fighter made it to his 30's, he's probably good.
# 
# However, the data shows that in UFC, as it is in life, it's better to be younger:

# In[ ]:


age_df = df[df.age_diff!= 0]
rand = np.random.binomial(len(age_df), 0.5, 1000000)
plt.hist(rand, bins = 300)
plt.xlabel('Number of wins for the older fighter')
plt.ylabel('Number of simulations')
plt.plot(len(age_df[age_df.age_diff > 0]),0,'o', markersize = 20, color = 'r')
plt.legend(['# of actual older fighter wins','Binomial Distribution'], loc = 2)
plt.show()

print(np.true_divide(len(age_df[age_df.age_diff > 0]),len(age_df)))


# In[ ]:


df['Round_2_age'] = np.true_divide(df.age_diff,2)
df['Round_2_age'] = df['Round_2_age'].apply(lambda x: 2*round(x))
age_diff = np.arange(0,10,2)
prob = []

for age in age_diff:
    pos_age = age
    neg_age = -age
    pos_wins = len(df[df.Round_2_age == pos_age])
    neg_wins = len(df[df.Round_2_age == neg_age])
    prob.append(np.true_divide(pos_wins,pos_wins + neg_wins))

    
plt.bar(2*np.ones(len(prob))*range(len(prob)),prob,width = 1.8)
plt.xticks(2*np.ones(len(prob))*range(len(prob)),['0 - 2','2 - 4','4 - 6','6 - 8','8 - 10'])
plt.ylabel('Winning probability')
plt.xlabel('Age Difference')
plt.show()


# In[ ]:




