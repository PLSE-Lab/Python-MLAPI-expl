#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Why this analysis?
# 
# Before going on to the statistical modelling part of the problem, one must make sure that the data he is going to train his model on the data which must be somewhat similar to the test data. Not only is this a good practice for production, but also of course, for this particular competition

# In[ ]:


df_train = pd.read_csv('../input/train_V2.csv', nrows = 100000)
df_test = pd.read_csv('../input/test_V2.csv', nrows = 100000)


# In[ ]:


df_train.head()


# In[ ]:


# importing important libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Utility Function 1
# 
# This will give me a ready to plot dataframe which will be contataining **Mean Line** , **Upper Control Limit** and **Lower Control Limit**.
# 
# ### Before we move on, what are upper and lower control limits?
# 
# **The control limits of your control chart represent your process variation and help indicate when your process is out of control.**
# Control limits are the horizontal lines above and below the center line that are used to judge whether a process is out of control. The upper and lower control limits are based on the random variation in the process. By default, our control limits are displayed 3 standard deviations above and below the center line.
# 
# ### One may arguably ask, isn't control limits strictly for time based data?
# 
# Well, lets just assume here that our index are strictly in an increasing period of time. That may or not be the best assumption yet, but let's keep moving for now

# In[ ]:


# utility function 1
def mean_dev_stuff(df, column_name, limit = 20000):
    if len(df) > limit:
        df = df[:limit]
    
    data_bar = df[column_name].mean()
    df[column_name+"_squaredmean"] = (df[column_name] - data_bar)**2
    std_dev = (sum(df[column_name+"_squaredmean"])/len(df))**0.5
    
    UCL = np.array([data_bar + 3 * std_dev]*len(df))
    LCL = np.array([data_bar - 3 * std_dev]*len(df))
    mean_line = np.array([data_bar]*len(df))
    
    new_df = pd.DataFrame({column_name : df[column_name], "UCL" : UCL, "LCL" : LCL, "Mean" : mean_line})
    return new_df


# # Data Check One : The amount of kills
# 
# Let's have a look at the killing statistics of our players and see of it needs some tweaking or not!

# In[ ]:


# killing definition function
killdf_train = mean_dev_stuff(df_train, 'kills', 20000)
killdf_test = mean_dev_stuff(df_test, 'kills', 20000)


# In[ ]:


# kills by a player

plt.figure(figsize = (25, 10))
plt.subplot(2,2,1)
plt.plot(killdf_train.index ,killdf_train['kills'], color = "pink", linewidth = 1, label = "Kills by players")
plt.plot(killdf_train.index, killdf_train['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL" )
plt.plot( killdf_train.index, killdf_train['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Kill Records of first 20,000 players in training", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)

plt.subplot(2,2,2)
plt.plot(killdf_test.index ,killdf_test['kills'], color = "orange", linewidth = 1, label = "Kills by players")
plt.plot(killdf_test.index, killdf_test['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL")
plt.plot(killdf_test.index, killdf_test['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Kill Records of first 20,000 players in testing", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)


# In[ ]:


mean_df_train = killdf_train['Mean'].iloc[0]
mean_df_test = killdf_test['Mean'].iloc[0]
std_dev_train = (killdf_train['UCL'].iloc[0] - mean_df_train)/3
std_dev_test = (killdf_test["UCL"].iloc[0] - mean_df_test)/3

print("Mean in train : {:.5f}, Mean in test : {:.5f}".format(mean_df_train, mean_df_test))
print("Standard Deviation in train : {:.5f}, Standard Deviation in test : {:.5f}".format(std_dev_train, std_dev_test))


# After having a look at the datasets right here, I don't think that this particular data has much difference in quality, so lets move to the next area of concern.
# 
# # Data check two : The amount of damage dealt by a player

# In[ ]:


damagedf_train = mean_dev_stuff(df_train, 'damageDealt')
damagedf_test = mean_dev_stuff(df_test, 'damageDealt')


# In[ ]:


# damage dealt by a player

plt.figure(figsize = (25, 10))
plt.subplot(2,2,1)
plt.plot(damagedf_train.index ,damagedf_train['damageDealt'], color = "pink", linewidth = 1, label = "Damage dealt by players")
plt.plot(damagedf_train.index, damagedf_train['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL" )
plt.plot( damagedf_train.index, damagedf_train['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Damage Dealings of first 20,000 players in training", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)

plt.subplot(2,2,2)
plt.plot(damagedf_test.index ,damagedf_test['damageDealt'], color = "orange", linewidth = 1, label = "Damage dealt by players")
plt.plot(damagedf_test.index, damagedf_test['UCL'], color = "red", linewidth = 2, linestyle = "--", label = "UCL")
plt.plot(damagedf_test.index, damagedf_test['Mean']  , color = "blue", linewidth = 2, linestyle = "--", label = "Mean")

plt.title("The Damage Dealings of first 20,000 players in testing", fontsize = 16)
plt.xlabel("Player Index")
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)


# In[ ]:


mean_df_train = damagedf_train['Mean'].iloc[0]
mean_df_test = damagedf_test['Mean'].iloc[0]
std_dev_train = (damagedf_train['UCL'].iloc[0] - mean_df_train)/3
std_dev_test = (damagedf_test["UCL"].iloc[0] - mean_df_test)/3

print("Mean in train : {:.5f}, Mean in test : {:.5f}".format(mean_df_train, mean_df_test))
print("Standard Deviation in train : {:.5f}, Standard Deviation in test : {:.5f}".format(std_dev_train, std_dev_test))


# This doesn't look like a huge deal to worry about either. Let's do something more interesting.

# In[ ]:


# utility for encoding


# In[ ]:


df_train[(df_train['damageDealt'] > 0) & (df_train['kills'] <= df_train['DBNOs'])][['damageDealt', 'kills', 'DBNOs']].head(20)


# In[ ]:


print("MIN : {}, MAX : {}, MEAN : {}".format(min(df_train['matchDuration']), max(df_train['matchDuration']), df_train['matchDuration'].mean()))


# In[ ]:


study_df = pd.read_csv('../input/train_V2.csv', nrows = 1000000)


# In[ ]:


study_df.info()


# In[ ]:


Top10 = study_df[study_df['matchDuration'] >= 0.9]


# In[ ]:


Top10.head()


# In[ ]:


Top10.columns[3:]


# In[ ]:


x = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals',
       'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill',
       'matchDuration', 'maxPlace', 'numGroups', 'rankPoints',
       'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints',
       'winPlacePerc']


# In[ ]:


Top10.groupby(['matchType'], axis = 0)[x].mean()


# In[ ]:





# In[ ]:




