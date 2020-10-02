#!/usr/bin/env python
# coding: utf-8

# # EDA---Exploratory Data Analysis #
# Exploration of the features that can impact player's Wins RPM.

# ## 1. Import Packages and Dataset ##

# In[ ]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d
import numpy as np


# In[ ]:


#import twitter dataset
twitter= pd.read_csv("../input/nba_2017_twitter_players.csv");
twitter.head()


# In[ ]:


#Import plear behavior dataset
plus = pd.read_csv("../input/nba_2017_real_plus_minus.csv");
plus.head()


# In[ ]:


#Import salary dataset
salary= pd.read_csv("../input/nba_2017_salary.csv");
salary.head()


# ## 2. Data Cleaning ##

# In[ ]:


#Manage salary dataset
salary = salary.rename(columns={'NAME' : 'PLAYER'})
salary["SALARY_MILLIONS"] = round(salary["SALARY"]/1000000, 2)
salary.drop(["SALARY","TEAM"], inplace=True, axis=1)
salary.head()


# In[ ]:


#manage plus dataset
plus.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus.drop(["PLAYER"], inplace=True, axis=1)
plus["PLAYER"] = players
plus.head()


# In[ ]:


#Merger salary,plus and twitter dataset
total=twitter.merge(salary)
total=total.merge(plus,how="inner", on="PLAYER"); 
total.head(5)


# In[ ]:


total.info()


# ## 3. Data Visualization ##

# In[ ]:


#1.plot the salary and wins_rpm to see if the higher salary, the higher wins_rpm
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=total)


# In[ ]:


#2.heatmap
plt.subplots(figsize=(7,5))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:STATS & SALARY")
corr = total.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,cmap="BuPu")


# In[ ]:


#3 barplot
plt.figure(figsize = (10,4))
sns.barplot(x='POSITION', y='SALARY_MILLIONS', data=salary).set_title("Position Vs. Saraly")
plt.show()


# In[ ]:


#4 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
# Generate the values
x_vals = total['TWITTER_FAVORITE_COUNT']
y_vals = total['TWITTER_RETWEET_COUNT']
z_vals = total['SALARY_MILLIONS']
# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c = 'r', marker='o')
ax.set_xlabel('TWITTER_FAVORITE_COUNT')
ax.set_ylabel('TWITTER_RETWEET_COUNT')
ax.set_zlabel('SALARY_MILLIONS')
plt.show()


# In[ ]:




