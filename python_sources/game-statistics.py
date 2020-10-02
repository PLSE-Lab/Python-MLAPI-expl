#!/usr/bin/env python
# coding: utf-8

# # Statistics on a Blockbreaker-like Game

# The author is in the process of creating a blockbreaker-like game, in which the jumping-off point is the "Block Breaker" section of the Udemy course, [Complete C# Unity Developer 2D: Learn to Code Making Games](https://www.udemy.com/share/1000PUA0EacVlURH4=/)
# 
# After making lots of levels, the author needed to sort them by difficulty.  How does one measure the difficulty of a level?  A first-cut solution is 
# to make an auto-play bot that is not perfect, and see how well the bot does on each level, using thousands of trials.
# 
# [Here is a video](https://youtu.be/AVHsnsCWcU4) of the game in auto-play action.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-deep")
get_ipython().run_line_magic('matplotlib', 'inline')


# The unity game outputs a log file (GameStats.csv), which looks something like this:

# In[ ]:


#"Date","Level","NumBlocks","IsWin","ElapsedTime","Score","Accuracy"
#"9/9/2019 3:47:23 PM","Level_468","56","True","28.914","4900","0.3725966"
#"9/9/2019 3:47:53 PM","Level_469","56","True","29.72704","2800","0.3271338"
#"9/9/2019 3:48:13 PM","Level_471","39","True","20.75284","2100","0.3765845"
#"9/9/2019 3:48:37 PM","Level_494","50","True","23.229","4000","0.3834174"
#"9/9/2019 3:49:05 PM","Level_495","40","True","27.80006","2000","0.350372"
#"9/9/2019 3:49:13 PM","Level_530","43","False","7.810999","1350","0.3234939"
#"9/9/2019 3:49:30 PM","Level_262","36","True","16.548","1800","0.3528175"


# The fields are:
# 
# * Date: date and time the game was auto-played
# * Level: the name of the level (the 3-digit number is an estimate of the difficulty from a previous run, no longer valid after tweaking)
# * NumBlocks: how many blocks have to be broken to win the level
# * IsWin: True if autoplay broke all the blocks, False if the ball fell past the paddle.
# * ElapsedTime: Seconds until either won or lost (game is played at 4x speed, so multiply by 4 to get an estimate of how long a human might play it)
# * Score: total score when the game was won or lost
# * Accuracy: the autoplay is tuned with a randomly-chosen accuracy.  Higher numbers are more likely to win.
# 

# In[ ]:


# read the data, use Level and Date as multilevel index, and parse the data.  Sort by level, then date.
df = pd.read_csv("../input/GameStats.csv", index_col=['Level', 'Date'], parse_dates=['Date']).sort_index()
# Compute difficulty by level and append to dataframe: 1000 * fraction of wins
g = df.groupby(level="Level")['IsWin']
s=np.round((1-g.sum()/g.count())*1000)
#also append 3 sigma value: 99.7% confidence interval based on number of samples
s = pd.DataFrame({"Difficulty":s.apply(lambda x:np.int(x)), "3 sigma":3*np.sqrt((1-g.sum()/g.count())*(g.sum()/g.count())*g.count())})
df=df.merge(s, left_index=True, right_index=True)
#Append number of samples per level
s = np.round(g.count())
s = pd.DataFrame({"Count":s.apply(lambda x:np.int(x))})
df=df.merge(s, left_index=True, right_index=True)
df.info()


# In[ ]:


df.head()


# In[ ]:


#compute summary statistics per level
#start with mean difficulty of each level
means = df.groupby(level="Level").mean().sort_values('Difficulty')
#try to determine the value of "Accuracy" such that when playing the level
# at this accuracy, it has a 50% chance of winning this level.
means['ThresholdAccuracy'] = 0
for level in means.index:
    l = df.loc[level]
    r = np.linspace(l.Accuracy.min(), l.Accuracy.max(),100)
    for x in r:
        left = l[l['Accuracy'] < x]    
        right = l[l['Accuracy'] > x]    
        if left['IsWin'].sum() >= right['IsWin'].sum():
            means.loc[level,'ThresholdAccuracy']  = x
            break
means


# In[ ]:


#check that the Difficulty depends more on the actual level difficulty than on the accuracy of the autoplay
from sklearn import linear_model
model = linear_model.LinearRegression()
X=means[['Accuracy']]
y=means['Difficulty']
model.fit(X,y)
print(model.coef_, model.intercept_, means['Accuracy'].corr(means['Difficulty']))
y_pred = model.predict(X)
sns.regplot(means['Accuracy'], means['Difficulty'])


# In[ ]:


#Level difficulties with 3-sigma error intervals
#As you can see, statistically, many levels cannot be distinguished in difficulty without far more trials
plt.figure(figsize=(20,6))
plt.bar(means.index, means['Difficulty'], yerr=means['3 sigma'], capsize=10)


# In[ ]:


# how many trials by level: between about 300 and 500
plt.figure(figsize=(20,6))
plt.bar(means.index, means['Count'])


# In[ ]:


# expect ThresholdAccuracy and Difficulty to correlate very well, so the former can be an alternative to the latter
sns.regplot(means['ThresholdAccuracy'], means['Difficulty'])
means['ThresholdAccuracy'].corr(means['Difficulty'])


# In[ ]:


#the author used this to determine which range of accuracies should be chosen on the next test.  Between .35 and .38 seems to cover all the difficulty levels well.
means['ThresholdAccuracy'].min(),means['ThresholdAccuracy'].max()


# In[ ]:




