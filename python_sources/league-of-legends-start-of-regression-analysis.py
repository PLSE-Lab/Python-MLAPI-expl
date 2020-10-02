#!/usr/bin/env python
# coding: utf-8

# # Winning Factors in high Diamond League of Legends Games
# 
# 
# **Preface:** I have 3 hours before I need to get back to my studies, so we'll see how far we can get with this.
# 
# ![](https://cdn.app.compendium.com/uploads/user/a7c086f7-9adb-4d2c-90fa-e26177af8317/c2dea8f7-8c26-44de-ae5f-5dc019485c8c/Image/60691dbc9e7de7390b93ea5284177459/data_analytics_banner.png)
# 
# ## Intro 
# * We will try to identify this by running a regression, to see if there are some factors that are exceptionally good indication of when you will win a game or not. 
# * ***Note:*** The data is for the first 10 minutes of high elo (Diamond) games, so this could change after the 10 minute mark.
# * Let's start by arranging the data and the libraries needed.
# 
# 

# In[ ]:


import pandas as pd

df = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv", index_col=0)
pd.set_option('display.max_columns', None) # this basically makes so that we can see all the column when we do data.head(), just a preference of mine.
df.head(3)


# ### Fix the type of Data
# To make our regression, we need the data to be integers and not floats. So let's see if there are any columns we need to fix.

# In[ ]:


df.dtypes


# In[ ]:


df = df.astype(int)
df.dtypes


# In[ ]:


# Let's see if it worked for one column that was a float and should be a int now. 
df["blueGoldPerMin"].head(2)

# Looks good!


# In[ ]:


df_blue = df[['blueWins','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills','blueDeaths','blueAssists','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','blueTotalGold','blueAvgLevel','blueTotalExperience','blueTotalMinionsKilled','blueTotalJungleMinionsKilled','blueGoldDiff','blueExperienceDiff','blueCSPerMin','blueGoldPerMin']]
#df_red = df[['redWardsPlaced', 'redWardsDestroyed','redFirstBlood', 'redKills', 'redDeaths', 'redAssists','redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed','redTotalGold', 'redAvgLevel', 'redTotalExperience','redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff','redExperienceDiff', 'redCSPerMin', 'redGoldPerMin']]

df_blue.head(3)


# # Linear Regression Time!
# 
# So we need to establish the y and x variables. Y should in this case be if they won or not, to see what affects the winning in the high diamond games, then we set the rest of the variables as x variables. 
# 
# As we have a binary result of win/loss, let's use a Logistical Regression model. 

# In[ ]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
np.set_printoptions(precision=6, suppress=True)


# In[ ]:


X = df_blue.iloc[:,df_blue.columns != "blueWins"]
Y = df_blue.iloc[:, 0]

#X.head()
#Y.head()


# In[ ]:


lr = LogisticRegression()  # create object for the class
lr.fit(X, Y)  # perform regression
Y_pred = lr.predict(X)  # make predictions


# In[ ]:


column_labels = np.array(X.columns.values) 
column_labels # to remind us what each label is for each coefficient 


# In[ ]:


# Let's see the results of each variable
coef = lr.coef_
coef


# In[ ]:





# ## Results
# 
# The coefficients are small enough that I would not try to make any assumptions here, I would rather get more data to see if it would make a statistical difference to show what are them most important factors in order to win a game in high diamond. 
# 
# # What else can be done?
# 
# We should test the model with a goodness of fit test and check for multicollinearity. From what I understand the .score(x,y) function from sklearn can be used as a goodness of fit test, so let's start with that; 

# In[ ]:


lr.score(X, Y)


# ## Result
# With a score of 0.73 it would indicate that the coefficients are not fully explaining the outcome of the game, but they sure are something worth taking a look at as it is close to 1.0.
# 
# Next thing would be to test for multicollinearity and see if the different variables truly are independant of each other. 

# ### Final comments
# Here is where I'm leave it for this time because I need to get back to my uni studies. Will hopefully pick this up at a later point again, but might do it on a larger dataset.
# 
# *Cheers.*
# 
# ![](https://pbs.twimg.com/media/DdtlqFdUQAIQbR5.jpg)
# 
