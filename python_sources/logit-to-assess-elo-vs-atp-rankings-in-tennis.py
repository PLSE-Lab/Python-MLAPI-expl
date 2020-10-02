#!/usr/bin/env python
# coding: utf-8

#  <span style="color:red"> **Using ATP data on 36537 tennis games, we evaluated whether players ATP rank or their ELO score was a better predictor of their ability to win a game. Both measures significantly outperform random guessing. ELO score is a slightly better predictor of victory than ATP Rank predicting accurately 2/3 of games.** </span>

# **Title - A LOGIT model to assess ELO predictive power of winning a tennis game relative to ATP Rankings
# **
# The Goal of this Study is to determine if ELO rating system is a better predictors than ATP tennis rankings of a players likelihood to win a tennis game.
# 
# *** Hypothesis: The Elo Rating System is a better predictor than the ATP player ranks to predict a players likelyhood to win a tennis game ***
# 
# While ELO rankings have widely been used to rank chess players they are not commonly used to rank Tennis players. We believe that ELO's ranking system is better than the ATP ranking system for two reasons: 1) a players ELO scores carries information about the strength of all previous players that he has defeated and 2) the distribution of ELO allows us to derive a probability of victory between any two players. Intuitively, two players with near identical ATP ranks may have different ELOs if they achieved those ranks by facing off players with different skill levels. Under an ELO ranking system, defeating Rafael Nadal twice can yield a higher ELO than defeating 50 anonymous players in a row. Secondly - the ELO ranking methodology theoretically allows for intergenerational comparison of players that the ATP ranking system does not.
# 
# ** Methodology **
# To assess our hypothesis we are going to use players ATP ranking and their ELO scores to estimate their probability of winning tennis games using a LOGIT model.

# In[ ]:


# Importing the various modules I will need for this analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
# input data files are available in the input directory


# ## Getting our Data

# In[ ]:


# reading our data
data = pd.read_csv('../input/Tennis_ATP_ELO_Rank_Augmented.csv')


# In[ ]:


data.columns


# ## Exploring Data

# In[ ]:


# We want to include a dummy for the Series in our analysis
# b/c different Series attract players of different calibers
data['Series'].unique()


# In[ ]:


fig, ax =plt.subplots(1,1,figsize=(12,10))
plt.subplots_adjust(hspace = 0.2, top = 0.4)

g0 = sns.countplot(x="Series", data=data)
g0.set_xticklabels(g0.get_xticklabels(),rotation=45)


# In[ ]:


data['Court'].unique()


# In[ ]:


data['Court'].value_counts()


# In[ ]:


# Some surfaces are harder than other
data['Surface'].unique()


# In[ ]:


fig, ax =plt.subplots(1,1,figsize=(12,10))
plt.subplots_adjust(hspace = 0.2, top = 0.4)

g0 = sns.countplot(x="Surface", data=data)
g0.set_xticklabels(g0.get_xticklabels(),rotation=45)


# In[ ]:


data['Best of'].value_counts()


# ## Creating Dummy Variables to Run our Analysis
# 
# To test our hypothesis we need to create categorical variables to distinguish between between Surfaces {Carpet;  Clay; Grass; and Hard}, Series {ATP250; ATP500; Grand Slam; International; International Gold; International Series; Master; and Master 1000}, Court {Outdoors or Indoors}, and Best of {3 or 5}.

# In[ ]:


data_dv = pd.get_dummies(data,columns=['Series','Court','Surface','Best of'])


# In[ ]:


# I will drop INTERNATIONAL, OUTDOOR, HARD, "Best of" 3 from my dummies set
# I choose those dummy variables because they are my base categories
# I choose the base categories by choosing the categories with the most data
# data_dv.drop(['Series_International','Court_Outdoor',
#                             'Surface_Hard','Best of_3'], axis=1, inplace=True)
data_dv.columns


# In[ ]:


# creating VARS with only relevant quantitative variables both dependent and independent
vars = data_dv.iloc[:,[0,29,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]]


# In[ ]:


vars.columns


# ### Creating the set of dependent variables and explanatory variables
# WON is a binary variable indicating whether or not the player has won his game.
# Since we are comparing ELO ranks explanatory power relative to ATP rank difference, I will create two set of explanatory variables [x_elo and x_ATP]. I will also create two dependent variable y_won for a LOGIT regression and y_spm (set points margin).

# In[ ]:


x_elo=vars.iloc[:,[1,4,5,6,7,8,9,10,11,12,13,14,15,16]]
x_atp=vars.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
y_won = vars.iloc[:,0]
x_spm = vars.iloc[:,[2]]


# ### Removing #Value! from Opponent Rankings Diff

# In[ ]:


x_atp1 = x_atp.replace(('#VALUE!'),np.nan)
x_atp1 = pd.concat([y_won,x_atp1],axis=1)
x_atp1 = x_atp1.dropna()


# In[ ]:


x_atp1.head()


# In[ ]:


x_atp1.tail()


# In[ ]:


y_won1 = x_atp1.iloc[:,0] # Creating a vector of Won games with the unranked players removed


# In[ ]:


y_won1.astype(float).head()


# In[ ]:


x_atp1 = x_atp1.iloc[:,1:]


# ## Running our Logit Model
# 

# In[ ]:


model = LogisticRegression()


# ### ATP difference on explaining likelyhood of winning
# Using SciKitLearn to evaluate the the difference in ATP scores ability to predict the winner of the game

# In[ ]:


model = model.fit(x_atp1,y_won1)


# In[ ]:


model.score(x_atp1, y_won1)


# In[ ]:


model_elo = model.fit(x_elo,y_won)


# In[ ]:


model_elo.score(x_elo,y_won)


# In[ ]:


print(model_elo.intercept_, model_elo.coef_)


# # Summary
# 
# From this analysis we can see that ELO is a better predictor of the probability to win than the difference of players ranks.
# However ** the difference is quite small 0.66 v 0.65 ** and *both variables beat a coin flip - 0.5* 
# 
# Given the small difference in model performance, I ran spearman rank correlation test (see below) to check if the variables are highly correlated. They are positively correlated but only moderately so.
# 
# ## Possible extension - Using more variables about players performance
# 
# * Different players specialize in different court styles - we could create a more sophisticated model that would include a variable that reflects a players performance or handicap on a given court style.
# 
# * We could also include an indicator for fatigue. Players who have won easily previous games (by playing fewer sets) have an advantage, ceteris paribus.
# 
# * 80 / 20 - more complexity will be more time incentive without a significant improvement in performance.

# In[ ]:


y_vars = vars.iloc[:,[0,1,3]]


# In[ ]:


print(y_vars.corr(method='spearman',))

