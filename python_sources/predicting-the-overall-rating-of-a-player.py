#!/usr/bin/env python
# coding: utf-8

# ## Data Import

# In[2]:


import seaborn as sns
import sqlite3
import pandas as pd
conn = sqlite3.connect('../input/database.sqlite')

player_data = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
player_data.head()


# ## Data Manipulation

# In[3]:


player_data.columns


# In[4]:


req_cols = ['overall_rating', 'crossing', 'finishing', 'heading_accuracy','short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy','long_passing', 'ball_control', 'acceleration', 'sprint_speed','agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina','strength', 'long_shots', 'aggression', 'interceptions', 'positioning','vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle','gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning','gk_reflexes']
data = player_data[req_cols]


# In[5]:


data.describe()


# In[70]:


player_data.columns


# ## Feature Selection

# In[71]:


data = player_data.drop(labels = ['id', 'player_fifa_api_id', 'player_api_id', 'date', 'potential', 'preferred_foot', 'attacking_work_rate', 'defensive_work_rate'], axis = 1)
data.fillna(0, inplace=True)
#data.isnull().values.any()
data.corr()


# In[72]:


import statsmodels.formula.api as smf
lm = smf.ols(formula = 'overall_rating ~ crossing + finishing + heading_accuracy + short_passing + volleys + dribbling + curve + free_kick_accuracy + long_passing + ball_control + acceleration + sprint_speed + agility + reactions + balance + shot_power + jumping + stamina + strength + long_shots + aggression + interceptions + positioning + vision + penalties + marking + standing_tackle + sliding_tackle + gk_diving + gk_handling + gk_kicking + gk_positioning + gk_reflexes', data = data).fit()
lm.summary()


# Because of insignificant p-value, let's not consider 'volleys' as a feature.

# And because of low correlation coefficients, we can neglect goal-keeping skills and check rsquared value again

# In[73]:


lm2 = smf.ols(formula = 'overall_rating ~ crossing + finishing + heading_accuracy + short_passing + volleys + dribbling + curve + free_kick_accuracy + long_passing + ball_control + acceleration + sprint_speed + agility + reactions + balance + shot_power + jumping + stamina + strength + long_shots + aggression + interceptions + positioning + vision + penalties + marking + standing_tackle + sliding_tackle', data = data).fit()
lm2.summary()


# There is a significant change in rsquared value so we shall consider the first linear model itself.

# ## Model Creation

# In[74]:


from sklearn.cross_validation import train_test_split
feature_cols = ['crossing', 'finishing', 'heading_accuracy', 'short_passing',
       'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']

x = data[feature_cols]
y = data.overall_rating

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)


# In[75]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)


# ## Prediction And Validation

# In[76]:


predicted_overall_rating = regressor.predict(x_test)


# In[77]:


from sklearn.metrics import mean_squared_error
import numpy as np
msr = mean_squared_error(y_test, predicted_overall_rating)
rmsr = np.sqrt(msr)
print('Mean Squared Error = ', msr)
print('Root Mean Squared Error = ', rmsr)

