#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sqlite3


# In[ ]:


import pandas as pd


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


from math import sqrt


# In[ ]:


cnx = sqlite3.connect('../input/soccer/database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


features = [
       'potential', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']


# In[ ]:


target = ['overall_rating']


# In[ ]:


df = df.dropna()


# In[ ]:


X = df[features]


# In[ ]:


y = df[target]


# In[ ]:


X.head()


# In[ ]:


X.iloc[2]


# In[ ]:


y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=324)


# In[ ]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:


y_prediction = regressor.predict(X_test)


# In[ ]:


y_prediction


# In[ ]:


y_test.describe()


# In[ ]:


RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))


# In[ ]:


print(RMSE)
RMSE


# In[ ]:


regressor = DecisionTreeRegressor(max_depth=20)
regressor.fit(X_train,y_train)


# In[ ]:


y_prediction = regressor.predict(X_test)
y_prediction


# In[ ]:


y_test.describe()


# In[ ]:


RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))


# In[ ]:


print(RMSE)

