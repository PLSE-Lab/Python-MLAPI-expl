#!/usr/bin/env python
# coding: utf-8

# Below are the accuracy comparission for Linear Regression, Decission Tree Regression and Decission Tree Classifier.

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3


# In[ ]:


conn=sqlite3.connect('../input/database.sqlite')
data=pd.read_sql_query('SELECT * FROM Player_Attributes',conn)
data.head()


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


data=data.dropna()


# In[ ]:


target ='overall_rating'


# In[ ]:


data[target].describe()


# In[ ]:


X=data[features].copy()


# In[ ]:


Y=data[target].copy()


# In[ ]:


X.sample()


# In[ ]:


Y.head()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.33, random_state=99)


# In[ ]:


reg = LinearRegression()
reg.fit(X_train,Y_train)


# In[ ]:


Y_pred = reg.predict(X_test)
Y_pred


# In[ ]:


Lin_Reg_RMSE = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=Y_pred))


# Mean Error for Linear Regression:

# In[ ]:


Lin_Reg_RMSE


# *************

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


reg=DecisionTreeRegressor(max_depth=20)
reg.fit(X_train,Y_train)


# In[ ]:


Y_pred=reg.predict(X_test)
Y_pred


# In[ ]:


Dtree_Reg_RMSE = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=Y_pred))


# Mean Error for Decission Tree Regression:

# In[ ]:


Dtree_Reg_RMSE


# *****************************

# In[ ]:


target_c='overall_level'
data[target_c]=3
data.loc[data[target]>=85,target_c]=1
data.loc[(data[target]<85) & (data[target]>=70),target_c]=2


# In[ ]:


data[[target,target_c]]


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


Y=data[target_c].copy()


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=99)


# In[ ]:


X_train.head()


# In[ ]:


clasif=DecisionTreeClassifier(max_leaf_nodes=15,random_state=0)
clasif.fit(X_train,Y_train)


# In[ ]:


pred=clasif.predict(X_test)


# Accuracy Score for Deccision Tree Classifier:

# In[ ]:


accuracy_score(y_true=Y_test,y_pred=pred)


# ******************************

# In[ ]:




