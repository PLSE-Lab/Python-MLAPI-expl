#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to preform simple linear regression in two ways: 1. non paramteric regression with Least squares method (model) and simple parametric linear regression (model_2). The model looks to answer the question of how the total points of an NBA game depends on the total number of steals and assists (assists - turnovers). Real NBA box scores data was pulled from Kaggle (https://www.kaggle.com/pablote/nba-enhanced-stats). 

# In[ ]:


import pandas as pd


file_path = '../input/2012-18_teamBoxScore.csv'
X2012_18_teamBoxScore = pd.read_csv(file_path) 
X2012_18_teamBoxScore.describe()


# In[ ]:


y = X2012_18_teamBoxScore.teamPTS


# In[ ]:


nba_features = ['teamSTL', 'teamAST', 'teamTO']
x = X2012_18_teamBoxScore[nba_features]
x_2 = X2012_18_teamBoxScore["teamSTL"] + X2012_18_teamBoxScore["teamAST"] - X2012_18_teamBoxScore["teamTO"]


# In[ ]:


x.describe()


# In[ ]:


import statsmodels.api as sm
import numpy as np
from sklearn import linear_model


model = sm.OLS(y, x).fit()
predictions = model.predict(x)


# In[ ]:


lm = linear_model.LinearRegression()
model_2 = lm.fit(x,y)
predictions_2 = model_2.predict(x)


# In[ ]:


print("The predictions are")
print(predictions_2)


# In[ ]:


print("The predictions are")
print(predictions)
print(model.summary())


# In[ ]:


import matplotlib.pyplot as plt 

plt.scatter(x_2,y)


# In[ ]:


import matplotlib.pyplot as plt 

plt.scatter(x_2,y)
plt.plot(x_2, predictions,color='green')
plt.plot(x_2,predictions_2, color='red')

plt.show() 


# Next I wanted to implement a better model by implementing the numbers of shots in a game (shots attempted (FGA) - shots made (FGM)).

# In[ ]:


nba_features = ['teamSTL', 'teamAST', 'teamTO', 'teamFGA', 'teamFGM']
z = X2012_18_teamBoxScore[nba_features]
z_2 = X2012_18_teamBoxScore["teamSTL"] + X2012_18_teamBoxScore["teamAST"] - X2012_18_teamBoxScore["teamTO"] + X2012_18_teamBoxScore["teamFGA"] - X2012_18_teamBoxScore["teamFGM"]


model_3 = sm.OLS(y, z).fit()
predictions_3 = model_3.predict(z)

print(model_3.summary())

plt.scatter(z_2,y)
plt.plot(z_2,predictions_3, color='yellow')

plt.show() 

