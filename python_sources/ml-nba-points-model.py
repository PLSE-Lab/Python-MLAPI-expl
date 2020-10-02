#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to use data from the 2013-2018 NBA seasons to train a model to predict the # of points for a given game. 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
NBA = pd.read_csv("../input/seasons-1318/NBA.csv")


# In[ ]:


y = NBA.teamPTS

NBA_features = ['teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamFGM', 'pace', 'poss', 'team2P%', 'team3P%', 'teamFTM', 'teamORB' ]
X = NBA[NBA_features]
X.describe()


# A better model might include some more advanced stats (like assist ratio, EFG, ect.)

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# We first train the model with a Decision Tree; and then use the Random Forest model to search for better results. 

# In[ ]:


NBA_model = DecisionTreeRegressor(random_state=1)
NBA_model.fit(train_X, train_y)
NBA_pred = NBA_model.predict(val_X)


# In[ ]:


print("The actual results")
print(y.head(20))
print("The predictions are")
print(NBA_pred[:20])


# The error ended up being alot smaller than I expected (off by 3 points on average). However, looking at the data game to game we see that the model is off by more than 20 points at times. So we want to implement a better model. 

# In[ ]:


mean_absolute_error(val_y, NBA_pred)


# Next, we implement Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


NBA_model_2 = RandomForestRegressor(random_state=1)
NBA_model_2.fit(train_X, train_y)
NBA_pred_2 = NBA_model_2.predict(val_X)
print("The actual results")
print(y.head(20))
print("The predictions are")
print(NBA_pred_2[:20])


# In[ ]:


print(mean_absolute_error(val_y, NBA_pred_2))


# The new model is close to 35% better based off the MSE calculation! After learning more ML I'll return to this model to see if I can further improve it. 
