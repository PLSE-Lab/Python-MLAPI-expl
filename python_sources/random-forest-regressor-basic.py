#!/usr/bin/env python
# coding: utf-8

# # Random Forest Regressor
# 
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
print(os.listdir("../input"))


# ### read the data

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# ### pull data into target (y) and predictors (X)

# In[ ]:


train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_cols]


# ### random forest regressor model

# In[ ]:


model = RandomForestRegressor()
model.fit(train_X, train_y)


# ### read the test data

# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


test_X = test[predictor_cols]

predicted_prices = model.predict(test_X)
print(predicted_prices)


# ### prepare submission file

# In[ ]:


submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
submission.to_csv('submission.csv', index=False)


# In[ ]:




