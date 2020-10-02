#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor


# In[ ]:



# Read the training data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()


# In[ ]:


# Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.head()


# In[ ]:


# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']


# In[ ]:


# Create training predictors data
train_X = train[predictor_cols]

# Create test predictors data
test_X = test[predictor_cols]


# In[ ]:


model = XGBClassifier()
model.fit(train_X, train_y)
print(model)


# In[ ]:


pred=model.predict(test_X)
print(pred)


# In[ ]:


submission = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})

submission.to_csv('submission.csv', index=False)

