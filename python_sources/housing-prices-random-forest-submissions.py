#!/usr/bin/env python
# coding: utf-8

# This is a submissions fork!

# In[ ]:


#import packages
import pandas as pd
import numpy as np

print('All Good!')


# In[ ]:


#Read training data
train_data = pd.read_csv('../input/train.csv')

#Prepare Training data
independent_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train_data[independent_cols]
train_y = train_data.SalePrice

print('All Good!')


# In[ ]:


#Build Random Forest model on raw data
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()

RF_model.fit(train_X, train_y)


# In[ ]:


#Read testing data
test_data = pd.read_csv('../input/test.csv')

#Predict using RF_model
test_X = test_data[independent_cols]

pred_y = RF_model.predict(test_X)

print(pred_y)


# In[ ]:


#Store in the predictions and make submission
RF_submissions1 = pd.DataFrame({'Id':test_data.Id, 'SalePrice':pred_y})

RF_submissions1.to_csv('RF_submissions1.csv', index=0)

print('All Good!')

