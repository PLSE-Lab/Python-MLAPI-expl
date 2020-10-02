#!/usr/bin/env python
# coding: utf-8

# # Model for House Prices using Regression Techniques
# **Basic Machine Learning  model for predicting House SalePrice**
# 
# 
# 

# In[1]:


import pandas as pd

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
# print('Some output from running this cell')
print(data.describe())


# In[2]:


print(data.columns)


# In[3]:


data_col = data.SalePrice
print(data_col.head())


# In[7]:


new_col = ['OverallCond','KitchenAbvGr']
new_data = data[new_col]
#print(new_data.describe())
print(new_data.dtypes)


# In[8]:


y = data_col
predictors = ['LotArea','OverallQual','OverallCond','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','YearBuilt']
X = data[predictors]


# In[9]:


from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split as tts

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# print(test.columns)
predictors_test = ['LotArea','OverallQual','OverallCond','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','YearBuilt']
test_X = test[predictors_test]
# enc_test_X = pd.get_dummies(test_X)
# print(test_X.dtypes)

train_X, val_X, train_y, val_y = tts(X,y, random_state=0)
forest_model = RFR()
forest_model.fit(train_X,train_y)
# train_preds = forest_model.predict(val_X)
# print(train_preds)
out = forest_model.predict(test_X)
print(out)


# In[10]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': out})
my_submission.to_csv('submission.csv',index=False)

