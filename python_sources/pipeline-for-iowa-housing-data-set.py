#!/usr/bin/env python
# coding: utf-8

# # Using Pipelines to create a more efficient Model
# 
# "Pipelines are a simple way to keep your data processing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step." (DanB, Pipelines)

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
#so pandas doesn't spit out a warning everytime

# DATA PREPROCESSING
# Loading in Iowa housing data
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True) #drops data with missing SalePrice value
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
# this is the path to the Iowa data we will use
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

print('Setup Complete...')


# Now we will make our pipeline by using an Imputer to fill in our missing values and a RandomForestRegressor to make our predictions.

# In[ ]:


from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), XGBRegressor())


# Now we will fit our model and make our predictions

# In[ ]:


my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
predictions = test.select_dtypes(exclude=['object'])

predicted_prices = my_pipeline.predict(predictions)


# # Making our submission

# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
print('Submitted!')


# 
