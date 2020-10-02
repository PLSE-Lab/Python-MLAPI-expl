#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning).**
# 
# You will need to translate the concepts to work with the data in this notebook, the Iowa data. Each page in the Machine Learning course includes instructions for what code to write at that step in the course.
# 
# # Write Your Code Below

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#Read the data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

# pull data into target (y) nd predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

#Create training predictors data
train_X = train[predictor_cols]

#Create RandomForestRegressor Model
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# In[ ]:


#Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

#Treat the test data in the same way as training data.In this case, pull some columns

test_X = test[predictor_cols]

#Use the model to make predictions
predicted_prices = my_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible
print(predicted_prices)


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice' : predicted_prices})
my_submission.to_csv('submmission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 

# 

# 

# 

# 

# 
