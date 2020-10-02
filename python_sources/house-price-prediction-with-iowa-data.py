#!/usr/bin/env python
# coding: utf-8

# # Intro
# **This is your workspace for Kaggle's Machine Learning course**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.
# 
# The tutorials you read use data from Melbourne. The Melbourne data is not available in this workspace.  Instead, you will translate the concepts to work with the data in this notebook, the Iowa data.
# 
# # Write Your Code Below
# 

# In[3]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
y = data.SalePrice
predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[predictors]
model = RandomForestRegressor()
model.fit(X,y)
test = pd.read_csv('../input/test.csv')
test_X = test[predictors]
prices = model.predict(test_X)
print(prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
