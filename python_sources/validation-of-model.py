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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left

selected_cols = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']                
X = data[selected_cols]
y = data.SalePrice

lowa_model = DecisionTreeRegressor()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

print ("Describe train_X")
print (train_X.describe())

print ("Describe train_y")
print (train_y.describe())

lowa_model.fit(train_X, train_y)
predicted_prices = lowa_model.predict(val_X)
mean_absolute_error(val_y, predicted_prices)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
