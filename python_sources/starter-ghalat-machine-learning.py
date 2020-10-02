#!/usr/bin/env python
# coding: utf-8

# # GML - Ghalat Machine Learning! Brain+Machine Adding AI Revolution
# 
# GML is an automatic machine learning and feature engineering library in python built on top of Multiple Machine Learning packages. with this library,you can find and fill the missing values in your data, encode them, generate new features from them, select the best features and train your data on multiple machine learning algorithms and a neural network! not only training but scaling the data for normal distribution and after scaling and training, testing the data on validation data. in AUTO Machine Learning, there would be two rounds, in first round all the models will compete for top 5 and after that in second round those top 5 will compete for number one spot. the first ranked model will be returned (untrained, so you can train it yourself and check results).
# 
# Source: https://github.com/Muhammad4hmed/Ghalat-Machine-Learning

# In[ ]:


get_ipython().system('pip install GML')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from GML.Ghalat_Machine_Learning import Ghalat_Machine_Learning


# In[ ]:


# Read the data
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train_X = train[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']]
train_Y = train.SalePrice

# Read the test data
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']]

# # Use the model to make predictions
# predicted_prices = my_model.predict(test_X)
# # We will look at the predicted prices to ensure we have something sensible.
# print(predicted_prices)


# In[ ]:


gml = Ghalat_Machine_Learning()


# In[ ]:



new_X,y = gml.Auto_Feature_Engineering(train_X,train_Y,type_of_task='Regression',test_data=None,
                                                          splits=6,fill_na_='median',ratio_drop=0.2,
                                                          generate_features=True,feateng_steps=2)


# here is our new X

# In[ ]:


new_X


# # Lets import our own model to make it compete with rest

# In[ ]:


from sklearn.neural_network import MLPRegressor


# In[ ]:


best_model = gml.GMLRegressor(new_X,y,neural_net='Yes',epochs=100,models=[MLPRegressor()],verbose=False)


# #  Working in progress

# In[ ]:





# # Final
