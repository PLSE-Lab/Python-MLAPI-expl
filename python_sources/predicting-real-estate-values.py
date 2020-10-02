#!/usr/bin/env python
# coding: utf-8

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


# Decision Tree: 
# We'll start with a model called the Decision Tree. There are fancier models that give more accurate predictions. But decision trees are easy to understand, and they are the basic building block for some of the best models in data science.
# 
# As an example, we'll look at data about home prices in Melbourne, Australia. In the hands-on exercises, you will apply the same processes to a new dataset, which has home prices in Iowa.
# 
# The example (Melbourne) data is at the file path ../input/melbourne-housing-snapshot/melb_data.csv.
# 
# We load and explore the data with the following commands:

# # **Load the Data**

# In[ ]:


# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()


# **Examine the Columns**

# In[ ]:


melbourne_data.columns


# In[ ]:


# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# So we will take the simplest option for now, and drop houses from our data.
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)


# # Selecting The Prediction Target

# In[ ]:


y=melbourne_data.Price


# # Choosing Features
# The columns that are inputted into our model (and later used to make predictions) are called "features." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features.
# 
# For now, we'll build a model with only a few features. Later on you'll see how to iterate and compare models built with different features.
# 
# We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).

# In[ ]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
#print(melbourne_features)


# By convention, this data is called X.

# In[ ]:


X = melbourne_data[melbourne_features]
#print(X)

X.describe()


# In[ ]:


X.head()


# # Building the Model
# I am going to use the scikit-learn library to create your models. When coding, this library is written as sklearn, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.
# 
# The steps to building and using a model are:
# 
# 1. **Define**: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# 2. **Fit**: Capture patterns from provided data. This is the heart of modeling.
# 3. **Predict**: Just what it sounds like
# 4. **Evaluate**: Determine how accurate the model's predictions are.

# In[ ]:


#import the DecisionTreeregressor from the package sklearn.tree
from sklearn.tree import DecisionTreeRegressor

#specify the model
#For model reproducibility, set a numeric value for random_state when specifying the model
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit the model
melbourne_model.fit(X,y)


# # Make Predictions
# Make predictions with the model's predict command using X as the data. Save the results to a variable called predictions.

# In[ ]:


predictions = melbourne_model.predict(X)

#checking the predictions with the Price that I stored in a variable y
print(predictions)
print(y)

