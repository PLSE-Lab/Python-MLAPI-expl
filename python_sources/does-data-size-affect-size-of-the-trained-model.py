#!/usr/bin/env python
# coding: utf-8

# Do the numbers of columns or rows in the training data affect the size of the trained model in sklearn? (Here we're looking at just logistic regression.)

# In[20]:


# libraries we'll us
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import sys
import datetime

# read in our data
data = pd.read_csv("../input/optical_interconnection_network.csv")

# tidy up column names
data.columns = data.columns.str.strip()

# replace , with .
data = data.stack().str.replace(',','.').unstack()

# drop columns with missing values
data = data.dropna(axis=1)


# In[21]:


# check our data
data.head()


# In[22]:


# column we'll predict & use to predict
all_predictors = ["T/R", "Processor Utilization", "Channel Waiting Time", "Input Waiting Time", "Network Response Time", "Channel Utilization"]
to_predict = "Spatial Distribution"


# In[32]:


# basic model
basic_model = linear_model.LogisticRegression()
basic_model.fit(data[all_predictors], data[to_predict])

# print info on sizes
print(f"Model size (in bytes): {sys.getsizeof(basic_model)}")
print(f"Data size (in bytes): {sys.getsizeof(data[all_predictors])}")


# In[33]:


# smaller data (fewer rows)
small_data = data[:10].append(data[100:110])

# basic model
less_data_model = linear_model.LogisticRegression()
less_data_model.fit(small_data[all_predictors], small_data[to_predict])


# print info on sizes
print(f"Model size (in bytes): {sys.getsizeof(less_data_model)}")
print(f"Data size (in bytes): {sys.getsizeof(small_data[all_predictors])}")


# In[35]:


# fewer predictors (fewer columns)
fewer_predictors = ["T/R", "Processor Utilization", "Channel Waiting Time"]

# basic modelfewer_predictors
fewer_predictors_model = linear_model.LogisticRegression()
fewer_predictors_model.fit(data[fewer_predictors], data[to_predict])

# print info on sizes
print(f"Model size (in bytes): {sys.getsizeof(fewer_predictors_model)}")
print(f"Data size (in bytes): {sys.getsizeof(data[fewer_predictors])}")


# So it appears that, at least for a smaller dataset, the number of rows and columns does not affect the size of the trained model. (My intution is that if you have a very large number of columns you're using for your predictions, you'll get a larger model, however)
