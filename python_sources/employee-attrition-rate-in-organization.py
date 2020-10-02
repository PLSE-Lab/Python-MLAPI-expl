#!/usr/bin/env python
# coding: utf-8

# ### Competition Link:
# [click here](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-employee-attrition-rate/machine-learning/predict-the-employee-attrition-rate-in-organizations-1d700a97/) 
# 
# ### Problem statement
# Employees are the most important part of an organization. Successful employees meet deadlines, make sales, and build the brand through positive customer interactions.
# 
# Employee attrition is a major cost to an organization and predicting such attritions is the most important requirement of the Human Resources department in many organizations. In this problem, your task is to predict the attrition rate of employees of an organization. 
# 
# ### Data
#     * Train.csv
#     * Test.csv
# 
# ### Submission format
# You are required to write your predictions in a .csv file that contain the following columns:
#     * Employee_ID
#     * Attrition_rate

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the dataset as trainning data and submission data

# In[ ]:



data = pd.read_csv('../input/hackerearth-employee-attrition-rate/Train.csv')
submission_data = pd.read_csv('../input/hackerearth-employee-attrition-rate/Test.csv')


# In[ ]:


data.shape


# In[ ]:


submission_data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# ## List the important features through xgboost features importance 
# ### Tutorial
# 1. [DataCamp](https://www.datacamp.com/community/tutorials/xgboost-in-python)
# 2. [machinelearningmastery](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/#:~:text=XGBoost%20is%20an%20implementation%20of,first%20XGBoost%20model%20in%20Python.)
# 
# I have done this on seperate notebook

# In[ ]:


features = ['Age', 'Compensation_and_Benefits', 'Work_Life_balance', 'Post_Level', 'growth_rate', 'Time_of_service', 'Pay_Scale', 'Hometown', 'Education_Level']


# In[ ]:


data[features].isna().sum()


# In[ ]:


# Convert the categorical data into numaric value

for feature in features:
    if data[feature].dtype == 'object':
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes


# ## Observation after many iterations 
#     * mean value ->(null value replace with mean will get 81.208% score)
#     * 75% value ->(null value replace with mean will get 81.283% score)
#     * Highest score is 81.668%

# In[ ]:


data['Age'].fillna(52, inplace=True)
data['Work_Life_balance'].fillna(3, inplace=True)
data['Time_of_service'].fillna(21, inplace=True) 
data['Pay_Scale'].fillna(8, inplace=True)


# In[ ]:


data[features].isna().sum()


# In[ ]:


submission_data['Age'].fillna(52, inplace=True)
submission_data['Work_Life_balance'].fillna(3, inplace=True)
submission_data['Time_of_service'].fillna(21, inplace=True) 
submission_data['Pay_Scale'].fillna(8, inplace=True)


# In[ ]:


submission_data[features].isna().sum()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[ ]:


X, y = data[features].values, data['Attrition_rate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

def prepare_inputs(X_train, X_test):
    ohe = OrdinalEncoder()
    ohe.fit(X_train)
    X_train_enc = ohe.transform(X_train)
    X_test_enc = ohe.transform(X_test)
    return X_train_enc, X_test_enc

X_train, X_test = prepare_inputs(X_train, X_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


model = LinearRegression()
model.fit(X_train, y_train)

print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)


# In[ ]:


output = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': output.flatten()})
df


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, output))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, output))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, output)))


# In[ ]:


XX = submission_data[features].values

ohe = OrdinalEncoder()
ohe.fit(XX)
XX = ohe.transform(XX)


# In[ ]:


y_predict = model.predict(XX)

import csv

with open('5th_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Employee_ID", "Attrition_rate"])
    
    for i in range(3000):
        writer.writerow([submission_data['Employee_ID'][i], y_predict[i]])

