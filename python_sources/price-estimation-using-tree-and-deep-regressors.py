#!/usr/bin/env python
# coding: utf-8

# # Price Estimations Using Multiple Regressors
# 
# Welcome to the price estimation kernel by Uddeshya Singh
# 
# ![](https://aia.es/wp-content/uploads/2012/10/price_estimation.jpg)
# 
# I will be covering:
# * Basic EDA
# * Feature Engineering
# * Deep Learning Modelling
# * Benchmarking Scheme
# 
# All for the **Total Price** of a particular customer

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# # The Dataset!

# In[ ]:


df.head()


# # Exploratory Data Analysis
# Let's go through various visualisations to have a deeper grasp at what our data really represents!

# In[ ]:


sns.countplot(x = 'SeniorCitizen', data = df, hue = 'gender')


# As one might guess, 0 represents that the customer is not a senior citizen . Hence, one may infer that only about 14% of the customers are actually 60+. A pretty classic scenario in current generation.

# In[ ]:


sns.countplot(x = 'Contract', data = df, hue = 'gender')


# Another inference that might be drawn from the Contract's count plot that **Month-to-Month** plans are best served and preferred among the consumers. To attract retentivity, one may think about offering **Free Subscriptions and premium support** for the first month as a trial!

# In[ ]:


sns.countplot(x = 'DeviceProtection', data = df, hue = 'gender')


# Looking at the scenario, not many customers prefer Device Protection If offered. But for the sake of business, the company may leverage into offering a wider internet access and there maybe a **30-40%** chance of getting a new preemium customer who may opt for Device Protection. Similar scenarios can be scene in Tech Support too!

# # Feature Engineering
# 
# Now let's move on to the feature engineering section. The motive of this part is to make sure that we convert our features in algorithm processable quantities.
# 
# What I will be doing is mostly applying* One Hot Encodings in the columns with >2 unique values* and explicitly change **Gender** Column to match the 0 and 1 categories.

# In[ ]:


# Categorizing Male and Female in 1s and 0s
def gender_labels(element):
    if element == 'Male':
        return 0
    elif element == 'Female':
        return 1
# Making a new column in the dataframe
df['GenderLabel'] = df['gender'].apply(gender_labels)

#Dropping the original gender column
df.drop(['gender'] ,axis = 1, inplace=True)    


# With the step one of feature engineering done, let's move on to code our utility to put **binary labels** on the columns with unique values of **Yes, No and No Internet**!

# In[ ]:


# Now, to relable the columns which have just "Yes" and "No" as their entries!
listOfColumns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Churn', 'StreamingMovies', 'StreamingTV', 'DeviceProtection', 'PaperlessBilling']

# The Labelling Function
def Labelizer(input_value):
    '''Returns 1 for a Yes and a 0 for any other No'''
    if input_value == 'Yes':
        return 1
    else:
        return 0
    
for i in listOfColumns:
    newCol = i+'_label'
    df[newCol] = df[i].apply(Labelizer)

df.drop(listOfColumns, axis = 1, inplace=True)


# With the binary operations all done, its time to fix our remaining columns with **One Hot Encoding** utility

# In[ ]:


list_nonBinary = ['Contract', 'PaymentMethod', 'InternetService']
for i in list_nonBinary:
    df = pd.concat([df, pd.get_dummies(df[i])], axis = 1)
    df.drop([i], axis = 1, inplace=True)

#print("Post feature Engineering, the columns are as follows : ", df.columns.values)


# ## The Total Charges Dilemma
# One thing which I shoud mention here is that I couldn't really fix the total charges columns to the datatype of **float** by the convention *.astype*.
# So I confirmed to the maneover mentioned below to get through that errors!

# In[ ]:


df['TotalChargesNew'] = df['tenure']*df['MonthlyCharges']
df.drop(['MonthlyCharges', 'TotalCharges'], axis = 1, inplace = True)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Dropping the Customer ID for obvious reasons
df.drop(['customerID'], axis = 1, inplace=True)


# # Simple Train Test split

# In[ ]:


from sklearn.model_selection import train_test_split
y = df['TotalChargesNew']
X = df.drop(['TotalChargesNew'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # Model Making
# 
# I will be using the following models to checkout the estimation performance!
# 1. Decision Tree Regressor
# 2. XGBoost Regressor
# 3. Linear Regressor
# 4. Gradient Boost Regressor
# 5. A Naive Deep Learning Model

# # Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
myRegressor = DecisionTreeRegressor(criterion='mse')
myRegressor.fit(X_train, y_train)
prediction = myRegressor.predict(X_test)


# Checking out our predicitons along side by making a simple dataset 

# In[ ]:


final_df_Decision = pd.DataFrame({'Predictions':prediction, 'True' : y_test})
final_df_Decision.head()


# # The Curves matter
# An excellent curve fitting result of first 250 test values and seeing how our model fares against them

# In[ ]:


Prediction_Line = go.Scatter(
    x = [i for i in range(250)],
    y = prediction[:250]
)
Actual_Line = go.Scatter(
    x = [i for i in range(250)],
    y = y_test.values[:250]
)

data = [Prediction_Line, Actual_Line]
iplot(data)


# In[ ]:


from sklearn.metrics import mean_squared_error
print("Decision Tree metrics are about accurate to %.2f dollars (+ and -)"% (mean_squared_error(prediction, y_test)**0.5))


# # Linear Regression

# In[ ]:


from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.linear_model import LinearRegression
clf2 = LinearRegression()
clf2.fit(X_train, y_train)
preds2 = clf2.predict(X_test)

print("RMSE Score for Linear Regressor is %.2f"%(mean_squared_error(y_test, preds2))**0.5)


# # Gradient Boost Regressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
clf3 = GradientBoostingRegressor()
clf3.fit(X_train, y_train)
preds3 = clf3.predict(X_test)

print("RMSE Score for Gradient Boost Regressor is %.2f"%(mean_squared_error(y_test, preds3))**0.5)


# # XGBoost Regressor

# In[ ]:


from xgboost import XGBRegressor
clf4 = XGBRegressor()
clf4.fit(X_train, y_train)
preds4 = clf4.predict(X_test)

print("RMSE Score of XGBoost Regressor is %.2f"%(mean_squared_error(y_test, preds4))**0.5)


# # DeepLearning Model Design!
# 
# The following is the model design of our naive model which we will be testing our dataset upon

# In[ ]:


from keras.models import Sequential
from keras.layers import (Dense, Dropout, BatchNormalization)

model = Sequential()
model.add(Dense(25, input_dim = 25, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.9))

model.add(Dense(12, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(6, activation='relu'))
model.add(Dropout(0.9))

model.add(Dense(1))
model.compile(optimizer='adam', loss= 'mse')


# In[ ]:


model.summary()
model.lr = 0.05


# In[ ]:


history = model.fit(X_train, y_train, epochs=1000, verbose=2)


# # Benchmarks
# 
# Have a look at the benchmark performances and decide for yourselves, which is the best regressor and whom you are going to opt!

# In[ ]:


benchmarks = pd.DataFrame({"Naive Deep Learning Model" : (sum(history.history['loss'])/len(history.history['loss']))**0.5,
                          "XG Boost Regressor" : mean_squared_error(y_test, preds4)**0.5,
                          "Gradiant Boost Regressor" : mean_squared_error(y_test, preds3)**0.5,
                          "Decision Tree Regressor" : mean_squared_error(y_test, prediction)**0.5,
                          "Linear Regressor" : mean_squared_error(y_test, preds2)**0.5
                          }, index = range(1)).T
benchmarks.columns=['RMSE']
benchmarks['Regressor'] = benchmarks.index
benchmarks.index = range(5)
benchmarks


# # Note : The RMSE score is almost equivalent to the actual tolerance of the prediction.
# 
# ## As you may have noticed by now, XG Boost and Gradient Boost comes out on top with an approximate error of 111 dollars!

# In[ ]:




