#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


data=pd.read_csv('../input/melb_data.csv')#Reading CSV file and saving it as a panda dataframe


# In[3]:


print(data.describe())#Data Description


# In[4]:


data.columns #Displaying the cumn names of the dataframe 


# In[5]:


print(data.columns)


# In[6]:


p_data=data.Address
p_data.head() #head gives first 5 rows of the dataframe
p_data.tail() #tail gives last 5 rows of the dataframe


# In[7]:


data.Price.head()


# In[8]:


col=['Landsize', 'BuildingArea']
t=data[col]
t.describe()


# In[9]:


data=data.dropna() #Removing NULL Values
#We are dividing the data into predictors(INDEPENDENT) and prediction(DEPENDENT) arrays
predict=['Price']
y=data[predict]
predictors=['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X=data[predictors]


# In[10]:


#using decision tree model
from sklearn.tree import DecisionTreeRegressor as dt
model = dt()
model.fit(X, y)
    


# In[11]:


model.predict(X.head())#predicting our independent variable for first 5 roes


# In[12]:


model.predict(X.tail())


# In[13]:


model.predict([[2,1.0,156.0,79.0,1900.0,-37.8079,144.9934]]) #prediction for custom input


# In[14]:


#MODEL VALIDATION using mean absolute error
from sklearn.metrics import mean_absolute_error
predict=model.predict(X)
mean_absolute_error(y,predict)


# In[16]:


#MODEL VALIDATION by dividing data into training and tesing dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dt

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model =dt()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# In[17]:


#OPtimium fitting using max_leaf_nodes as a meetric for finding out optimum number of leaves to use
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[19]:


#Using Random forrests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y.values.ravel())
#The function expects train_y to be a 1D array,ravel() converts the 2d array to 1d array 
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

