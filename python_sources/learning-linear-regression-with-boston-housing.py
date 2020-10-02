#!/usr/bin/env python
# coding: utf-8

# ## Learning Linear Regression ##
# In this notebook I am going to try to understand what goes on behind the scenes of linear regression using numpy and pandas <br>
# I am going to load in the Boston Housing dataset and go from there

# In[ ]:


# import numpy and pandas
# also import warnings and ignore them to keep notebook clean
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # ignores warnings
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# loading in the dataset and peeking at the first five rows
data = pd.read_csv('../input/BostonHousing.csv')
data.head()


# In[ ]:


# Taking some time here to play with some Pandas methods
#data.index # shows range of indices
#data.to_numpy() # returns df as a matrix
#data.describe() # some summary statistics. Notice the counts are all equal so there are no missing values--clean
#data.info() # gives us some more information and count and datatypes
#data.T # switches columns with rows
#data.sort_index(axis=1, ascending = False)# sorts the dataset by the index of each row
#data["tax"] # returns a single column (series)
#data[0:3] # row slicing
#data["tax"][3] # locate a specific value in column. Also can be done with data['tax'].loc[3]
#data.loc[:, ["tax","nox"]] # returns all the rows of two features
#data.isnull().sum() the first checks for missing values, the second sums them up


# In[ ]:


# Create empty list for coefficients
coefficients = []


# In[ ]:


# Creating helper functions to make model more viewable
def reshape_X(X):
    return X.reshape(-1,1) # numpy.reshape returns the m x n matrix of the arguments in this case


# In[ ]:


# The second helper matrix concatenates a feature of ones to the matrix
def concatenate_ones(X):
    ones = np.ones(shape=X.shape[0]).reshape(-1,1) # np.ones() creates an array of ones
    return np.concatenate((ones, X), 1) # concatenate basically appends the newly created vector of ones


# In[ ]:


# creating our function to fit the training data
def fit(X,y):
    global coefficients
    if len(X.shape) == 1:
        X = reshape_X(X)
    X = concatenate_ones(X)
    coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y) # math to obtain coeff ie. slope
    print(coefficients)


# In[ ]:


# creating a predict function to predict coefficient(??)
def predict(entry):
    b0 = coefficients[0] #initial slope
    other_betas = coefficients[1:] 
    prediction = b0 # initial prediction
    
    for xi, bi in zip(entry, other_betas): 
        # we avoid declaring two for loops by assigning xi to entry and bi to coef.
        # zip function creates a tuple out of the entry and other_betas
        prediction += (bi * xi)
    return prediction
        


# ### Training ###
# To train the model, we will fit the dataset without the median value feature (because that is what we are trying to predict) and test how accurate it is in predicting the target

# In[ ]:


X = data.drop("medv", axis=1).values # drops the medv column from the data
y = data["medv"].values # setting our target equal to the values we just dropped


# First I am going to run OLS (ordinary least squares regression) on the model, then do a train-test split

# In[ ]:


fit(X,y) # fits our dataset with the model


# In[ ]:


predict(X[0])


# So my prediction for the first median value is 30. Let's see for the whole dataset!

# In[ ]:


predictions = []
for row in X:
    predictions.append(predict(row))


# In[ ]:


results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})


# Let's try using scikit for linear regression

# In[ ]:


# importing matplotlib for graphs 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# In[ ]:


linear_regressor = LinearRegression()
linear_regressor.fit(X,y)
Y_pred = linear_regressor.predict(X)


# In[ ]:


plt.scatter(predictions, y)
plt.plot(predictions, Y_pred, color='red')
plt.show()


# Some things to figure out: how to properly format pyplot, other ML models I can use, figuring out train-test split

# ### References ###
# Inspiration for this primarily comes from: https://towardsdatascience.com/multiple-linear-regression-from-scratch-in-numpy-36a3e8ac8014 <br>
# ***Additional Resources*** <br>
# https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d <br>
# https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html <br>
# https://docs.scipy.org/doc/numpy/ <br>
# https://www.markdownguide.org/basic-syntax
