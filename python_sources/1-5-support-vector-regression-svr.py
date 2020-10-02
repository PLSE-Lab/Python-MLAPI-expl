#!/usr/bin/env python
# coding: utf-8

# # 1.5 Support Vector Regression (SVR)

# For a better understanding of SVR if u don't have any idea what so ever, visit the link: 
# [Support Vector Machining](https://towardsdatascience.com/support-vector-machines-svm-c9ef22815589)

# For better understanding of current notebook for beginners go through the links:
# 
#  [1.1 Data Preprocessing](http://www.kaggle.com/saikrishna20/data-preprocessing-tools)
# 
# 
# [1.2 Simple linear Regression](https://www.kaggle.com/saikrishna20/1-2-simple-linear-regression) 
# 
# 
# [1.3 Multiple linear Regression with Backward Elimination](http://www.kaggle.com/saikrishna20/1-3-multiple-linear-regression-backward-eliminat)
# 
# [1.4 Polynomial Linear Regression](https://www.kaggle.com/saikrishna20/1-4-polynomial-linear-regression)
# 
# 
# It basically tells u about the preprocessing & Linear Regression which will help u in understanding this notebook better

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # Level in the data
y = dataset.iloc[:, -1].values # salary in the data


# In[ ]:


print(X)


# In[ ]:


print(y)


# In[ ]:


y.shape


# Here y is a one dimensional array, we have to convert it to a 2 Dimensional array if we want to apply standard scaling.
# 
# Hence we do the reshape to convert it to 2D array.

# In[ ]:


y = y.reshape(len(y),1)


# In[ ]:


y.shape


# In[ ]:


print(y)


# ## Feature Scaling

# For the Support Vector Machining(SVM) - Support vector Regressor(SVR), we need to scale the data because it doesn't have coefficients like the one we seen in linear and polynomial regression.
# 
# We are going to scale the X (features) and y (target) seperately as they are different and for scaling and inverse scaling we will be using the same scaling as of those variable.
# 
# Only sc_X will be used for all scaling and inverse scaling/ transform of features and the same is applicable for y.
# 
# Never apply the scaling of features object to inverse scaling or transform the y (target) as their mean and standard deviation and other values saved in the scalar object will be different. so make sure to remember this one.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # scalar object of X
sc_y = StandardScaler() # scalar object of y
X = sc_X.fit_transform(X) # scaled values of X
y = sc_y.fit_transform(y) # scaled values of y


# In[ ]:


print(X)


# In[ ]:


print(y)


# ## Training the SVR model on the whole dataset

# The SVR has different kernals which we will be dealing in the upcoming notebooks but let's keep this one simple.
# 
# class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
# 
# 
# Visit the link for more details: 
# [SVR kernals and info](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
# 
# 
# Why did we train the model on the whole data?
# 
# Because we have very less data and hence we are training the model on entire data.

# In[ ]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Gaussian rbf kernal is used in here
regressor.fit(X, y)


# Now we have trained the model and now we can use the model to predict values from it.

# ## Predicting a new result

# Let's predict the salary of a person whose level is 6.5 i.e a person who will be working in between the level of 6  and 7

# Our model is trained on the scaled data and is related to the scaled target. 
# 
# If we want to predict salary for the 6.5 level we should provide the model after scaling it using sc_X so the model predicts the value which is a scaled output of y (target)
# 
# We have to inverse transform the output/prediction using the sc_y to convert into normal value which can be understood by us.
# 
# the sc_X.transform( )  takes a 2 Dimensional array so we have to convert the level 6.5 into a 2D array by [[6.5]] and this is supplied to the sc_X.transform( )
# 
# Now we need to provide this data to the model which is regressor.predict( ) which will be predicting a scaled output in terms of y
# 
# Then the sc_y.inverse_transform( ) is applied to make it look normal value.
# 

# In[ ]:


sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# ## Visualising the SVR results

# Here we will be using a unscalled data by inverse transform and plotting it.
# 
# Here the RED points are of the original data.
# 
# The blue line is the predicted values from our models which are quiet near to the real values except for the last one. which might be considered as an outlier by our model and hence.
# 
# From this we can be sure that the model has not been overfitted.

# In[ ]:


plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Visualising the SVR results (for higher resolution and smoother curve)

# In[ ]:


X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Like this notebook then upvote it.
# 
# # Need to improve it then comment below.
# 
# # Enjoy Machine Learning
