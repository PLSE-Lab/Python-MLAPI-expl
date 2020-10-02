#!/usr/bin/env python
# coding: utf-8

# # 1.8 Regression Model Selection

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
# [1.5 Support Vector Regression (SVR)](https://www.kaggle.com/saikrishna20/1-5-support-vector-regression-svr/edit/run/37240657)
# 
# [1.6 Decision Tree Regressor](https://www.kaggle.com/saikrishna20/1-6-decision-tree-regression)
# 
# [1.7 Random Forest Regression](https://www.kaggle.com/saikrishna20/1-7-random-forest-regression)
# 
# It basically tells u about the preprocessing & different models of Regression which will help u in understanding this notebook better

# # Ps: This notebook is not very friendly for the people who started in machine learning without any knowledge, if u want to understand everything then go through the previous notebooks from starting which have been explaiined in dumb and detail then u can understand this better.

# # Visit the link for the  [Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)  which we use to calculate the accuracy of our model and select the best model based on our accuracy.
# 
# * In here we are going to use the R-square metrics for evaluating the accuracy and determining the best model.
# 
# * There is no direct method to detect the best model applicable for our data we need to test each model on our data and take the model which has more accuracy for the starters, in the upcoming notebooks we will see different techniques which will help to finalize a particular model for our data.
# 
# * For now we go by trial and error method by using all models.
# 
# 
# 

# ![Screenshot%20%281%29.png](attachment:Screenshot%20%281%29.png)

# **In the above pic the dark line on which Yi^ is plotted is the predicted result and the red points Yi are the real values.
# We need to get the difference square of these two points as minimum as possible so that the predicted result will be closer to the real**

# ![Screenshot%20%282%29.png](attachment:Screenshot%20%282%29.png)

# SSres which is sum of squares of residuals(i.e. the difference btwn Yi & Yi^)
# SStot which is the sum of squares of total(i.e. the diff between Yi and Y average)
# 
# We can see the expression of R- square in the pic and the SStot will be constant the only thing we can decrease in a particular dataset is the SSres, to reduce it the predicted should be close to the actual/ real values.
# 
# R- square value in general lies between 0 & 1, there can Negative values also, which means the model is failed/ it's the worst.
# We try to bring the value of R- Square as nearest to 1 for better accuracy.

# ![Screenshot%20%284%29.png](attachment:Screenshot%20%284%29.png)

# We will be using adjusted R-square if there are more features i.e more columns in X because by using more features the SStot value will increase and make the value of R-Square to not decrease, to compensate this we use adjusted R- Square

# ![Screenshot%20%283%29.png](attachment:Screenshot%20%283%29.png)

# In the formula of adjusted R- Square the p is the number of regressors or features so if it increases due to negative sign it decreases the value of denomenator and inturn increase the value which is to be subtracted from 1. so we get a better results.

# # Multiple Linear Regression

# ## Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[ ]:


dataset = pd.read_csv('../input/best-regression-model/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training the Multiple Linear Regression model on the Training set

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[ ]:


from sklearn.metrics import r2_score
print('The score of Multiple linear regression is:')
r2_score(y_test, y_pred)


# ## Training the **Polynomial Regression model** on the Training set

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)


# ## Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[ ]:


from sklearn.metrics import r2_score
print('The score of Polynomial Regression is:')
r2_score(y_test, y_pred)


# ## Training the Decision Tree Regression model on the Training set

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[ ]:


from sklearn.metrics import r2_score
print('The score of Decision Tree Regression is:')
r2_score(y_test, y_pred)


# ## Training the Random Forest Regression model on the whole dataset

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[ ]:


from sklearn.metrics import r2_score
print('The score of Random Forest regression is:')
r2_score(y_test, y_pred)


# # Support Vector Regression (SVR)

# In[ ]:


y = y.reshape(len(y),1)


# ## Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)


# ## Training the SVR model on the Training set

# In[ ]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[ ]:


y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Evaluating the Model Performance

# In[ ]:


from sklearn.metrics import r2_score
print('The score of Support Vector Regression is:')
r2_score(y_test, y_pred)


# The score of Multiple linear regression is: 
# 0.9325315554761302
# 
# The score of Polynomial Regression is:
# 0.9458192606428147
# 
# The score of Decision Tree Regression is:
# 0.9226091050550043
# 
# The score of Random Forest regression is:
# 0.9615980699813017
# 
# The score of Support Vector Regression is:
# 0.9480784049986258
# 
# * All the numbers 0.94.. mean that the predicted values are very nearer to the original values and the predicted are less deviating from the original values.
# 
# * If the score is 1.0 which means that predicted outcomes are a perfect match to the real values.
# 
# * If the score is less, the model is not predicting a good outcome which we can rely on.
# 
# * R-square score generally tells us how near is the predicted values to the real values.
# 
# 
# * The maximum score is obtained from Random Forest model, so we will choose this model and even it's an ensemble model we prefer this model.
# 
# * There are much things to learn about model selection in the upcoming notebooks but u can get the basic idea of which model to select & how to select.
# 

# # Credits: Machine Learning A-Z, Udemy.

# ![Regression.JPG](attachment:Regression.JPG)

# # Like this notebook then upvote it.
# 
# 
# # Need to improve it then comment below.
# 
# 
# # Enjoy Machine Learning
