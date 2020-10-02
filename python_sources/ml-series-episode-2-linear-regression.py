#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# ## Simple Linear Regression Model

# In statistics, **simple linear regression** is a linear regression model with a single explanatory variable. That is, it concerns two-dimensional sample points with one independent variable and one dependent variable (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line) that, as accurately as possible, predicts the dependent variable values as a function of the independent variables. The adjective simple refers to the fact that the outcome variable is related to a single predictor.

# **Case study:** To predict the Salary of an employee of abc company based on the number of experience of an employee using simple linear regression in python. Here, the independent variable would be 'YearsExperience' and dependent variable would be 'Salary'. Also, plot the predicted value using regression line on the plot using matplotlib library.

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Read dataset

# In[ ]:


#supress warning
pd.set_option('mode.chained_assignment', None)

#Importing dataset
df = pd.read_csv("../input/Salary_Data.csv")
X = df.iloc[:,:-1]
Y = df.iloc[:,1]


# ## Data Preparation

# ### Split the dataset into training and testing dataset

# In[ ]:


#split data into train and test dataset
from sklearn.model_selection import train_test_split  #(for python2)
#from sklearn.model_selection import train_test_split  (for python3)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=1/3, random_state=0)


# ### Model Fitting

# In[ ]:


#Fitting simple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ### Model Prediction

# In[ ]:


#predicting the test dataset
y_pred = regressor.predict(X_test)


# ## Visualizing the results

# ### Training dataset 

# In[ ]:


#visualize the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# ### Testing dataset

# In[ ]:



#visualize the testing set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Testing set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# We can infer from the above results that the model performed well for linear data and got good predictions for the new data.

# Citation: https://www.udemy.com/, https://en.wikipedia.org/
