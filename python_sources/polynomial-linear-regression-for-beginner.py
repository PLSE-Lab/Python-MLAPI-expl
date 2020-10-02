#!/usr/bin/env python
# coding: utf-8

# Here we have data of Salary of employees based on their designation.We will be using varies  regression methods to  predict the salary of an employee.This is a work in process and please do vote if you like my work.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **1. Importing modules needed for the work **

# In[ ]:


import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
plt.style.use('fivethirtyeight')  
import warnings
warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook


# **2. Importing the salary data **

# In[ ]:


dataset=pd.read_csv('../input/Position_Salaries.csv')
dataset


# We can see that the dataset has 10 levels and the corresponding salary paid to the employee

# In[ ]:


X=dataset.iloc[:,1:2].values  # For the features we are selecting all the rows of column Level represented by column position 1 or -1 in the data set.
y=dataset.iloc[:,2].values    # for the target we are selecting only the salary column which can be selected using -1 or 2 as the column location in the dataset
X


# Position and the Level of the employee represent the same thing.So in out machine learning model it is sufficient to consider only on feature.In this case we can select the column Level.

# **3. Splitting the data into training and test data **

# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# We will not be splitting the dataset as here we have only 10 data points.So we can consider all the values for out training purpose.

# **4.Feature Scaling **

# In[ ]:


"""from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)"""


# We need not use feature scaling for this particular model 

# **5.1 Linear Regression **

# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)


# **5.2 Ploynomial Linear Regression **

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg2=PolynomialFeatures(degree=2)
X_poly=poly_reg2.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


# In[ ]:


poly_reg3=PolynomialFeatures(degree=3)
X_poly3=poly_reg3.fit_transform(X)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly3,y)


# **6.1 Visualizing Linear Regression result **

# In[ ]:


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth Or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# **6.2 Visualizing Plolynomial Linear Regression result **

# In[ ]:


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg2.fit_transform(X)),color='blue')
plt.plot(X,lin_reg_3.predict(poly_reg3.fit_transform(X)),color='green')
plt.title('Truth Or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# We can clearly see that the Polynimal Linear Regression model has much better result compared to Linear Regression Model.As we increase the degree of the polynomial regression the correlation increases.4th degree Polynomial Linear Regression will gives us the best correlation for salary data.

# **6.3 Smoothing out the curve using more points on X axis**

# In[ ]:


X_grid=np.arange(min(X),max(X),0.1) # This will give us a vector.We will have to convert this into a matrix 
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_3.predict(poly_reg3.fit_transform(X_grid)),color='blue')
#plt.plot(X,lin_reg_3.predict(poly_reg3.fit_transform(X)),color='green')
plt.title('Truth Or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# We can see that by increasing the number of points in the X we can smooth out the curve.

# **7. Predicting the salrary of the employee **

# In[ ]:


lin_reg.predict([[6.5]])  # We are assuming the level of the employee is 6.5


# In[ ]:


lin_reg_2.predict(poly_reg2.fit_transform([[6.5]]))


# In[ ]:


lin_reg_3.predict(poly_reg3.fit_transform([[6.5]]))


# We can see that the linear regression predicted values if higher.So this is not a good model.We can see that Polynomial Linear regression has better prediction.Accuracy of Polynomial Linear regression increases with the increase in the degree of the Polynomial.
