#!/usr/bin/env python
# coding: utf-8

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


# From this given dataset we need to predict the Salary for a specific 'years of experience'

# In[ ]:


#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#Importing the dataset 

dataset= pd.read_csv('../input/Position_Salaries.csv')

dataset.head(6)


# From the Dataset, it can be seen that the the outcome variable(Salary) is a continous one,so we will go for for bulidng a model using 'Regression method'.
# 
# There are two predictor variables columns. The second column provides the level and the first column provide the postion. Each data is unique in the first column which is categorical where the second column represents each level with numerical value so we dont need to 'LabelEncode' the categorical column.
# And we can ignore the first column.
# 

# In[ ]:


#Assiging the dataset into X and y varialbles


# In[ ]:


X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2:]

print(X.head())
print(y.head())


# As the dataset is having very less examples we will not split the data into train test

# In[ ]:


#Fitting the linear Regression model into the dataset


# In[ ]:


from sklearn.linear_model import LinearRegression

regressor_LR = LinearRegression().fit(X, y)

y_predict = regressor_LR.predict(X)

y_predict


# In[ ]:


#Visualising the dataset using the Linear Rergression o/p


# In[ ]:


plt.scatter(X , y, color = 'red')
plt.plot(X, regressor_LR.predict(X), color = 'blue')
plt.title('Positon Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# The above graph shows that the plotted line of the predicted variables is not going on hand on hadn with the scattered values.
# 
# So,the linear regression model for this dataset is not yileding any good results, so now we will try for Polynomial regression to check.

# In[ ]:


#Fitting the Polynomial Regression into the dataset


# For implementing the Polynomial Regression into the X variable, we need to convert the X variable into the polynomial Feature

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

X_poly


# In[ ]:


#Now fitting the regression model using the X_poly


# In[ ]:


regression_PR = LinearRegression().fit(X_poly, y.values)

y_predict_PR= regression_PR.predict(poly_reg.fit_transform(X))

y_predict_PR


# In[ ]:


#Visualising the dataset using the Polynomial Rergression o/p


# In[ ]:


plt.scatter(X , y, color = 'red')
plt.plot(X, regression_PR.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Positon Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


#Visualising the polynomial regression model using higer definition


# In[ ]:


X_new = X.values


# In[ ]:


X_grid = np.arange(min(X_new), max(X_new), 0.1)
X_df = pd.DataFrame(X_grid)
X_df.head()
X_df.shape


# In[ ]:



plt.scatter(X , y, color = 'red')
plt.plot(X_df, regression_PR.predict(poly_reg.fit_transform(X_df)), color = 'purple')
plt.title('Positon Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




