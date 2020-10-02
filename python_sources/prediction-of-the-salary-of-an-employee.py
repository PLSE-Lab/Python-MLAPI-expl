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


# *Importing necessary libraries *

# In[ ]:


import numpy as np    #numpy is a library for making computations
import matplotlib.pyplot as plt    #it is a 2D plotting library
import pandas as pd    # pandas is mainly used for data analysis
import seaborn as sns    # data visualization library
get_ipython().run_line_magic('matplotlib', 'inline')
#magic function to embed all the graphs in the python notebook


# In[ ]:


#Import the salary dataset
#Reading the csv file using the read_csv function of the pandas module
df=pd.read_csv("../input/Salary_Data.csv")
#The read_csv function converts the data into dataframe


# In[ ]:


#Look how the data looks like
#Lets print the first 5 rows of the dataframe
df.head()


# In[ ]:


X=df.iloc[:,:-1].values
#Storing the column 1 in X and column 2 in y
y=df.iloc[:,:1].values


# **Visualization of the Dataset to understand the data in a better way**

# In[ ]:


sns.distplot(df['YearsExperience'],kde=False,bins=10)
#This plot is used to represent univariate distribution of observations


# In[ ]:


#Show the counts of observations in each categorical bin using bars
sns.countplot(y='YearsExperience',data=df)


# In[ ]:


#Plotting a barplot
sns.barplot(x='YearsExperience',y='Salary',data=df)


# In[ ]:


#Representing the correlation among the columns using a heatmap
sns.heatmap(df.corr())


# In[ ]:


sns.distplot(df.Salary)


# ***Now we will use the scikit learn package to create the Linear Regression model***

# Split the data into training and testing set

# In[ ]:


from sklearn.model_selection import train_test_split
#splitting the data using this module and setting the test size as 1/3 . Rest 2/3 is used for training the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# **Creating the Linear Regression Model and Fitting the training data**

# In[ ]:


#importing the linear regression model
from sklearn.linear_model import LinearRegression
#creating the model
lr=LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)
#fitting the training data


# In[ ]:


X_train.shape 
#Counting the number of observations in the training data


# In[ ]:


y_train.shape


# **Predicting the Test Results**

# In[ ]:


y_pred=lr.predict(X_test)
y_pred
#Predicted data


# **Visualizing the training data**

# Plotting the actual y training values VS the y values predicted by the model using training data

# In[ ]:


plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title('Salary vs Years of Experience (Training Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of an employee')
plt.show()


# We see that the data is fitted so well and the predicted and actual data is almost the same

# **Visualizing the Test Data**

# Plotting the y test data vs y predicted data

# In[ ]:


plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,lr.predict(X_test),color='red')
plt.title('Salary vs Years of Experience (Test Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of an employee')
plt.show()


# We see that the predicted data fits the regression line so well

# **Calculating the errors so as to check the difference between the actual value and predicted model value... There are certain metrics to find these error such as Mean Squared Error, Root Mean Squared Error and Mean Absolute Error****

# In[ ]:


from sklearn import metrics
print('Mean Absolute Error of the Model:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error of the Model: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error of the Model: ',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))


# Looking at the values we see  that the error is very minute and hence we can see our model gives very accurate values

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,y_pred)
#This shows that our model is completely accurate
#R value lies between 0 to 1. Value of 1 represents it is completely accurate


# In[ ]:




