#!/usr/bin/env python
# coding: utf-8

# **Copy and Edit this notebook**. **Rename your copy**. **Do not Edit this copy**

# **Run the codes below and try to understand the executions**

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


# In[ ]:


advertising = pd.read_csv('../input/tvmarketing/tvmarketing.csv')#Telling the computer what to read and from where
advertising.head(5)
advertising.tail(5)


# In[ ]:


advertising.info()#get information about the dataset


# In[ ]:


advertising.describe()


# In[ ]:


import seaborn as sns#load plotting packages
get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(advertising, x_vars=['TV'], y_vars= 'Sales', height=4, aspect=0.9, kind='scatter')


# **Assigning the x and y values**

# In[ ]:


#y=mx+c or y= c+ m*TV
X= advertising['TV']
X.head()


# In[ ]:


y= advertising['Sales']
y.head()


# **Splitting Data**

# In[ ]:


from sklearn.model_selection import train_test_split #split your data to predict from available data
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100 )


# In[ ]:


# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


import numpy as np
X_train = X_train[:, np.newaxis]#add a second dimension
X_test = X_test[:, np.newaxis]
print(X_train.shape)#see how it looks like


# **Importing and Running Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression #import package for linear regression
lr = LinearRegression()#Creating a LinearRegression object
lr.fit(X_train, y_train)#Run Linear Regression on prepared data


# **Predicting with the test values**

# In[ ]:


y_pred = lr.predict(X_test)#Derive predictions from the test dataset: X_test
#print (y_pred.shape)


# In[ ]:


import matplotlib.pyplot as plt#Import another plotting package
lr_line = [i for i in range(1,61,1)]#create indices for your prediciton
graph = plt.figure()#create an object named graph to title the graph
plt.plot(lr_line, y_test, color="green", linewidth=3)#Actual Values
plt.plot(lr_line, y_pred, color="purple", linestyle='-.')#Predicted Values
graph.suptitle("Actual versus Predicted Values")#Title
plt.xlabel("Index")#label the axis
plt.ylabel("Sales")


# **Now Create a graph that shows the error in this Linear Prediction**

# In[ ]:


#This time you do not need to import the package since you've done it earlier
#1 create indices for your prediciton
#2 create an object named graph to title the graph
#3 plot the Error Values that are defined by the difference between the Actual Values and the Predicted Values 
#4 Title the graph as "Error in Predicted Values"
#5 xlabel should be plotted as "Index"
#6 ylabel should be plotted as "ytest-ypred"


# **Uncomment the following code to see and understand error values and scores of this Linear Prediction model**

# In[ ]:


#from sklearn.metrics import mean_squared_error, r2_score
#mse = mean_squared_error(y_test, y_pred)
#r_squared = r2_score(y_test, y_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[ ]:


#print('Mean_Squared_Error :' ,mse)
#print('r_square_value :' ,r_squared)


# In[ ]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :' ,r_squared)


# In[ ]:


plt.scatter(y_test,y_pred, color="red")#plot to see y values from test set versus predicted y values


# **Congrats! Now you know how to run Linear Regression models and how to see their error margins. You can import your own datasets and do regression on them using the code provided here.**

# In[ ]:


import pandas as pd
tvmarketing = pd.read_csv("../input/tvmarketing/tvmarketing.csv")

