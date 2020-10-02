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


#  I import the required libraries 

# In[ ]:


#import libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# I import the dataset 

# In[ ]:


# import dataset:
dataset=pd.read_csv("../input/Salary_Data.csv")


# I see the dataset to find out the (dependent) and (independent ) parameters:

# In[ ]:


dataset.head()


#  I am going the describe the dataset:

# In[ ]:


dataset.describe()


# I am going to the make matrix form of dataset  in the form of independent  form:

# In[ ]:


# make the dataset into independen form:
X=dataset.iloc[:,:-1].values


# In[ ]:


X


# I am going to the make matrix form of the dataset in the form of dependent form:

# In[ ]:


# make the datset into the dependent form:
y=dataset.iloc[:,1].values


# In[ ]:


y


# i am going to the see the graphical repesentation of yearExperience  with the Salary
# by using the pairplot:

# In[ ]:


# dataset visualization:
sns.pairplot(dataset)


# I am going to the another way of visualization yearExperience  with Salary  by using the 
# heatmap:

# In[ ]:


sns.heatmap(dataset)


# By the help of Scikit librari we easily find out our (training data) and (target data).
# from the scikit libraries we are going to import the train_test_split:

# In[ ]:


# spliting the dataset into train and test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# After the import the train_test_spilt.
# we import the our regressoin algorithm
# and also make the object of the regression alg

# In[ ]:


#fitting the model on linear_regression:
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:


#prediction of new result:
y_pred=regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


# visualixation of train dataset:
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Expriance')
plt.ylabel('salary')
plt.title('Expirence vs Salary')
plt.show()


# In[ ]:


# vizualixation of test dataset :
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Expirence Vs Salary')
plt.xlabel('Expirence')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




