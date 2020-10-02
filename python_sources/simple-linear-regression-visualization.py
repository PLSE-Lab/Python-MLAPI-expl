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


# In[ ]:


Dataset=pd.read_csv('../input/Salary_Data.csv')


# In[ ]:


Dataset.isnull().sum()


# In[ ]:


Xloc=Dataset.iloc[:,0:1].values
Yloc=Dataset.iloc[:,1:2].values


# In[ ]:


Xloc.shape


# In[ ]:


from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(Xloc,Yloc,test_size=0.20,random_state=0)


# In[ ]:


XTrain.shape


# In[ ]:


YTrain.shape


# In[ ]:


XTest.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()


# In[ ]:


Regressor.fit(XTrain,YTrain)


# In[ ]:


X=np.array([[6.5]])


# In[ ]:


Regressor.predict(X)


# In[ ]:


YPred=Regressor.predict(XTest)


# In[ ]:


XTest


# In[ ]:


#Visualization Train Results
import matplotlib.pyplot as plt


# In[ ]:


plt.scatter(x=XTrain,y=YTrain,color='red')
plt.plot(XTrain, Regressor.predict(XTrain), color = 'blue')
plt.xlabel('Years of experiencce')
plt.ylabel('Salaries')
plt.title('Years of experience Vs Salaries')
plt.show()


# In[ ]:


#Visualization Test Results
plt.scatter(x=XTrain,y=YTrain,color='red')
plt.plot(XTest, Regressor.predict(XTest), color = 'blue')
plt.xlabel('Years of experiencce')
plt.ylabel('Salaries')
plt.title('Years of experience Vs Salaries')
plt.show()

