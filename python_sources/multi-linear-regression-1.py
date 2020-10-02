#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#importing all libraries 
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from ipywidgets import interact


# In[ ]:


#attaching file from system and uploading to data
df=pd.read_csv("../input/Real estate.csv")
df.head()




# In[ ]:


#converting independent variable x into 2D and dependent variable y into 1D
x = df.iloc[:, 1:4].values
y = df.iloc[:, -1].values


# In[ ]:


#creating with predifined 
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5,random_state=16)
print(xtrain[0][0])


# In[ ]:


#Creating with our own functions
def Train_Test_Split(x,y,test_size=0.5,random_state=None):
    n=len(y)
    if len(x)==len(y):
        if random_state:
            np.random.seed(random_state)
        shuffle_index=np.random.permutation(n)
        x=x[shuffle_index]
        y=y[shuffle_index]
        test_data=round(n*test_size)
        xtrain,xtest=x[test_data:],x[:test_data]
        ytrain,ytest=y[test_data:],y[:test_data]
        return xtrain,xtest,ytrain,ytest
    else:
        print("Data should be in same size.")
    


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5,random_state=3)
xtrain[0][0]


# In[ ]:


#training the machine
model=LinearRegression()
model.fit(xtrain,ytrain)
   


# In[ ]:


m =model.coef_
c =model.intercept_


# In[ ]:


#prediction by mathematics by using formula
#y=m1*x1+m2*x2+m3*x3+c
x=[12,123,7]#random value
y_pred=sum(m*x)+c
y_pred


# In[ ]:


#predicting by math function
x=[12,123,7]#random value
y_pred=model.predict([x])
y_pred


# In[ ]:


#prediction for houseprice
ytest_predict=model.predict(xtest)


# In[ ]:


def HousePricePredict(Age, Distance, Stores):
    y_pred=model.predict([[Age,Distance,Stores]])
    print("HousePrice is:",y_pred[0])
    


# In[ ]:


HousePricePredict(12,132,9)


# In[ ]:


age_min=df.iloc[:,1].min()
age_max=df.iloc[:,1].max()
dis_min=df.iloc[:,2].min()
dis_max=df.iloc[:,2].max()
st_min=df.iloc[:,3].min()
st_max=df.iloc[:,3].max()


# In[ ]:


interact(HousePricePredict, Age = (age_min, age_max),Distance = (dis_min, dis_max), Stores  = (st_min, st_max))


# In[ ]:




