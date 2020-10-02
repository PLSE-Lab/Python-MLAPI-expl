#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


def load_data():
    dataset = pd.read_csv('../input/50_Startups.csv')
    return dataset


# In[ ]:


#Loading data
dataset = load_data()
print(dataset.head())
#Finding missing values
print(dataset.isnull().sum())


# In[ ]:


#Finding outliers in the column R&D Spend
import seaborn as sn
sn.boxplot(dataset['R&D Spend'])


# In[ ]:


#Finding outliers in the column Administration
sn.boxplot(dataset['Administration'])


# In[ ]:


#Finding outliers in Marketing Spend
sn.boxplot(dataset['Marketing Spend'])


# In[ ]:


#Finding outliers in the column Profit
sn.boxplot(dataset['Profit'])


# In[ ]:


#Profit column is having outlier so, we have to handle the outlier
#finding z-score to handle outliers
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(dataset['Profit']))
print(z)


# In[ ]:


#if the z score value is higher than the threashhold value then it is the outlier
threashhold = 3
print(np.where(z>threashhold))


# In[ ]:


#Finding unique values in State column to create lable encode from scratch
State_label = dataset['State'].values
print(np.unique(State_label))


# In[ ]:


unique_label = np.unique(State_label)
unique_label


# In[ ]:


#Lable encoding from scratch
California = np.array([])
New_York = np.array([])
a = 0
b = 1
Florida = np.array([])
for i in dataset['State']:
    if(i in unique_label[0]):
        California = np.append(California, b)
        New_York = np.append(New_York, a)
        Florida = np.append(Florida, a)
    elif(i in unique_label[1]):
        California = np.append(California, a)
        New_York = np.append(New_York, a)
        Florida = np.append(Florida, b)
    else:
        California = np.append(California, a)
        New_York = np.append(New_York, b)
        Florida = np.append(Florida, a)
print(California)
print(len(California))
print(Florida)
print(dataset['State'])


# In[ ]:


#Dropping the State column
dataset['California'] = California
dataset['New_York'] = New_York
dataset = dataset.drop(columns = 'State')
print(dataset.head())


# In[ ]:


#Data devided into X and Y (X is independent features, and Y is target or dependent feature)
dataset = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'California', 'New_York', 'Profit']]
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,5].values
#print(X)
print(Y)


# In[ ]:


v = np.ones((50,1))
v.shape
X = np.c_[v, X]
X.shape


# In[ ]:


#Converting X to X transpose matrix
XT = X.T
XT.shape


# In[ ]:


#X transpose * X
XtX = XT.dot(X)
print(XtX.shape)
print(XtX)


# In[ ]:


#Finding inverse of X transpose X 
invXtX = np.linalg.inv(XtX)
invXtX


# In[ ]:


Y = Y.reshape(50,1)
XtY = XT.dot(Y)
XtY.shape


# In[ ]:


#Finding co-efficients or theta values
Theta = invXtX.dot(XtY)
print(Theta.size)
print(Theta)


# In[ ]:


theta0 = Theta[0]
theta1 = Theta[1]
theta2 = Theta[2]
theta3 = Theta[3]
theta4 = Theta[4]
theta5 = Theta[5]
print(theta0)


# In[ ]:


X.shape
print(Theta.shape)


# In[ ]:


#finding predictions
y_hat = np.array([])
y_hat1 = 0
for i  in range(X.shape[0]):
    for j in range(X.shape[1]):
        #print(X[i][j])
        y_hat1 = y_hat1 + (Theta[j] * X[i][j])
    y_hat=np.append(y_hat,y_hat1, axis=0)
    y_hat1 = 0
y_hat = y_hat.reshape(50,1)
y_hat.shape


# In[ ]:


#R squared
numer = 0.0
denom = 0.0
y_mean1 = 0
for i in range(len(Y)):
    y_mean1 = y_mean1 + Y[i]
y_mean1 = y_mean1/len(Y)
y_mean = np.mean(Y)
#print(y_mean1)
#print(y_mean)
for i in range(len(Y)):
    numer = numer + ((y_hat[i] - y_mean1)**2 )
    denom = denom + ((Y[i] - y_mean1)**2)
Rsquared = numer / denom
print("RSquared =",Rsquared)

