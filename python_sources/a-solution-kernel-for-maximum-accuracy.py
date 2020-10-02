#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


data = pd.read_csv("../input/trainData3D.csv")
dx = data.X.values 
dy = data.Y.values
dz = data.Z.values
X = np.append(dx.reshape(-1,1),dy.reshape(-1,1), axis =1)
Y = dz
X.shape, Y.shape


# # Visualizing the dataset

# In[ ]:


ax = Axes3D(plt.figure())
ax.scatter(X[:,0],X[:,1],Y)
ax


# # Clearly, locally weighted regression is one solution

# In[ ]:


def getW(X,q,tau):
    
    #Create W
    m = X.shape[0]
    W = np.eye(m) 
    
    for i in range(m):
        W[i,i] = np.exp(-np.dot((X[i]-q),(X[i]-q).T)/(2*tau*tau))
    
    return W
    
def getTheta(X,Y,q,tau):
    m = X.shape[0]
    ones = np.ones((m,1))
    q = np.append(np.array([1]), q, axis = 0)
    X = np.append(ones, X, axis = 1)
    W = getW(X,q,tau)
    Y = Y.reshape((-1,1))
    
    theta = np.dot(np.linalg.pinv(np.dot(np.dot(X.T,W),X)),np.dot(np.dot(X.T,W),Y))
    return theta,W
    
    


# In[ ]:


theta,W = getTheta(X,Y,[0.6,0.7],0.1)


# In[ ]:


print(theta.shape)
print(W)


# In[ ]:


# X_Test = np.linspace(-20,20,100)
X_Test = pd.read_csv("../input/testData3D.csv").values
# print(X_Test)
Y_Test = []

for xt in X_Test:
#     print(xt)
    theta,W = getTheta(X,Y,xt,0.73)
#     print(xt)
    pred = theta[0][0]*1 + theta[1][0]*xt[0] + theta[2][0]*xt[1]
    Y_Test.append(pred)
    
Y_Test = np.array(Y_Test)
Y_actual = pd.read_csv("../input/actualYTest3D.csv").values
Y_Test.shape, Y_actual.shape


# In[ ]:


from sklearn.metrics import r2_score
r2_score(Y_actual,Y_Test)


# # Locally weighted really performs well, it adapts to the function quickly.
# 
# One thing you will notice is that as you increase tau, the training accuracy is actually behaving abnormally. There is a good reason for that. Can you figure it out?
# 
# Hint: Implement linear regression to see

# In[ ]:



ax = Axes3D(plt.figure())
ax.scatter(
    X_Test[:,0],
    X_Test[:,1],
    Y_actual.reshape(-1,1)
)
plt.title("Redrawn predictions!")

X_Test[:,1].shape,X_Test[:,0].shape, Y.shape


# Above is weighted_regression in practice. It is a deterministic approach.

# In[ ]:


from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest = tts(X,Y, random_state = 1)


# # Can you figure out why linear regression works so well?

# In[ ]:


from sklearn.linear_model import LinearRegression as LR
reg = LR()
reg.fit(xtrain,ytrain)
reg.score(xtest,ytest)


# # Can you figure out why support vector regressor performs so badly?
# #### Hint, draw plot, see decision boundary

# In[ ]:


from sklearn.svm import SVR
reg = SVR()
reg.fit(xtrain,ytrain)
reg.score(xtest,ytest)


# # Just another ensemble technique you can read about.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor as GBR
reg = GBR()
reg.fit(xtrain,ytrain)
reg.score(xtest,ytest)


# In[ ]:




