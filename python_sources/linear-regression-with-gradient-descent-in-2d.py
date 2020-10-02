#!/usr/bin/env python
# coding: utf-8

# # Handling and  Visualising Data
# 
# Let's start by playing a bit with the dataset, first and foremost load all the necessary libraries, and then let's take a look at the data

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import math
from scipy import stats
df = pd.read_csv('../input/train.csv')
df = df[df['LotArea']<30000] # remove outliers
df.head()


# As we can see, whe have an Id, some features of different nature and at last our label, SalePrice. As we're going to use Gradient Descent in 2D I'll choose just one of our features to build a linear regression upon, in this instance LotArea.
# Having chosen our x and y, we can start normalising our feature and labels to have similar ranges, a process known as scaling.
# 
# With a normalised x and a normalised y, we can plot the first scatter, just to visualise our dataset.
# 

# In[2]:


x = df['LotArea']
max_x = x.max()
mean_x = x.mean()
x_norm = list(map(lambda elem: (elem-mean_x)/max_x,list(x)))
y = df['SalePrice']
max_y = y.max()
mean_y = y.mean()
y_norm = list(map(lambda elem: (elem-mean_y)/max_y,list(y)))

plt.figure(figsize=(10,10))
plt.scatter(x_norm,y_norm, alpha=0.6)
plt.show()


# Well I can't say much about our result, but maybe our model will!
# Let's start by defining our cost function and its derivative
# 
# # Fitting the data a.k.a. training the model
# 
# Let's define some numbers such as **n**, the size of the dataset, **err**, the cost function and its derivative and **der**, its gradient. 
# We also initialise the regressor coefficients **a** and **b**, and set the hyperparameters **alpha** and **max_iter**, which are, respectively, the step of the gradient descent and the number of iterations computed before stopping the algorithm.
# 
# We define the cycle where the gradient descent is actually computed and then plot the error for each iteration

# In[3]:


n = len(x)
a = 0.01 # [a,b] also known as theta
b = 1
alpha = 0.5
max_iter = 3000

def err (a,b,x,y): # J(theta) = 1/2n * (y - theta dot x + theta_0)^2
    est_y = list(map(lambda elem: a*elem+b,x)) # theta dot x + theta_0
    err = np.subtract(np.array(est_y),np.array(y)) # y^ - y
    err_2 = np.power(err,2)
    return (1/(2*n)) * sum(err_2)

def der(a,b,x,y): # gradient of J(theta) = [1/n * (y - theta dot x + theta_0) dot x, 1/n*(y - theta dot x + theta_0) dot 1]
    est_y = list(map(lambda elem: a*elem+b,x)) # theta dot x + theta_0
    err = np.subtract(np.array(est_y),np.array(y)) # y^ - y
    return (1/n * np.dot(err,x),1/n*sum(err))

err_iter = []
for i in range(0,max_iter): 
    deriv = der(a,b,x_norm,y_norm)
    a -= alpha*deriv[0]
    b -= alpha*deriv[1]
    err_iter.append(math.sqrt(err(a,b,x_norm,y_norm)))
    
index = list(range(0,max_iter))
#plt.figure(figsize=(10,10))
plt.plot(index,err_iter)
plt.show()


# As you can see the error decreases super fast at first and then flattens nicely

# In[4]:


def y_from_x(x): # theta dot x + theta_0
    return a*x+b
plt.figure(figsize=(10,10))

plt.scatter(x_norm,y_norm, alpha=0.6)
plt.plot([0,1],[y_from_x(0),y_from_x(1)])
plt.show()


# And this is the result! A fine linear regressor for an unforgiving dataset, as you can see, the predictions might leave you unsatisfied, let's review our R squared and the p-value

# In[5]:


# R^2 is ESS / TSS, Explained sum of squares over total sum of squares
y_avg=sum(y_norm)/len(y_norm)
y_hat = np.apply_along_axis(y_from_x, 0, x_norm)
ESS = np.sum(np.add(y_hat, -1*y_avg)**2)
TSS = np.sum(np.add(y_norm, -1*y_avg)**2)

degf = len(y_norm)

se = math.sqrt(np.sum(np.add(y_norm,-y_hat)**2)/degf) / math.sqrt(np.sum(np.add(x_norm,-sum(x_norm)/len(x_norm))**2))

slope, intercept, r_value, p_value, std_err = stats.linregress(x_norm,y_norm)

tscore = (a - 0)/(se)
p = stats.t.sf(tscore,df=degf)
prova = stats.t.ppf(p_value,df=degf)
print("R Squared:", ESS/TSS)
print("p-value", p)


# Nice stat

# In[6]:


df_2 = pd.read_csv('../input/test.csv')
x_test = df_2['LotArea']
y_test = list(map(y_from_x,x))
output = pd.DataFrame(df_2['Id'])
output['SalePrice'] = pd.Series(y_test)
output.to_csv("output.csv", index=False)

