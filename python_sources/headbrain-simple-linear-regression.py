#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/headbrain.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# <li>Our dataset has no categorical values we can move forward.</li>
# <li> we don't have any null values in our dataset.

# In[ ]:


df.shape


# In[ ]:


# Taking x and y variables
X = df['Head Size(cm^3)'].values
Y =  df['Brain Weight(grams)'].values


# In[ ]:


X.shape


# In[ ]:


Y.shape


# #### Method 1:  munual coding

# In[ ]:


mean_X = np.mean(X)
mean_Y = np.mean(Y)

n = len(X)

num =0
denom = 0

for i in range(n):
    num += (X[i]-mean_X)* (Y[i]-mean_Y)
    denom +=(X[i]-mean_X)**2
m = num/denom
c = mean_Y - (m*mean_X)

print(m,',',c)


# Here , we calculate m and b. Now we need to find the line

# In[ ]:


plt.scatter(X,Y)


# ### creating dummy test set

# In[ ]:


min_x = np.min(X)-100
max_x = np.max(X)+100


# In[ ]:


x = np.linspace(min_x,max_x,1000)


# In[ ]:


y = m*x+c


# In[ ]:


plt.scatter(X,Y,color='g')
plt.plot(x,y,color='r')
plt.title('Simple Linear Regression')
plt.xlabel('Head size cm^3')
plt.ylabel('Brain weight in grams')


# #### Calculating the error

# In[ ]:


sum_pred = 0
sum_act = 0

for i in range(n):
    y_pred = (m*X[i]+c)
    sum_pred += (Y[i]-y_pred)**2
    sum_act +=(Y[i]-mean_Y)**2

r2 = 1-(sum_pred/sum_act)
print(r2)


# Here we can observe that we got R**2> 0.5 . so we have good model

# In[ ]:


def predict(x):
    y = m*x + c
    print(y)


# In[ ]:


predict(4177)


# here we predict the brain wieght for given head size(cm^3)

# #### Method 2:  using scikit learn

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X  = X.reshape((n,1))


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


lg = LinearRegression()


# In[ ]:


lg.fit(X,Y)


# In[ ]:


y_pred = lg.predict(X)


# In[ ]:


mse = mean_squared_error(Y,y_pred)


# In[ ]:


rmse = np.sqrt(mse)


# In[ ]:


r2_score = lg.score(X,Y)


# In[ ]:


print(rmse)
print(r2_score)


# we got the same error R**2 value as above method-1

# In[ ]:


lg.predict([[4177]])


# In[ ]:


lg.intercept_

