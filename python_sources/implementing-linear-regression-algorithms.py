#!/usr/bin/env python
# coding: utf-8

# **           Implementing linear regression algorithms (Closed form and Batch gradient descent) from scracth!
# **
# 
# Quick outline:
# 
# * Data cleaning
# * Data visualization
# * Closed form least square algorithm implementation 
# * Scikit-learn solution comparision
# * Batch gradient descent algorithm implementaion
# * Scikit-learn stochastic gradient descent solution
# 

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


data = pd.read_csv('../input/bigcar_matlab.csv')


# Here, lets take a quick look at data see if there is anything missing. 400 non-null value out of 406 entries, so it needs cleaning. 

# In[29]:


print(data.head())
print(data.info())


# Drop Nan rows from original data, or you can chose to replace with some statistical value. 

# In[30]:


data = data.drop(data.index[data['Horsepower'].isnull()])
print(data.info())


# Lets take a look at the data. What do we see?
# * Could use linear linear regression to fit a model, but it does not have to be.
# * Feature and target values are in different scale, keep in mind applying feature scaling at some point.
# 
# <font size="4">Caution: If you are implementing algorithm by yoursel and often see result weight vector blows up, that means you NEED to do feature scaling before do anything!</font>

# In[31]:


plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2)
plt.xlabel('Horsepower');plt.ylabel('Weight')
plt.show()


# Below is my own implementation of least square closed form solution (aka,Normal equation solution).
# Important note: Again, if features are not normalized , the algrorithm will not converge. I will show at the very end!

# In[32]:


# construct input and target values
data_arr = np.array(data); n = len(data_arr)
horsepower = data_arr[:,0].reshape(n,1)
weight = data_arr[:,1].reshape(n,1)

# normalization of feature and target
x = np.c_[np.ones((n,1)),weight/np.max(weight)]
t = horsepower/np.max(horsepower)

# closed form least square solution
Weight_closed = np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x)).dot(t)

# map weight vector to orginal space , before normalization
Weight_closed[0] = max(horsepower)*Weight_closed[0]
Weight_closed[1] = (max(horsepower)*Weight_closed[1])/(max(weight))
print('weight vectors:',Weight_closed)

# plotting data with model
y_plot_closed = Weight_closed[0] + Weight_closed[1]*(weight)
plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')
plt.plot(data['Weight'],y_plot_closed,c='b',label='closed form linear model')
plt.legend()
plt.xlabel('Weight');plt.ylabel('Horsepower')
plt.show()


# Below is the solution from scikit-learn. And you can see weight vectors are the same!

# In[33]:


from sklearn import linear_model
reg_lin = linear_model.LinearRegression()
reg_lin.fit(weight,horsepower)
print ('weight vectors: ',reg_lin.intercept_,reg_lin.coef_)

plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')
plt.plot(weight,reg_lin.predict(weight),c='b',label='scikit-learn closed form solution')
plt.legend()
plt.xlabel('Weight');plt.ylabel('Horsepower')
plt.show()


# Now, let's implement the batch gradient descent algorithm. This can be extended to more robust stochastic gradient descent algorithm. I will do it in the future if anyone is intersted!

# In[47]:


# assing initial parameters 
w_gd = np.random.randn(2,1); lr = 0.5; misfit = 10; max_iteration = 5000
cos_func_val=np.zeros((max_iteration,1))

for i in range(max_iteration):
    thetax = x.dot(w_gd)
    #gradient of cost function
    grad = (1/n)*(np.transpose(x).dot(thetax-t))
    #update weight vectors
    w_gd = w_gd - lr*grad
    #calculate cost function during iteration for plotting
    cos_func_val[i] = (thetax-t).T.dot(thetax-t)
    
    # early stopping based on misfit barely changes between iterations
    if i>0:
        misfit = abs(cos_func_val[i]-cos_func_val[i-1])
    if misfit <0.00001:
        break

# map weight vector to orginal space , before normalization
w_gd[0] = max(horsepower)*w_gd[0]
w_gd[1] = (max(horsepower)*w_gd[1])/(max(weight))
print('weight vectors:',w_gd)

plt.plot(cos_func_val[0:i])
plt.xlabel('iteration number');plt.ylabel('cost function value')
plt.show()


# Let's plot result from our implementation, also (above) you can find a nice decrease of cost function as iteration increases which confirms algorithm works! Results from closed form and batch gradient descent solutions are pretty close by looking at graph and weight vectors.

# In[35]:


plt.figure(figsize=(20,10))
y_plot_closed = Weight_closed[0] + Weight_closed[1]*(weight)
y_plot_gd = w_gd[0] + w_gd[1]*(weight)
plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')
plt.plot(data['Weight'],y_plot_closed,c='b',label='closed form linear model')
plt.plot(data['Weight'],y_plot_gd,c='k',label='gradient descent linear model')
plt.legend()
plt.xlabel('Weight');plt.ylabel('Horsepower')
plt.show()

print('weight vectors from closed form:',Weight_closed)
print('weight vectors from gradient descent:',w_gd)


# Below is the example of what happens if data is not properly scaled. Notice weight vectors!

# In[36]:


sgd_lin = linear_model.SGDRegressor(loss='squared_loss',penalty=None,max_iter=5000,eta0=0.5,tol=0.0001)
sgd_lin.fit(weight,horsepower.ravel())
print ('weight vectors: ',sgd_lin.intercept_,sgd_lin.coef_)


# I used the simplest scaling method and it works! Of course you can use other scaling methods from scikit-learn library!
# **Again, beaware to do weight vector transformation for plotting purpose if scaling applied to data!!!**

# In[37]:


sgd_lin = linear_model.SGDRegressor(loss='squared_loss',penalty=None,max_iter=5000,eta0=0.5,tol=0.0001)
sgd_lin.fit(weight/max(weight),horsepower.ravel()/max(horsepower))
sgd_lin.intercept_ = max(horsepower)*sgd_lin.intercept_
sgd_lin.coef_ = (max(horsepower)*sgd_lin.coef_)/(max(weight))
print ('weight vectors: ',sgd_lin.intercept_,sgd_lin.coef_)


# Let's plot scikit-learn solution.

# In[38]:


plt.figure(figsize=(20,10))
y_plot_sgd = sgd_lin.intercept_ + sgd_lin.coef_*(weight)
plt.plot(weight,reg_lin.predict(weight),c='b',label='skitlearn closed form solution')
plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')
plt.plot(data['Weight'],y_plot_sgd,c='k',label='skitlearn SDG solution')
plt.legend()
plt.xlabel('Weight');plt.ylabel('Horsepower')
plt.show()


#    **Let me know what you think and feel free to leave a comment;)**

# In[ ]:




