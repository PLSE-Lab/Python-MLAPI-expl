#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression model with gradient descent in python from scratch

# __Here in this notebook just to understand the working of gradient descent in python from scratch we took a simple house price prediction dataset and only took one input feature(# of bedrooms) for simplicity, and defined a gradient descent function for intercept and coefficients and compared the results with sklearn Linear regression model__

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


cols=['bedrooms','price']
df=pd.read_csv("../input/kc_house_data.csv",usecols=cols)
df.head()


# In[ ]:


x=df['bedrooms']
y=df['price']


# In[ ]:


# standardizing the input values
def standardize(x):
    return (x-np.mean(x))/np.std(x)


# In[ ]:


X=standardize(x)
X=np.c_[np.ones(x.shape[0]),X]


# In[ ]:


alpha=0.01
m=y.size
np.random.seed(23)
theta=np.random.rand(2)
iterations=2000

def gradient_descent(x,y,theta,alpha,iterations):
    previous_costs=[]
    previous_thetas=[theta]
    for i in range(iterations):
        prediction=np.dot(x,theta) #line equation (theta*x)
        error=prediction-y # error value
        cost=1/(2*m)*np.dot(error.T,error) #cost function
        previous_costs.append(cost)
        theta=theta-(alpha*(1/m)*np.dot(x.T,error)) #updating theta values
        previous_thetas.append(theta)
    return previous_costs,previous_thetas
costs,thetas=gradient_descent(X,y,theta,alpha,iterations)


# In[ ]:


plt.title('Cost Function')
plt.xlabel('# of iterations')
plt.ylabel('Cost')
plt.plot(costs)
plt.show()


# __Comparing our results with sklearn Linear regression model__

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)
lm=LinearRegression().fit(x_train,y_train)
predictions=lm.predict(x_test)


# In[ ]:


print("Linear Regression model Intercept:",lm.intercept_)
print("Linear Regression model Theta1",lm.coef_[1])


# __As we can see the results of our gradient descent function are Intercept: 540088.14075994,Theta1: 113200.90438675__
# __are much closer to the results of sklearn Linear regression model i.e. Intercept: 540059.9446691773,Theta1: 540059.9446691773__

# ### Thankyou for your interest,please update the notebook if u find any improvements required.
