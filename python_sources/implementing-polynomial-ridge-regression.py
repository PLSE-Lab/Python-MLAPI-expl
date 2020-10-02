#!/usr/bin/env python
# coding: utf-8

# **           Implementing regularized (Ridge Regression) polynomial regression algorithms  from scracth!
# **
# 
# Quick outline:
# 
# * Generate synthetic data
# * Implement ridge polynomial regression from scrath
# * Compare with Scikit-learn solution 
# * Example of overfitting 
# * Effect of penalty term
# * Scikit-learn Ridge regression

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# First lets generate synthetic data fron sin() function and add gaussian noise to it.

# In[2]:


# number of input data, you can change this and see how it effects all the result!
N = 50 
np.random.seed(20)
x_train = np.sort(np.random.rand(N,1),axis=0)
noise = np.random.normal(0,0.3,size=(N,1))
t_train = np.sin(2*np.pi*x_train) + noise

# function to generate plot with different size
def vis_input_data(N,wit,hig):
    np.random.seed(20)
    plt.figure(figsize=(wit,hig))
    x_train = np.sort(np.random.rand(N,1),axis=0)
    noise = np.random.normal(0,0.3,size=(N,1))
    t_train = np.sin(2*np.pi*x_train) + noise

    plt.scatter(x_train,t_train,c='b',marker='o',label='noise added input data')
    plt.plot(np.linspace(0,1,50),np.sin(2*np.linspace(0,1,50)*np.pi),c='g',linewidth=2,label='function generates input data')
    plt.title('number of training data N =' + str(N)) 
    plt.xlabel('x');plt.ylabel('t')
    #plt.show()
    
vis_input_data(N,6,5)
plt.legend()
plt.show()


# This my own implementation using closed form solution. I implemented gradient descent in earlier kernel and it's similar. 
# The main task here is to generate input matrix (Q_train here) then the rest is same as linear regression. 
# Another feature I added here is penalty term (lam here) which takes role of regularization! 

# In[3]:


#ridge regression added, adjust lamda to make it work, if lamda = 0 becomes traditional
# LS with no regularization
lamda = 0.0  # ridge regression penalty term
poly_deg = 3

Q_train = np.zeros(shape = (N,poly_deg+1))
Q_train[:,0] = 1
for i in range(1,poly_deg+1):
    Q_train[:,i] = np.power(x_train,i).reshape((N,))    

W = np.linalg.pinv((Q_train.T.dot(Q_train) + lamda*np.eye(poly_deg+1))).dot(Q_train.T).dot(t_train)

vis_input_data(N,6,5)
plt.plot(x_train,Q_train.dot(W),'r',label='poly fit degree =' + str(poly_deg))
plt.legend()
plt.show()


# Make my own polynomial fit function using code above.

# In[4]:


def poly_fit(x_train,t_train,lam,polyfit_deg):
    
    Q_train = np.zeros(shape = (len(x_train),polyfit_deg+1))
    Q_train[:,0] = 1
    for i in range(1,polyfit_deg+1):
        Q_train[:,i] = np.power(x_train,i).reshape((len(x_train),))
    W = np.linalg.pinv((Q_train.T.dot(Q_train) + lam*np.eye(polyfit_deg+1))).dot(Q_train.T).dot(t_train)
    J_cos = 0.5*(Q_train.dot(W)-t_train).T.dot(Q_train.dot(W)-t_train)
    E_rms = np.sqrt(J_cos/len(x_train))
    return W,E_rms,Q_train


# Here I am showing root-mean-square error(RMSE) vs polynomial increase, as you can see if one increases polynomial degree( more complex model) the RMSE will decrease.
# But is it always good? If not, we will overfit our data!

# In[5]:


# plot roor-mean-square error
err = np.zeros((10,1))
for pol_deg in range(10):
    w,er,q_train = poly_fit(x_train,t_train,0,pol_deg)
    err[pol_deg]=er
    
plt.plot(err)
plt.xlabel('polynomial degree');plt.ylabel('RMSE')
plt.show()


# Lets bring in scikit-learn to compare with my implementaion.

# Below, I compare my implementation with scikit-learn. By looking at weight vectors and input matrix, both gives identical results.
# Note: Q_train in my code is same as fit transformed data from scikit-learn preprocessing method.

# In[6]:


from sklearn.preprocessing import PolynomialFeatures
sk_poly_deg=3
poly_feature = PolynomialFeatures(degree=sk_poly_deg,include_bias=False)
x_poly = poly_feature.fit_transform(x_train)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_poly,t_train)

vis_input_data(N,6,5)
plt.plot(x_train,lin_reg.predict(x_poly),'r',label='scikit poly fit degree =' + str(sk_poly_deg))
plt.legend()
plt.show()
print('scikit_learn input transformed matrix:',x_poly)
print('scikit_learn weight vectors:',lin_reg.intercept_,lin_reg.coef_)

vis_input_data(N,6,5)
plt.plot(x_train,Q_train.dot(W),'r',label='my implementaion poly fit degree =' + str(sk_poly_deg))
plt.legend()
plt.show()
print('my implementation input transformed matrix:',Q_train)
print('my implementation weight vectors:',W)


# Let us compare models with different polynomial degree.
# * Yellow curve is second degree polynomial which is underfitting data
# * Red curve is third degree polynomial which is correctfitting data
# * Black curev is overfit 40th degree polynomial which is overfitting data

# In[7]:


vis_input_data(N,7,6)
deg_1=2;deg_2=3;deg_3=40
w,er,q_train = poly_fit(x_train,t_train,0,deg_1)
plt.plot(x_train,q_train.dot(w),'y',label='poly fit degree =' + str(deg_1))
w,er,q_train = poly_fit(x_train,t_train,0,deg_2)
plt.plot(x_train,q_train.dot(w),'r',label='poly fit degree =' + str(deg_2))
w,er,q_train = poly_fit(x_train,t_train,0,deg_3)
plt.plot(x_train,q_train.dot(w),'k',label='poly fit degree =' + str(deg_3))
plt.legend()
plt.show()


# How do we handle overfit data? One way is using regularization.
# Below is using lam 0 and 0.02 to 60th degree poly fit. By adding regularization value of 0.02 , model performs better.

# In[8]:


vis_input_data(N,8,6)
deg_4 = 60;
w,er,q_train = poly_fit(x_train,t_train,0,deg_4)
plt.plot(x_train,q_train.dot(w),'y',label='poly fit degree =' + str(deg_4) + ' without regularization')
w,er,q_train = poly_fit(x_train,t_train,0.02,deg_4)
plt.plot(x_train,q_train.dot(w),'r',label='poly fit degree =' + str(deg_4) + ' with regularization')
plt.legend()
plt.show()


# Now I used sklern Ridge() function to do the same task, and results are the same. 
# In sklearn alpha is regularization term same as lam in my implementaion. 

# In[10]:


from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

vis_input_data(N,8,6)

model = make_pipeline(PolynomialFeatures(deg_4),Ridge(alpha=0.02,solver='cholesky'))
model.fit(x_train,t_train)
plt.plot(x_train,model.predict(x_train),'r',label='sklearn poly fit degree =' + str(deg_4) + ' with regularization')

model = make_pipeline(PolynomialFeatures(deg_4),Ridge(alpha=0.00,solver='cholesky'))
model.fit(x_train,t_train)
plt.plot(x_train,model.predict(x_train),'y',label='sklearn  fit degree =' + str(deg_4) + ' without regularization')
plt.legend()
plt.show()


# Let me know if you want to see other function implementations!
