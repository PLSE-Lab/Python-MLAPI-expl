#!/usr/bin/env python
# coding: utf-8

# # Custom SGD implementation for Linear Regression on Boston House dataset
#    ### Importing libraries
#    ### Data Loading and Preprocessing
#    ### Fixing Total Number of Iterations for 1. SKLearn SGD and 2. Custom SGD
#     
#    ## 1. SKLearn Implementation of SGD
#        1.1 Plot and MSE for the SK Learn SGD
#        1.2 Obtaining Weights from SKLearn SGD
#    ## 2. Custom Implementation Of SGD
#        . Setting custom parameters
#        2.1 Plot and MSE for the Custom SGD
#        2.2 Obtaining Weights from Custom SGD
#    ## 3. Improved Custom SGD
#        . Setting new custom parameters
#        3.1 Plot and MSE for the Custom SGD Improved
#        3.2 Obtaining Weights from Custom SGD Improved
#    ## Comparison
#    ## Conlusion
#    ## References
#    ----------------------------------------------

# ## Importing libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_boston
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split
print("DONE")


# ## Data Loading and Preprocessing:

# In[ ]:


boston_data=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
Y=load_boston().target
X=load_boston().data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


# In[ ]:


# data overview
boston_data.head(3)


# In[ ]:


print(X.shape)
print(Y.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


## Before standardizing data
x_train


# In[ ]:


# standardizing data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)


# In[ ]:


## After standardizing data
x_train


# In[ ]:


x_test


# In[ ]:


## Adding the PRIZE Column in the data
train_data=pd.DataFrame(x_train)
train_data['price']=y_train
train_data.head(3)


# In[ ]:


x_test=np.array(x_test)
y_test=np.array(y_test)


# In[ ]:


type(x_test)


# ## Fixing Total Number of Iterations for 1. SKLearn SGD and 2. Custom SGD

# In[ ]:


n_iter=100


# In[ ]:





# # 1. SKLearn Implementation of SGD

# In[ ]:


# SkLearn SGD classifier
clf_ = SGDRegressor(max_iter=n_iter)
clf_.fit(x_train, y_train)
y_pred_sksgd=clf_.predict(x_test)
plt.scatter(y_test,y_pred_sksgd)
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()


# ## 1.1 MSE for the SK Learn SGD

# In[ ]:


print('Mean Squared Error :',mean_squared_error(y_test, y_pred_sksgd))


# ## 1.2 Obtaining Weights from SKLearn SGD

# In[ ]:


# SkLearn SGD classifier predicted weight matrix
sklearn_w=clf_.coef_
sklearn_w


# In[ ]:


type(sklearn_w)


# # 2. Custom Implementation Of SGD

# In[ ]:


def My2CustomSGD(train_data,learning_rate,n_iter,k,divideby):
    w=np.zeros(shape=(1,train_data.shape[1]-1))
    b=0
    cur_iter=1
    while(cur_iter<=n_iter): 
#         print("LR: ",learning_rate)
        temp=train_data.sample(k)
        #print(temp.head(3))
        y=np.array(temp['price'])
        x=np.array(temp.drop('price',axis=1))
        w_gradient=np.zeros(shape=(1,train_data.shape[1]-1))
        b_gradient=0
        for i in range(k):
            prediction=np.dot(w,x[i])+b
#             w_gradient=w_gradient+(-2/k)*x[i]*(y[i]-(prediction))
#             b_gradient=b_gradient+(-2/k)*(y[i]-(prediction))
            w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction))
            b_gradient=b_gradient+(-2)*(y[i]-(prediction))
        w=w-learning_rate*(w_gradient/k)
        b=b-learning_rate*(b_gradient/k)
        
        cur_iter=cur_iter+1
        learning_rate=learning_rate/divideby
    return w,b


# In[ ]:


def predict(x,w,b):
    y_pred=[]
    for i in range(len(x)):
        y=np.asscalar(np.dot(w,x[i])+b)
        y_pred.append(y)
    return np.array(y_pred)


# ## Setting custom parameters: As mentioned in the assignment video, the following parameters are set
# 1. As mentioned Learning Rate=1 initially and will be divided by 2 over the Iterations
# 2. As mentioned size of K is kept as k=10

# In[ ]:


w,b=My2CustomSGD(train_data,learning_rate=1,n_iter=100,divideby=2,k=10)
y_pred_customsgd=predict(x_test,w,b)

plt.scatter(y_test,y_pred_customsgd)
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))


# ## 2.1 MSE for the Custom SGD

# In[ ]:


print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))


# ## 2.2 Obtaining Weights from Custom SGD

# In[ ]:


# weight vector obtained from impemented SGD Classifier
custom_w=w
print(custom_w)
print(type(custom_w))


# # 3. Improved SGD

# ## Changes made in the following parameters to improve the result
# 1. Learning Rate=0.01 initially and will not be divided by any number over the Iterations
# 2. size of K is kept as k=10
# 3. Iterations = 1000

# In[ ]:


w,b=My2CustomSGD(train_data,learning_rate=0.01,n_iter=1000,divideby=1,k=10)
y_pred_customsgd_improved=predict(x_test,w,b)

plt.scatter(y_test,y_pred_customsgd_improved)
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd_improved))


# ## 3.1 MSE for the Custom SGD Improved

# In[ ]:


print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd_improved))


# ## 3.2 Obtaining Weights from Custom SGD Improved

# In[ ]:


# weight vector obtained from impemented SGD Classifier
custom_w_improved=w
print(custom_w_improved)
print(type(custom_w_improved))


# #  Comparision 

# In[ ]:


###
from prettytable import PrettyTable
x=PrettyTable()
x.field_names=['Model','Weight Vector','MSE']
x.add_row(['SKLearn SGD',sklearn_w,mean_squared_error(y_test, y_pred_sksgd)])
x.add_row(['Custom SGD',custom_w,mean_squared_error(y_test,y_pred_customsgd)])
x.add_row(['Custom SGD Improved',custom_w_improved,mean_squared_error(y_test,y_pred_customsgd_improved)])
print(x)


# ## Conclusion
# 1. We can see the our plain custom SGD performed very poor as compared to the SKLearn SGD.
# 1. When we changed the learning rate and  Batch size, the our custom SGD performed as good as the SKLearn SGD

# ## Reference

# [1] My Medium Blog: https://medium.com/@nikhilparmar9/simple-sgd-implementation-in-python-for-linear-regression-on-boston-housing-data-f63fcaaecfb1<br>
# [2]https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/ <br>
# [3]https://www.kaggle.com/premvardhan/stocasticgradientdescent-implementation-lr-python <br>
# [4]https://www.kaggle.com/arpandas65/simple-sgd-implementation-of-linear-regression/notebook <br>
# [5]https://www.kaggle.com/tentotheminus9/linear-regression-from-scratch-gradient-descent<br>
