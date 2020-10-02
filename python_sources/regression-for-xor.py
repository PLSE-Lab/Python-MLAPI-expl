#!/usr/bin/env python
# coding: utf-8

# # Regression for XOR:
# This notebook is based on the concept of [derivation in context of Logistic Regression](https://www.kaggle.com/hamzafar/derivation-in-context-of-logistic-regression). In this notebook we will extend the the beforementioned work and generalized it to *m* number of rows and *n_x* feature set (columns). The folow of notebook is as follows:
# 1. A function will generate data set of desired length and width (row, columns)
# 2. The labelled is created using *XOR*; the deata genereted in the above step then passed to this step to get the value of *XOR* operation
# 3. Implement Regression model (Generalized to deal with dynamic data set
# 4. Discuss the performance Regression on different dataset and different parameters.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting graph


# ### Generate Data
# To discuss Generalized behavior of Regression model, by Generalized we mean it can work on any shape of data, we have created the *generate_bits* functions. The function is simple as it takes desired number of rows *m* and number of feature *n_x* and it randomly generated binary data i.e. *0* and *1*.

# In[ ]:


def generate_bits(n_x, m):
# Generate a m x n_x array of ints between 0 and 1, inclusive:
# m: number of rows
# n_x : number of columns per rows/ feature set
    np.random.seed(1)
    data = np.random.randint(2, size=(n_x, m))
    return(data)


# ### Create Labels:
# For training/updating derivatives of parameters weight and bias, the loss function determine the difference between actual values and the activation values. The actual value is the value that each example(row) has as it label. Like the the actual value of *OR* operation:
# 
# 
# 
# 
# \begin{equation*}
# 1 + 0 = 1 =actualValue >:[or_{operation} = +]\\
# \end{equation*}
# 
# The *generate_label* function below takes data as input and apply *XOR* operation row wise.
# 

# In[ ]:


def generate_label(data, m):
    # generate label by appyling xor operation to individual row
    # return list of label (results)
        # data: binary data set of m by n_x size
    lst_y = []
    y= np.empty((m,1))
    k = 0
    for tmp in data.T:
        xor = np.logical_xor(tmp[0], tmp[1])

        for i in range(2, tmp.shape[0]):
            xor = np.logical_xor(xor, tmp[i])
    #     print(xor)
        lst_y.append(int(xor))
        y[k,:] = int(xor)
        k+=1
    return(y.T)


# ### Regression (Genearlized to m by n_x data-set):
# In [derivation in context of Logistic Regression](https://www.kaggle.com/hamzafar/derivation-in-context-of-logistic-regression) we have created computational graph using only tow input features *x1* and *x2*. Now we are generalizating the pervious concept with feature length equal to *n_x* . The computational graph for this generalization would be same as before but a minor change in the input to graph i.e. the *x's, w's*. The bias *b* will be only single value. Refer to following figure the concept is described:
# ![](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Regression_for_XOR/1.jpg?raw=true)

# To synchronize with the computational graph above, we will arrange dataset in the format that can easy fits with it. So,  we arrange dataset into *n_x by m* shape. This will easily fit the above conpcet because at each step you will pass all feature values of single sample at once that will result in conveyor belt scenario. Like you first chop front edge of dataset and pass it to graph and do computation simultaneously. 
# The activation are then stacked after the sigmoid node where you will be computing loss with compare it to actual values of *XOR*. Let's consider following figure to validate the concept:
# 

# ![](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Regression_for_XOR/2.jpg?raw=true)

# The three functions below **(i). initialize_param, (ii). sigmoid and (iii) get_activation_loss** are helper to get loss of the input data.
# The all three function will use matrix operations to optimize computations as for loops are computationaly expensive. and we have done two major changes in **get_activation_loss** function than the   as:
# 1. Instead of multiple each single input feature to respective weight, we applid matrix multiplication operation:
# \begin{equation*}
# w1*x1 + w2*x2+ ... +w_{nx}*x_{nx} = np.dot(w.T,x)
# \end{equation*}
# 2. Each sample in *Forward Pass* will yield into a loss and we will be having m losses as of total number of sample so we compute average of loss and it is called as *cost*
# 

# In[ ]:


def sigmoid(z):
    # Takes input as z and return sogmoid of value
    s = 1 / (1 + np.exp(-z))
    return s


# In[ ]:


def intialize_param(n_x):
    # initialize paramaters w and b to zero and return them
    # size of w equal to size fo feature set and b is single value
        # n_x: size input feature    
    w = np.zeros(shape=(n_x, 1))
    b = 0
    return(w,b)


# In[ ]:


def get_activation_loss(x, w, b):
    # this function return action, cost and z values
        # x: input data
        # w: weights
        # b: bias
    z = np.dot(w.T, x) + b
    a = sigmoid(z)

    cost = (1/m) * np.sum(-1 * (y * np.log(a) + (1 - y) * (np.log(1 - a))))
    return(a,cost, z)


# The partial derivative of b and w w.r.t. loss (of single sample) was calculated as:

# \begin{equation*}
# \frac{\partial loss}{\partial b} = (a-y)\\
# \frac{\partial loss}{\partial w} = x*(a-y)\\
# \end{equation*}

# But now we are having *m* number of sample that is giving us *m* activations *a* and actual values *y* . To get single value of cost we just get take average of each sample gradient.
# 
# 
# \begin{equation*}
# \frac{\partial loss}{\partial b} = (a_1-y_1)\\
# \frac{\partial loss}{\partial b} = (a_2-y_2)\\
# \frac{\partial loss}{\partial b} = (a_3-y_3)\\
# ...\\
# \frac{\partial loss}{\partial b} = (a_m-y_m)\\
# so,\\
# \frac{\partial cost}{\partial b} = \sum\limits_{i=1}^m  \frac{1}{m} * (a-y)\\
# \end{equation*}
# The similar will be used for partial derivatives of *cost* w.r.t *w*.
# 
# The function **update_paramters** implements the above equation in matrix operation format.

# In[ ]:


def update_paramters(x, w, b, a, y, lr, m):
    # find the gradient of paramaters and update them (w and b)
        # x: input data 
        # w, b: parameters (w and b)
        # a, y: activation and actual values
        # m, lr: total number of rows, learning rate
    dw = (1/m) * np.dot(x,(a-y).T)
    db = (1/m) * np.sum(a - y)
    
    w = w - (lr*dw)
    b = b - (lr*db)
    
    return(w, b)


# The one important thing that must be considered while taking partial derivate *cost* w.r.t to *w*, the shape of must be consist in the iteration process. So, below is the figure that give an intuation about the cycle. For example if we have sample of shape *(5,10)* and it is multiplied by matrix w *(5,1)* then the resulting would be of shape *(1,10)* and while taking partial derivative the resulting matrix would be the same shape of *w* i.e. *(5,1)*
# The figure below descibe the cycle of updating parameter *w* while keep track of dimensions.:
# 
# ![](https://github.com/hamzafar/deep_learning_toys/blob/master/images/Regression_for_XOR/3.jpg?raw=true)

# In[ ]:


def plt_res(lst, ylab, lr):
    #This will plot the list of values at y axis while x axis will contain number of iteration
    #lst: lst of action/cost
    #ylab: y-axis label
    #lr: learning rate
    plt.plot(lst)
    plt.ylabel(ylab)
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    plt.show()


# In[ ]:


def optimize_paramters(x, y, w, b, n_x, lr, num_iter):
    # this function returns upadated values of parameters and cost
    # It first initialize parameters and update them by computing partial derivatives
    # Then loop over 
        # x: input data
        # y: actual values (labels)
        # w, b: parameters
        # n_x, lr: input feature length, learning rate
        # num_iter: number of cycle
    lst_cost = []

    w, b = intialize_param(n_x)

    for i in range(num_iter):
        a,cost,z = get_activation_loss(x, w, b)
#         print('cost after iteration %i: %f' %(i,cost))
        w, b = update_paramters(x, w, b, a, y, lr, m)
        lst_cost.append(cost)
    
    return(w, b, lst_cost)


# we have implemented Regression using *10000* , *100000*  and *1000000* samples and sotred their respective learned paramteres (weights and bias) and cost of the function.

# In[ ]:


n_x = 50
m = 10000
num_iter = 1000
w, b = intialize_param(n_x)
x = generate_bits(n_x,m)
y = generate_label(x, m)
lr = 0.07

#w_s, b_s, lst_cost_s represent values when sample set is 10000
w_s,b_s, lst_cost_s = optimize_paramters(x, y, w, b, n_x, lr, num_iter)
##----------##

m = 100000
# num_iter = 150
w, b = intialize_param(n_x)
x = generate_bits(n_x,m)
y = generate_label(x, m)
# lr = 0.07

#w_m, b_m, lst_cost_m represent values when sample set is 100000
w_m,b_m, lst_cost_m = optimize_paramters(x, y, w, b, n_x, lr, num_iter)
##----------##

m = 1000000
num_iter = 15
w, b = intialize_param(n_x)
x = generate_bits(n_x,m)
y = generate_label(x, m)
# lr = 0.07

#w_l, b_l, lst_cost_l represent values when sample set is 1000000
w_l,b_l, lst_cost_l = optimize_paramters(x, y, w, b, n_x, lr, num_iter)


# **Prediction**
# In prediciton step, we have done following two steps:
# 1.  Calculate $\hat{Y} = A = \sigma(w^T X + b)$
# 2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector Y_prediction
# 
# To validate how the Regression is performing; we just created new dataset of *0.1 times m* size and computed its predictions from the **get_prediction** function. The label of created data is also generated that are matched with the prediction values to get accuracy. Since we have trained weights for two different datasets, we computed the accuracy for both.

# In[ ]:


def get_prediction(x, w, b, m):
    # returns the prediction on the dataset
        # x: input data (unseen)
        # w, b: parameters weights and bias
        # m: total sample set
    a = sigmoid(np.dot(w.T, x) + b)
    y_prediction = np.zeros((1, m))
    for i in range(a.shape[1]):
        y_prediction[0,i] = 1 if a[0, i] > 0.5 else 0
    return(y_prediction)


# In[ ]:


def get_accuracy(y, y_prediction, m):
    # return the accuracy by calculated the difference between actual and predicted label
        # y: actual values
        # y_prediction: prediction acquired from the get_prediction
        # m: total number of sample
    df = pd.DataFrame()
    df['actual'] = y[0]
    df['prediction'] = y_prediction[0]
    df['compare']= df['prediction'] == df['actual']

#     print(df[df['compare']==True])
#     print('Accuracy: ' ,len(df[df['compare']==True]['compare'])/m)
    return(len(df[df['compare']==True]['compare'])/m)


# In[ ]:


tm = int(0.1 * m)
x = generate_bits(n_x, tm)
y = generate_label(x, tm)

y_prediction = get_prediction(x, w_s, b_s, tm)
acc_s = get_accuracy(y, y_prediction, tm)

y_prediction = get_prediction(x, w_m, b_m, tm)
acc_m = get_accuracy(y, y_prediction, tm)

y_prediction = get_prediction(x, w_l, b_l, tm)
acc_l = get_accuracy(y, y_prediction, tm)


# In[ ]:


print('------- 10000 training set-------------')
print('Accurcy at 10000 training set: ', acc_s)
plt_res(lst_cost_s, 'cost', lr)

print('-------100000 training set-------------')
print('Accurcy at 100000 training set: ', acc_m)
plt_res(lst_cost_m, 'cost', lr)

print('-------1000000 training set-------------')
print('Accurcy at 1000000 training set: ', acc_l)
plt_res(lst_cost_l, 'cost', lr)


# ## Discussion:
# We have used three different data set with *10000*, *100000* and *1000000*. For first two dataset we ran loop about *1000* times to update paramters and for *1000000* the loop is ran about 15 times(due to computational cost we consider loops over small number).
# 
# Surperising all three dataset yielded into about same accuracy of *50%* and we were not able to increase it
# 
# 

# ---
