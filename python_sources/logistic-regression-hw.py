#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Preliminary
# In this kernel i aimed to implement logistic regression into a cancer dataset.First of i'll try to make it by myself and then i'll compare the accuracy with sklearn library.

# In[ ]:


data = pd.read_csv("../input/data.csv")


# # A quick review to our dataset to see what's included.

# In[ ]:


data.head()


#  We have 2 features that we don't need it such as 'id' and 'Unnamed:32'.We'll just drop it and prepare our dataset .

# In[ ]:


data.drop(["Unnamed: 32","id"],axis=1,inplace = True)


# In[ ]:


data.head()


# Therefore we are using logistic regression it has to include binary output like '0' or '1' . In this part we'll simply convert "Malign" into "1" and "Benign" into "0"

# In[ ]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]


# In[ ]:


print(data.info())


# In[ ]:


data.head()


# And now we finally prepared our data to perform logistic regression . We can dive into implementing part.

# In[ ]:


y_raw = data.diagnosis.values 


# In[ ]:


# We should make inplace "false" otherwise Python will consider as an error. 
x_raw = data.drop(["diagnosis"] , axis=1 ,inplace = False) 


# In[ ]:


x_normalized = (x_raw - np.min(x_raw)/np.max(x_raw) - np.min(x_raw)).values


# # Train , test split
# 
# *  As we have our inputs and output we need to split those dataframes as train and test frames. Common usage is 80% train and 20% test.
# 
# * We'll simply use sci-kit learn library.
# 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_normalized,y_raw,test_size = 0.2 , random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# *We Transposed the x and y matrixes because our features for instance in x matrix are the columns but we need it as rows that's why we convert it.
# For y matrix it's actually not so important cause we are considering it as an array so it's not really differs when we transposing it*

# # Initializing Parameters And Defining The Activation Function
# * We will consider weights and bias as a parameter.Also we need a probabilistic value as an output.Hereby we will use sigmoid function as our activation function.

# In[ ]:


def weights_and_bias(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


# # Defining Forward And Backward Propagation Function

# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    # **forward propagation**
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -(1-y_train)*np.log(1-y_head) - y_train*np.log(y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    # ***********************
    
    # **backward propagation**
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]      
    gradients = {"derivative_weight" : derivative_weight , "derivative_bias" : derivative_bias}
    # ***********************
    
    return cost,gradients


# **As we returned cost and gradients before now we can move on to the updating parameters part.
# We'll update weights and bias depends on the cost function.**

# In[ ]:


def update(w,b,x_train,y_train,learning_rate,num_of_iterations):
    cost_list = []
    cost_list_print = []
    index = []
    
    for i in range (num_of_iterations):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        if (i%10 == 0):
            cost_list_print.append(cost)
            index.append(i)
            print("Cost after {} iteration : {}".format(i,cost)) 
        
    
    parameters = {"weight" : w , "bias" : b}
    plt.plot(index,cost_list_print)
    plt.xticks(index,rotation = 'vertical')
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters,gradients,cost_list
        


# **And now time to make our prediction we are almost at the end and then we'll just call our functions and finishing the logistic regression part.**

# In[ ]:


def predict(w,b,x_test):
    #In this case we will consider x_test as an input for forward propagation.
    
    z = sigmoid(np.dot(w.T,x_test) + b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range (z.shape[1]):
        if (z[0,i] <= 0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction        
        
        
    


# # Logistic Regression Function

# In[ ]:


def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,num_of_iterations):
    #We will define a dimension.
    dimension = x_train.shape[0]
    w,b = weights_and_bias(dimension)
    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,num_of_iterations)
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    print("Test accuracy is {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# # Implementing Logistic Regression

# In[ ]:


logistic_regression(x_train,x_test,y_train,y_test,learning_rate = 3,num_of_iterations = 600)


# # Sci-kit Learn

# **Now we will use sklearn library to see the difference between our code and library.**

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("Test accuracy is {}".format(lr.score(x_test.T,y_test.T)))


# # Conclusion

# **After sklearn library we found 97.36% accuracy instead of 89.47% . On the other case we were searching learning rate and number of iterations by manually trying but sci-kit learn gives us an advantage that it's search and handle everything by itself.**

# **Special thanks for @dataiteam for further detail you can visit his page.**
