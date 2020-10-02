#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv(dirname + "/train.csv")


# ### Analyze Data

# In[ ]:


df.head()


# To make a quick neural network using the data above,<br>
# we can easily create a neural network using the following the columns:<br>
# '**Age**', '**Sex**', '**Fare**', '**Pclass**', '**SibSp**', '**Parch**'

# In[ ]:


# lets take out first the label
train_y = df['Survived']
train_y.head()


# In[ ]:


# function to filter the age, sex, fare pclass, sibsp, parch columns
def get_data(data):
    # take only this specific column
    data = data[['Age', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Parch']]
    
    # replace male by 1, female by 0
    data.replace({ 'male' : 1, 'female' : 0 }, inplace=True)
    
    # replace null/nan data by the mean (age and fare columns)
    data['Fare'].fillna(int(data['Fare'].mean()), inplace=True)
    data['Age'].fillna(int(data['Age'].mean()), inplace=True)
    
    # transform into a numpy array
    data = data.to_numpy().astype(float)
    
    # normalize (make sure the data is between -1 and 1)
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
    
    return data


# In[ ]:


train_x = get_data(df)
print(train_x)


# In[ ]:


print(train_x.shape)


# Shape will show us the number of rows and columns (891 and 6)

# In[ ]:


# same for the labels (contains 0 - 1 if the victim survived or not)
print(train_y.shape)


# ### Neural Network

# In[ ]:


# the activation function and derivative of the action function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# In[ ]:


# the loss function and its derivative
def loss_fn(y, y_hat):
    return 1/2 * (y - y_hat) ** 2

def dloss_fn(y, y_hat):
    return (y - y_hat)


# In[ ]:


# number of rows
instances = train_x.shape[0]

# number oof columns
attributes = train_x.shape[1]

# number of hidden node for first layer 
hidden_nodes = 8

# number of hidden node for second layer
hidden_nodes_two = 4

# number of output labels 
output_labels = 1


# In[ ]:


# Inititate the weights/biases
w1 = np.random.rand(attributes,hidden_nodes)
b1 = np.random.randn(hidden_nodes)

w2 = np.random.rand(hidden_nodes,hidden_nodes_two)
b2 = np.random.randn(hidden_nodes_two)

w3 = np.random.rand(hidden_nodes_two, output_labels)
b3 = np.random.randn(output_labels)

theta = w1, w2, w3, b1, b2, b3


# In[ ]:


# Neural Network Forward
def forward(x, theta):
    w1, w2, w3, b1, b2, b3 = theta
    
    k = np.dot(x, w1) + b1
    l = sigmoid(k)
    
    m = np.dot(l, w2) + b2
    n = sigmoid(m)
    
    o = np.dot(n, w3) + b3
    p = sigmoid(o)
    
    return k, l, m, n, o, p


# In[ ]:


# Neural Network Backward
def backward(x, y, sigma, theta):
    k, l, m, n, o, p = sigma
    w1, w2, w3, b1, b2, b3 = theta
    
    # db3 = dloss * dsigm(o) * 1
    # dw3 = dloss * dsigm(o) * n
    
    # db2 = dloss * dsigm(o) * w3 * dsigm(m) * 1
    # dw2 = dloss * dsigm(o) * w3 * dsigm(m) * l
    
    # db1 = dloss * dsigm(o) * w3 * dsigm(m) * w2 * dsigm(k) 
    # dw1 = dloss * dsigm(o) * w3 * dsigm(m) * w2 * dsigm(k) * x
    
    dloss = dloss_fn(p, y)
    dsigm_p = dsigmoid(o)
    dsigm_n = dsigmoid(m)
    dsigm_l = dsigmoid(k)
    
    db3 = dloss * dsigm_p
    dw3 = np.dot(n.T, db3)
    
    db2 = np.dot(db3, w3.T) * dsigm_n
    dw2 = np.dot(l.T, db2)
    
    db1 = np.dot(db2, w2.T) * dsigm_l
    dw1 = np.dot(x.T, db1)
    
    return dw1, dw2, dw3, db1, db2, db3


# In[ ]:


# use the avg of the gradients for the derivative of each bias
def avg_bias(grads):
    dw1, dw2, dw3, db1, db2, db3 = grads
    db1 = db1.mean(axis=0)
    db2 = db2.mean(axis=0)
    db3 = db3.mean(axis=0)
    return dw1, dw2, dw3, db1, db2, db3


# In[ ]:


# Use the SGD in order to optimize the weights and biases
def optimize(theta, grads, lr=0.001):
    dw1, dw2, dw3, db1, db2, db3 = grads
    w1, w2, w3, b1, b2, b3 = theta
    
    w1 -= dw1 * lr
    w2 -= dw2 * lr
    w3 -= dw3 * lr
    b1 -= db1 * lr
    b2 -= db2 * lr
    b3 -= db3 * lr
    
    return w1, w2, w3, b1, b2, b3


# In[ ]:


# return 1 if the prediction is higher than 0.5
# return 0 if not
def predict(x, theta):
    predict = forward(x, theta)[-1]
    return np.where(predict > 0.5, 1, 0)


# In[ ]:


# time to train our model
for epoch in range(1000):
    
    sigma = forward(train_x, theta)
    grads = backward(train_x, train_y, sigma, theta)
    theta = optimize(theta, avg_bias(grads))
    
    if(epoch % 100 == 0):
        print(loss_fn(sigma[-1], train_y).mean())


# ### Time to train the test data

# In[ ]:


test_df = pd.read_csv(dirname + "/test.csv")
test_x = get_data(test_df)


# In[ ]:


# Get test data predictions
test_preds = predict(test_x, theta)


# In[ ]:


# Add passengers ids to the test predictions
passenger_ids = test_df['PassengerId'].to_numpy()


# In[ ]:


# combine passenger ids with the predictions
final_result = np.array(list(map(list, zip(passenger_ids, test_preds))))


# In[ ]:


# arraay final_result to dataframe
df_final = pd.DataFrame(data=final_result, columns=["PassengerId", "Survived"])

# save the result
df_final.to_csv('submission.csv', index=False)

