#!/usr/bin/env python
# coding: utf-8

# **This code is a Sample for Using Convolutions with Tensorflow**

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
import matplotlib.pyplot as plt
import tensorflow as tf
# Any results you write to the current directory are saved as output.


# In[ ]:


X_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


Y_train = X_train['label']
X_train.drop('label',axis=1,inplace=True)
X_train.columns


# In[ ]:


Y_train.shape


# let's convert each row into 28*28
# 

# In[ ]:


m,n = X_train.shape
X_train = X_train.values.reshape(m,28,28)


# In[ ]:


m1,n1 = X_test.shape
X_test = X_test.values.reshape(m1,28,28)
X_test.shape


# In[ ]:


index = 4
plt.imshow(X_train[index][:])
plt.imshow(X_test[index][:])


# Let's change the X_train and X_test back to their original shape
# 

# In[ ]:


X_train = np.float32(X_train[:][:] / 255)
X_test = np.float32(X_test[:][:] / 255)


# In[ ]:


C=max(Y_train.unique()) +1
C


# In[ ]:


temp = np.zeros((m,C))
print(temp.shape)
def oneHot(Y_train):
    for i in range(0,m):
        #print(i,Y_train[i])
        temp[i][Y_train[i]]=  1
    return temp


# In[ ]:


Y_train= oneHot(Y_train)
Y_train.shape


# In[ ]:


X_train = X_train.reshape(m,28,28,1)


# ![image.png](attachment:image.png)
# 
# We will use the below model of Convolutions here as it has worked welll for others

# z1 = tf.nn.conv2d(X_train,W1,strides=[1,1,1,1]) #W1 = [5,5,1,6]   (n+2p-f/s) + 1 = 24*24*6
# A1 = tf.nn.relu(z1)
# 
# P1 = tf.nn.max_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
# 
# z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1]) #W2 = [5,5,1,16]  (n+2p-f/s) + 1 = 18*18*16
# 
# A2 = tf.nn.relu(z2) 
# 
# P2 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
# 
# 
# F1 = tf.contrib.layers.fully_connected(P2,num_outputs=5184,activation_fn=None)
# 
# F2 = tf.contrib.layers.fully_connected(F1,num_outputs=84,activation_fn=None)
# 
# 

# In[ ]:


n1 = X_train.shape[1]
p1=0
s1=1
f1=5
fsize1 = 6
n1_out = ((n1 + 2*p1 -f1)/s1) + 1

print("Conv-1 out size is :",n1_out)
pool1_n =  n1_out
pool1_p = 0
pool1_s = 2
pool1_f= 2
pool1_out = (n1_out + 2*pool1_p - pool1_f)/pool1_s + 1
print("Pool-1 out size is :",pool1_out)

n2 = pool1_out
p2=0
s2=1
f2=5
fsize2 = 12
n2_out = ((pool1_out + 2*p2 -f2)/s2) + 1
print("Conv-2 out size is :",n2_out)

pool2_n =  n2_out
pool2_p = 0
pool2_s = 2
pool2_f= 2
pool2_out = ((n2_out + 2*pool2_p -pool2_f)/pool2_s) + 1
print("Pool-2 out size is :",pool2_out)

w3_output = (np.int16)(pool2_out*pool2_out*fsize2)
w4_output = 84

print("FC1 out size is :",w3_output)
print("FC-2 out size is :",w4_output)


# In[ ]:


def create_placeholders():
    X = tf.placeholder(dtype=tf.float32,name='X',shape=(None,28,28,1))
    Y = tf.placeholder(dtype=tf.float32,name='Y',shape=(None,C))

    return X,Y

#Lets create a 2 layer NN with each layer having 300 and 100 neurons resp.
def init_params():

    Xintialzer = tf.contrib.layers.xavier_initializer()
    W1 = tf.Variable(Xintialzer(shape=(5,5,1,5)))
    W2 = tf.Variable(Xintialzer(shape=(5,5,5,15)))


    parameters = {
                    "W1": W1,
                    "W2": W2
                }
    return parameters

def fwd_move(X,parameters):

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    print("W1 shape is ",W1.shape)
    print("W2 shape is ",W2.shape)

    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED -> FULLYCONNECTED -> SOFTMAX
    """
    
    z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="VALID") #W1 = [5,5,1,5]   (n+2p-f/s) + 1 = 24*24*6
    print("Z1 shape is ",z1.shape)
    A1 = tf.nn.relu(z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("P1 shape is ",P1.shape)

    print("W2 shape is ",W2.shape)
    z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="VALID") #W2 = [5,5,5,15]  (n+2p-f/s) + 1 = 8*8*15
    print("Z2 shape is ",z2.shape)
    A2 = tf.nn.relu(z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("P2 shape is ",P2.shape)

    P2_flatten = tf.contrib.layers.flatten(P2)
    print("P2_flatten shape is ",P2_flatten.shape)

    F1 = tf.contrib.layers.fully_connected(P2_flatten,num_outputs=100,activation_fn=None)
    print("F1 shape is ",F1.shape)

    F2 = tf.contrib.layers.fully_connected(F1,num_outputs=10,activation_fn=None)
    print("F2 shape is ",F2.shape)

    return F2

def softmax_cost(Z,Y):
    print(Z.shape)
    print(Y.shape)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=Y))
    return cost


# In[ ]:


print(Y_train.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

Xtr, Xte,Ytr,Yte = train_test_split(X_train,Y_train,test_size=0.4,random_state=2)


# In[ ]:


def get_minibatch(X1, Y1, minibatch_size):
    m = X1.shape[0]
    num_batch = (np.int16)(m/minibatch_size)

    start = 0
    minibatch = []
    for i in range(0,num_batch):
        end = start+minibatch_size
        X_mini = X1[start:end]
        Y_mini = Y1[start:end]
        minibatch.append((X_mini,Y_mini))
        start = end

        X_mini = X1[start:m]
        Y_mini = Y1[start:m]
        minibatch.append((X_mini,Y_mini))

        return minibatch


# In[ ]:


def model(Xtr,Ytr,Xte,learning_rate=0.001,batch_size=64,num_epochs=100):    
    m = X_train.shape[0]
    X,Y = create_placeholders()
    parameters = init_params()
    Z = fwd_move(X,parameters)
    cost = softmax_cost(Z,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    num_batches = (np.int16)(m/batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        minibatches = get_minibatch(Xtr,Ytr,batch_size)
        for epoch in range(num_epochs):
            #print("Epoch Number is ",epoch)
            minibatch_cost = 0
            curr_minibatch_cost = 0
            for minibatch in minibatches:
                (X_mini,Y_mini) = minibatch
                _,mini_batch_cost = sess.run([optimizer,cost],feed_dict={X:X_mini,Y:Y_mini})
                #print(mini_batch_cost)
                curr_minibatch_cost = curr_minibatch_cost + mini_batch_cost
            epoch_cost = curr_minibatch_cost/batch_size

            if(epoch % 50 == 0):
                print("epoch cost is :",epoch_cost)
            else:
                epoch_cost=0


# In[ ]:


parameters = model(Xtr,Ytr,X_test,learning_rate=0.001,batch_size=128,num_epochs=100)


# In[ ]:




