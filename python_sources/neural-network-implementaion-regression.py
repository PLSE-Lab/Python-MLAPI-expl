#!/usr/bin/env python
# coding: utf-8

# **Title: Implementaion of Neural Network algorithms from scratch (Backpropagation)!**
# 
# Universal approximation theorem: **A 2-layer network** with linear outputs (regression) can uniformly approximate a**ny continoues functions** on a compact input domain to arbitary accuracy
# provided the network has **a large number of units**.
# 
# So, I am illustrating this therom with NN regression implementaion without using library. I hope this implementaion gives some of you ideas about under the hood of Neural Network (shallow).
# 
# * Generate synthetic data with guassian distributed noise
# * Impelemnt NN with feed-forward propagation and backpropagation
# * Predict with forward propagation 
# * Test shallow NN with different number of hidden units
# 
# **Architecture of network**:  2 layers, fully connceted, hidden layer units(3 and 30)
# , acctivation function for hidden layer:hyperbolic tangent; activation function for output layer: identity(regression),
# cost function: MSE, optimizer: batch gradient decsent with momentum.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#synthetic data with noise
np.random.seed(10)

X = 2*np.random.rand(1, 50) - 1
T = np.sin(2*np.pi*X) + 0.3*np.random.randn(1, 50)
N = np.size(T,1)
plt.scatter(X, T)
plt.show()


# In[ ]:


# NN implementaion with feed-forward propagation and backprojection 
def Neural_Network_Simple(LR,beta,max_ite,input_nodes,hidden_nodes,output_nodes):

    # weight initialization function
    W_1 = np.random.randn(hidden_nodes, input_nodes)
    W_2 = np.random.randn(output_nodes, hidden_nodes)
    B_1 = np.zeros((hidden_nodes, 1))
    B_2 = np.zeros((output_nodes, 1))

    # gradient descent with momentum
    Vdw_1 = np.random.randn(hidden_nodes, input_nodes)
    Vdw_2 = np.random.randn(output_nodes, hidden_nodes)
    Vdb_1 = np.zeros((hidden_nodes, 1))
    Vdb_2 = np.zeros((output_nodes, 1))

    # cost initialization
    Cost = np.zeros((max_ite,1))

    for i in range(max_ite):
        #feed-forward propagation
        A_1 = W_1.dot(X) + np.tile(B_1, (1, N))
        Z_1 = (np.exp(A_1) - np.exp(-A_1)) / (np.exp(A_1) + np.exp(-A_1))

        A_2 = W_2.dot(Z_1) + np.tile(B_2, (1, N))
        Z_2 = A_2
        
        #back propagation
        del_2 = Z_2 - T
        del_1 = W_2.T.dot(del_2) * (1 - Z_1 ** 2)
        
        #gradient
        dw_2 = del_2.dot(Z_1.T)
        dw_1 = del_1.dot(X.T)
        db_2 = np.sum(del_2, 1)
        db_1 = np.sum(del_1, 1).reshape(hidden_nodes,1)
        
        #GD with momentum
        Vdw_2 = beta * Vdw_2 + (1 - beta) * dw_2
        Vdw_1 = beta * Vdw_1 + (1 - beta) * dw_1
        Vdb_2 = beta * Vdb_2 + (1 - beta) * db_2
        Vdb_1 = beta * Vdb_1 + (1 - beta) * db_1
        
        #update weight and bias with batch GD
        W_2 = W_2 - LR * Vdw_2
        W_1 = W_1 - LR * Vdw_1
        B_2 = B_2 - LR * Vdb_2
        B_1 = B_1 - LR * Vdb_1

        Cost[i] = 0.5 * np.sum(del_2**2)/N
    return W_1,W_2,B_1,B_2,Cost


# In[ ]:


# prediction with forward propagation
def forwardNN_reg(W_1,W_2,B_1,B_2,X):
    A_1 = W_1.dot(X) + np.tile(B_1, (1, 1))
    Z_1 = (np.exp(A_1) - np.exp(-A_1)) / (np.exp(A_1) + np.exp(-A_1))

    A_2 = W_2.dot(Z_1) + np.tile(B_2, (1, 1))
    pred = A_2
    return pred


# In[ ]:


W_1,W_2,B_1,B_2,Cost = Neural_Network_Simple(0.01,0.8,5000,1,3,1)

x_pre =np.linspace(-1,1,100).reshape(1,100)
y_pre =forwardNN_reg(W_1,W_2,B_1,B_2,x_pre)

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.scatter(X, T,label = 'training data')
plt.plot(np.transpose(x_pre),np.transpose(y_pre),'r',label = 'NN prediction')
plt.xlabel('x');plt.ylabel('y')
plt.title('Number of hidden layers=' + str(3) + '; with MSE=' + str(Cost[-1]))
plt.legend()

plt.subplot(1,2,2)
plt.scatter(np.linspace(0,4999,5000),Cost)
plt.xlabel('Number of iterations');plt.ylabel('MSE')
plt.title('Cost function convergence plot')
plt.show()


# In[ ]:


W_1,W_2,B_1,B_2,Cost = Neural_Network_Simple(0.01,0.8,5000,1,30,1)

x_pre =np.linspace(-1,1,100).reshape(1,100)
y_pre =forwardNN_reg(W_1,W_2,B_1,B_2,x_pre)

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.scatter(X, T,label = 'training data')
plt.plot(np.transpose(x_pre),np.transpose(y_pre),'r',label = 'NN prediction')
plt.xlabel('x');plt.ylabel('y')
plt.title('Number of hidden layers=' + str(30) + '; with MSE=' + str(Cost[-1]))
plt.legend()

plt.subplot(1,2,2)
plt.scatter(np.linspace(0,4999,5000),Cost)
plt.xlabel('Number of iterations');plt.ylabel('MSE')
plt.title('Cost function convergence plot')
plt.show()


# **Takeaway** : Even we dont have deep NN, just by adding more units to hidden layers our predictions perform better. Mean squared error for 30 hidden layers is lower than 3 hideen layers.
# 
# **I also added Keras implementation of this same problem below, so it will help you connet the dots better when write your Keras library.**

# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers


# Below section is how I implemented in my code above, I only used one fully connected hidden layer which connceted to the 1-D ouput layer. 
# units: number of hidden units (30 for hidden layer and 1 for output layer)

# In[ ]:


model_NN = keras.models.Sequential()
model_NN.add(layers.Dense(units=30,activation='tanh',input_dim=1))
model_NN.add(layers.Dense(units=1))
optimiz = keras.optimizers.SGD(lr=0.2, momentum=0.8, decay=0.0, nesterov=False)

model_NN.compile(loss="mean_squared_error",optimizer=optimiz,metrics=['mean_absolute_error', 'mean_squared_error'])
history = model_NN.fit(X.T,T.T,batch_size=50,epochs=1500)


# You can also extract whole training process from** history** class.
# Ass you can see, MSE is decerasing gradually.

# In[ ]:


import pandas as pd
hist = pd.DataFrame(history.history)
hist.head(10)


# In[ ]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.scatter(X, T,label = 'training data')
plt.plot(np.transpose(x_pre),model_NN.predict(np.transpose(x_pre)),'r',label = 'NN prediction from Keras')
plt.xlabel('x');plt.ylabel('y')
plt.title('Number of hidden layers=' + str(30) + '; with MSE=' + str(hist['mean_squared_error'].iloc[-1]))
plt.legend()

plt.subplot(1,2,2)
plt.scatter(np.linspace(0,1499,1500),hist['mean_squared_error'])
plt.xlabel('Number of iterations');plt.ylabel('MSE')
plt.title('Cost function convergence plot from Keras')
plt.show()


# **Note:** Result from Keras and my implementation are really close. But due to different optimization and max iteration (I think in Kaggle kernerl you cannot do more than 2000 iterations) , the final MSE value and error convergence is different.
