#!/usr/bin/env python
# coding: utf-8

# This kernel presents a simple implementation of a 2-layer neural network. It heavily builds on the Machine Learning Mooc from Coursera.
# I know it's possible to do much simpler and faster with sklearn, but I want to post my own implementation as I first wanted to understand the algorithm before using black boxes.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as nm # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import optimize as opti


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


## Defining some functions.

## This is the sigmoid activation function
def sigmoid(z):
    return 1./(1+ nm.exp(-z));

## This is the gradient of the sigmoid function. Used in the backprop
def sigmoidGradient(z):
     return sigmoid(z)*(1-sigmoid(z));

    
## This is the cost function, that return the cost and the gradient of the cost
## as a function of the parameters nn_theta of the neural net. This function id hardcoded
## for two hidden layers

def nnCostFunction2(nn_params, input_layer_size, hidden_layer_size1,hidden_layer_size2, num_labels, X, y, lam):

    ## defining the number of parameters  in each layer
    N_elem_theta1=hidden_layer_size1*(1+input_layer_size)
    N_elem_theta2=(hidden_layer_size1+1) *hidden_layer_size2
    N_elem_theta3=(hidden_layer_size2+1)*num_labels


    Theta1=nn_params[0:N_elem_theta1].reshape((hidden_layer_size1, input_layer_size+1))
    Theta2=nn_params[N_elem_theta1:N_elem_theta1+N_elem_theta2].reshape((hidden_layer_size2, hidden_layer_size1+1))
    Theta3=nn_params[N_elem_theta1+N_elem_theta2:].reshape((num_labels, hidden_layer_size2+1))


    m= X.shape[0]
       
    J = 0;
    Theta1_grad = nm.zeros(Theta1.shape);
    Theta2_grad = nm.zeros(Theta2.shape);
    Theta3_grad = nm.zeros(Theta3.shape);

    YY=nm.zeros((m, num_labels));
    for i in range(1, num_labels):
        YY[nm.where(y==i)[0], i-1]=1
        
    YY[nm.where(y==0), -1]=1


    a1=nm.append(nm.ones((m,1)), X, axis=1)
    z2=nm.dot(Theta1,a1.T)
    a2=sigmoid(z2).T;
    a2=nm.append(nm.ones((m,1)), a2, axis=1);
    z3=nm.dot(Theta2, a2.T)
    a3=sigmoid(z3).T;
    a3=nm.append(nm.ones((m,1)), a3, axis=1);
    z4=sigmoid(nm.dot(Theta3, a3.T));

   
    #J=nm.sum( -YY   *  nm.log(z3.T)    - (1-YY)* nm.log(1-z3.T)) /m
    J=nm.sum(   nm.log( (z4.T)**(-YY) * (1-z4.T)**(-(1-YY))    ))/m
    #print  J, sum(YY), 1-nm.max(z3), nm.min(z3)
    ## adding regul
    J+=lam/(2.*m)    *(nm.sum(Theta1[:,1:]**2) + nm.sum(Theta2[:,1:]**2)+nm.sum(Theta3[:,1:]**2) )


    #### using backprop to compute the gradient.

    Delta_1ij=nm.zeros(Theta1.shape);
    Delta_2ij=nm.zeros(Theta2.shape);
    Delta_3ij=nm.zeros(Theta3.shape);


    delta4 = z4.T-YY;


#    delta3 = z3.T-YY;
    
    ##3compute delta2
    delta3=(nm.dot(delta4, Theta3)[:, 1:] * sigmoidGradient(z3).T)
    delta2=(nm.dot(delta3, Theta2)[:, 1:] * sigmoidGradient(z2).T)
    ##4 accumulate
    
    
    
    Delta_1ij=nm.dot(delta2.T,a1)/m;
    Delta_2ij=nm.dot(delta3.T, a2)/m;
    Delta_3ij=nm.dot(delta4.T, a3)/m;

    Theta1_grad+=Delta_1ij;
    Theta2_grad+=Delta_2ij;
    Theta3_grad+=Delta_3ij;
    
    Theta1_grad[:,1:]+=lam/m * Theta1[:,1:];
    Theta2_grad[:,1:]+=lam/m * Theta2[:,1:];
    Theta3_grad[:,1:]+=lam/m * Theta3[:,1:];
    
    
    grad=nm.append(nm.append(Theta1_grad.flatten(), Theta2_grad.flatten()), Theta3_grad.flatten())

    return J, grad

def predict2(Theta1, Theta2, Theta3,X):

    m = X.shape[0];
    num_labels = Theta3.shape[0];
    p = nm.zeros((X.shape[0], 1));
    X=nm.append(nm.ones((m,1)), X, axis=1)
    h1 = sigmoid(nm.dot(X , Theta1.T));
    h1= nm.append(nm.ones((m,1)), h1, axis=1)
    h2 = sigmoid(nm.dot(h1 , Theta2.T));
    h2=nm.append(nm.ones((m,1)), h2, axis=1)
    h3= sigmoid(nm.dot(h2 , Theta3.T));
    p=nm.argmax(h3, axis=1)+1
    p[p==10]=0
    return p

def train2(X, y, hidden_layer_size1,hidden_layer_size2, lam):
    input_layer_size =X.shape[1]
    num_labels=10

    m=X.shape[0]
    print('there are {} training samples here'.format(m))


    eps=0.12
    ## initialing params.
    initial_Theta1 =     nm.random.rand(hidden_layer_size1, 1+input_layer_size) *2*eps -eps   
    initial_Theta2 =   nm.random.rand(hidden_layer_size2,   1+ hidden_layer_size1) *2*eps -eps 
    initial_Theta3 =   nm.random.rand(num_labels,   1+ hidden_layer_size2) *2*eps -eps 


    initial_nn_params=nm.append(nm.append(initial_Theta1.flatten(), initial_Theta2.flatten()), initial_Theta3.flatten() )
    
    ## let's compute the cost function and its gradient with the initial_Theta's
    
    #nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam)
    nnCostFunction2(initial_nn_params, input_layer_size, hidden_layer_size1,hidden_layer_size2, num_labels, X, y, lam)

    
    #res2=opti.minimize(nnCostFunction, initial_nn_params, jac=True, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lam), method="L-BFGS-B", tol=1e-6,  options={'disp':True, 'maxiter':300, 'gtol':1e-6})
    res2=opti.minimize(nnCostFunction2, initial_nn_params, jac=True, args=(input_layer_size, hidden_layer_size1,hidden_layer_size2, num_labels, X, y, lam), method="L-BFGS-B", tol=1e-6,  options={'disp':False, 'maxiter':500, 'gtol':1e-6})
    
    N_elem_theta1=hidden_layer_size1*(1+input_layer_size)
    N_elem_theta2=(hidden_layer_size1+1) * hidden_layer_size2
    N_elem_theta3=(hidden_layer_size2+1) * num_labels


    Theta1=(res2.x)[0:N_elem_theta1].reshape((hidden_layer_size1, input_layer_size+1))
    Theta2=(res2.x)[N_elem_theta1:N_elem_theta1+N_elem_theta2].reshape((hidden_layer_size2, hidden_layer_size1+1))
    Theta3=(res2.x)[N_elem_theta1+N_elem_theta2:].reshape(( num_labels, hidden_layer_size2+1))

    return Theta1, Theta2, Theta3


# In[ ]:


## Main script
# Loading the dataset
A=pd.read_csv('../input/train.csv').values
B=pd.read_csv('../input/test.csv').values

## defining the target and the feature matrix.
X=(A[:, 1:]-128)/128.
B=(B[:,:]-128)/128.
y=nm.int32(A[:,0])

N=42000
X_train=X[:N,:]
y_train=y[:N]

X_val=X[N:,:]
y_val=y[N:]

## Plot one digit for illustration
id=244
plt.figure(1)
plt.clf()
plt.imshow(X[id, :].reshape(28, 28), cmap="Greys")
plt.title('True value = {}'.format(y[id]))


# In[ ]:


lam=0.2

## The next line launches the training, but this home-made implementation takes too much time. 
## Hence the line is commented.

##T1, T2,T3= train2(X_train, y_train, 15,7, lam)

