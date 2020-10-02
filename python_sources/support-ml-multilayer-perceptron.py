#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

main_path = '/kaggle/input/support-fashion-classification-challenge/'


# In[ ]:


# Multilayer Perceptron (MLP) ---> Fully connected network (FCN)

"""
Define class just as a repository, instead of implementation, 
poor implementation in practice, but good for simplicity, and readability
"""

class MLP:
    
    def __init__(self, lr, n_iter):

        # learning rate
        self.lr = lr
        
        # number of iterations
        self.n_iter = n_iter

        # x = a0
        self.x = None
        self.y = None

        # number of layers
        self.L = None
        
        # number of examples
        self.m = None
        
        # number of features
        self.n = None
        
        # classes
        self.classes = None
        
        # number of classes ---> len(self.classes)
        self.n_classes = None
        
        # weights repository (for the entire network)
        self.w = {}
        
        # bais repository (for the entire network)
        self.b = {}
        
        # layers input ----> a[0] = x
        self.a = {}
        
        # derivative [da_l / dz_l]  (cache for backward propagation)
        self.dz = {}
        
        # (cache for w, b update)
        self.dw = {}
        self.db = {}


# In[ ]:


def initlization(mlp, x, y, neurals_list):
    
    """
    mlp (object): is MLP class object
    
    neurals_list: list contains the number of neurons at each layer
    
    void ---> funtion: returns None, (just update mlp attributes)
    """
    
    # a[0] = x
    mlp.a[0] = x
    
    # number of layers
    mlp.L = len(neurals_list)
    
    # number of examples (row)
    mlp.m = len(x)
    
    # number of features (cols)
    mlp.n = x.shape[1]
    
    # unique classes
    mlp.classes = np.unique(y)
    
    # One Hot Encoding
    mlp.y = (y.reshape(-1, 1) == np.unique(y)).astype('float32')

    # number of class = neurals_list[L-1]    
    mlp.n_classes = len(mlp.classes)
    
    # Model Parameters (weight, bais) initlization,  l --> layer (l)

    """
    Update
    -------
    b_l_shape ----> (neurons, 1) not (neurons, ) or neurons to avoid 
    unexpected behavior due to numpy broadcasting.
    
    range(1, L)
    """ 
    for l in range(1, mlp.L):
        
        ### START
        w_l_shape = None
        b_l_shape = None 
        ### END
        
        # weights (* 0.01 for scaling weights)
        mlp.w[l] = np.random.randn(*w_l_shape) * 0.01
        
        # bais
        mlp.b[l] = np.zeros(b_l_shape)


# # Loss (Cost) Function - (Categorical Cross Entropy) 

# # $Loss =  \frac{1}{m} \sum -Y \ log(A) - (1-Y) \ log(1-A) $

# In[ ]:


def compute_loss(A, Y):
    """
    Y: is true value
    A_l: is a[L - 1], last (output) layer 
    """
    ### START
    loss = None
    ## END
    
    return loss 


# # Activation

# ## Sigmoid
# 
# # $l \rightarrow Layer \ (l),\\\\ l=1, 2, 3, 4, ..., L$
# 
# -----
# # $\sigma(z_l) = \frac{1}{1 + e^{-z_l}}$ 
# 
# # $\frac{d\sigma}{dz_l} = a \ (1 - a)$

# In[ ]:


def sigmoid(z_l): # return a
    pass


# In[ ]:


def softmax(z_L): # return a
    """
    z_L: is z[L-1] ---> output layer
    """
    
    pass


# # Forward Propagation $\equiv$ Hypothesis

# # $l \rightarrow Layer \ (l),\\\\ l=1, 2, 3, 4, ..., L$
# 
# -----
# # $x = a_0$
# # $z_l = w_l \cdot a_{l-1} + b_l$
# # $a_l = \sigma(z_l)$

# In[ ]:


def forward_propagation(mlp):
    
    """
    void ---> funtion: returns None, (just update mlp attributes) 
    
    mlp.a[i]
    """
    
    L = mlp.L
    
    a = mlp.a 
    
    w = mlp.w
    b = mlp.b
    
    ### START
    for l in range(1, L):
        pass
    ### END


# # Backward Propagation

# ## Review: 
# 
# [Composite Function]( https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:composite/x9e81a4f98389efdf:composing/v/function-composition)
# 
# [Chain Rule](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction) 

# # $l \rightarrow Layer \ (l),\\\\ l=1, 2, 3, 4, ..., L$
# 
# -----
# ## Output_Layer
# 
# -------
# # $\frac{d}{dz_L} = dz_L = (a_L - Y)$
# # $\frac{d}{dw_L} =  \frac{1}{m} dz_L^T \cdot a_{l-1}$
# # $\frac{d}{db_L} =  \frac{1}{m} \sum dz_L$
# 
# ------
# 
# ## Intermediate Layers
# 
# -------
# 
# # $\frac{d\sigma}{dz_l} = d_{sigmoid} = da_l = a \ (1 - a)$
# 
# ------
# 
# # $\frac{d}{dz_l} = dz_l = w_{l+1}^T dz_{l+1} \cdot da_l$
# # $\frac{d}{dw_l} = \frac{1}{m} dz_l \cdot a_{l-1}^T$
# # $\frac{d}{db_l} = \frac{1}{m} \sum dz_l$
# 

# In[ ]:


def backward_propagation(mlp):
    
    L = mlp.L
    
    a = mlp.a 
    
    w = mlp.w
    b = mlp.b
    
    dz = mlp.dz
    
    ### START
        
    # (backward_propagation) OUTPUT LAYER
    
    # 
    
   
    for l in reversed(range(1, L-1)):
        pass
    
    
    
    dw = mlp.dw
    db = mlp.db
    
    # Update w, b
    for l in range(1, L-1):
        pass
    
    
    mlp.w = w
    mlp.b = b
    
    ### END


# # Model

# In[ ]:


def fit_mlp(x, y, lr, n_iter, neurals_list: list):
    
    mlp = MLP(lr, n_iter)
    
    initlization(mlp, x, y, neurals_list)
    
    # number of layers
    L = mlp.L
    
    losses = []

    for i in range(n_iter):
        
        forward_propagation(mlp)
        
        A = mlp.a[L - 1]
        
        loss = compute_loss(A, y)
        
        if i % 10 == 0:
            print(f'Loss At {i} = {loss}\n', '=' * 50)
            
        losses.append(loss)
        
        backward_propagation(mlp)
    
    return losses, mlp


# # Prediction

# In[ ]:


def predict(x_test, mlp):
    
    L = mlp.L
    
    # change input layer values
    mlp.a[0] = x_test
    
    # call forward_propagation
    forward_propagation(mlp)
    
    # Get last layer output
    A = mlp.a[L - 1]
    
    # return prediction
    return A.argmax(axis = 1)


# # Debuge Functions

# In[ ]:


def print_shapes(mlp, key):

    types = {'w':mlp.w,
           'b':mlp.b,
           'a':mlp.a,
           'dz':mlp.dz,
           'dw':mlp.dw,
           'db':mlp.db}

    L = mlp.L

    for i in range(1, L):
        print(key, f' shape at layer_{i}:', types[key][i].shape)


# # Test Your Model

# In[ ]:


train = pd.read_csv(main_path + "train.csv").to_numpy()
test = pd.read_csv(main_path + "test.csv")
sample_submission = pd.read_csv(main_path + "sample_submission.csv")

# [:, np.newaxis] = .reshape(-1, 1)
X, Y = train[:, 2:].astype('float32'), train[:, 1][:, np.newaxis].astype('float32')

# normalize/scale input to avoid overflow as possible
X /= 255.0

# Test using  1000 trainning example
X, Y = X[:1000], Y[:1000]

print('Shape :', X.shape)

m, n = X.shape


# In[ ]:


# Create an object for testing process
mlp = MLP(0.001, 10)


# In[ ]:


# test -- initlization

initlization(mlp, X, Y, [n, 784, 800, 10])


# In[ ]:


print_shapes(mlp, 'w')

print()

print_shapes(mlp, 'b')

print()


# In[ ]:


# test -- forward_propagation

forward_propagation(mlp)


# In[ ]:


print_shapes(mlp, 'a')

print()


# In[ ]:


# test -- backward_propagation

backward_propagation(mlp)


# In[ ]:


print_shapes(mlp, 'dz')

print()

print_shapes(mlp, 'dw')

print()

print_shapes(mlp, 'db')

print()


# # Submission

# In[ ]:


test = None
prediction = None


# In[ ]:


def to_csv(prediction):
    pass


# In[ ]:




