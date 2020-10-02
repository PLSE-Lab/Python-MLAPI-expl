#!/usr/bin/env python
# coding: utf-8

# ### Goal
# 
# In this kernel I want to exeriment with approximation of different functions using ANN, to get better undersatanding how they work.
# 
# To achieve this goal I will create a function, that will do all work for us:
# 1. As input the function takes a function, that we need to approximate.
# 2. The function creates "x" values using [-10, 10) range and "y" values, using passed as argument function.
# 3. 50% of generated data will be used as training dataset and 50% as test dataset.
# 4. In addition to train/test data, a new "unseen" data in range [10, 20) will be generated to test, how model will handle data, that outside the train data range.
# 5. Train model using train data.
# 6. Make predictions using train, test and unseen data.
# 7. Plot results.
# 
# So, let's start coding:

# In[ ]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylab import meshgrid

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.initializers import he_normal, glorot_normal


# In[ ]:


def approximate_2d(y_func, epochs = 10, batch_size = 4, hidden_layers = [4], test_size = 0.5, init = 'glorot_normal', act = 'sigmoid'):
    # Train/test data
    x = np.arange(-10, 10, 0.1).reshape(-1, 1)
    y = y_func(x).reshape(-1, 1)
    
    # Data to see how model will handle unseen data
    unseen_x = np.arange(10, 20, 0.1).reshape(-1, 1)
    unseen_y = y_func(unseen_x).reshape(-1, 1)
    
    # Train test split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, shuffle = True)
    
    # Scaling data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_x = scaler_x.fit_transform(train_x)
    test_x = scaler_x.transform(test_x)
    unseen_x = scaler_x.transform(unseen_x)
    
    train_y = scaler_y.fit_transform(train_y)
    test_y = scaler_y.transform(test_y)
    unseen_y = scaler_y.transform(unseen_y)
    
    # Model
    if init == 'he_normal':
        init = he_normal(seed = 666)
    elif init == 'glorot_normal':
        init = glorot_normal(seed = 666)
    
    model = Sequential()    
    for i, l in enumerate(hidden_layers):
        model.add(Dense(l, input_shape = (1, ), kernel_initializer = init, activation = act)) if i == 0 else        model.add(Dense(l, kernel_initializer = init, activation = act))
    model.add(Dense(1, kernel_initializer = init))
    model.compile(optimizer = 'sgd', loss = 'mse')
    model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_split = 0.1, verbose = 0)
    
    # Predictions
    preds = scaler_y.inverse_transform(model.predict(test_x)) # Preds on test data
    preds_train = scaler_y.inverse_transform(model.predict(train_x)) # Preds on train data
    unseen_preds = scaler_y.inverse_transform(model.predict(unseen_x)) # Preds on unseen data
    
    # Inverse transform of data
    unseen_x = scaler_x.inverse_transform(unseen_x)
    unseen_y = scaler_y.inverse_transform(unseen_y)
    train_x = scaler_x.inverse_transform(train_x)
    test_x = scaler_x.inverse_transform(test_x)
    
    # Plotting results
    fig = plt.figure(figsize = (19, 6))
    
    # Learning curves plot
    plt.subplot(121)
    H = model.history.history
    plt.plot(H['loss'], label = 'loss')
    plt.plot(H['val_loss'], label = 'val_loss')
    plt.grid(); plt.legend()
    
    # Predictions plot
    plt.subplot(122)
    plt.plot(x, y, label = '$f(x)$')
    plt.scatter(test_x, preds, label = 'Test_preds', s = 15, c = 'g', marker = 'x')
    plt.scatter(train_x, preds_train, label = 'Train_preds', s = 5, c = 'y', alpha = 0.9)
    plt.scatter(unseen_x, unseen_preds, label = 'unseen_preds', s = 10, c = 'k', marker = '1')
    plt.plot(unseen_x, unseen_y, label = 'Unseen data')
    plt.grid(); plt.legend()
    plt.title(f'{len(hidden_layers)} hidden: {hidden_layers} neurons, {act} activation')
    plt.show()   


# ### $x^2+2x+5$ approximation
# 
# I want to start with something simple - quadratic function. So our ANN will have one input and one output. 

# In[ ]:


def quadratic(x):
    return x**2 + 2*x + 5

approximate_2d(quadratic, hidden_layers = [24], epochs = 500) 


# In this experiment I used 1 hidden layer with 24 neurons and sigmoid activation. After 500 epochs, I got a preety good approximation but, at values close to -10 and 10, the approximation starts to look like sinusoid, not a parabola. Let's try relu now:

# In[ ]:


approximate_2d(quadratic, hidden_layers = [10, 20], epochs = 200, act = 'relu', init = 'he_normal') 


# After some experiments, I decided to stop at next configuration: relu as activation, 2 hidden layers - first with 10 neurons and second with 20 neurons. After 200 epochs I got a very good approximation even on unseen data. 

# ### $sin(x)$ approximation
# 
# Next let's try $sin(x)$ function:

# In[ ]:


def sinusoid(x):
    return np.sin(x)

approximate_2d(sinusoid, hidden_layers = [16, 32, 64, 128], epochs = 500, act = 'relu', init = 'he_normal') 


# The $sin(x)$ function took much more layers and neurons to approximate: 4 hidden layers with, 16, 32, 64, 128 neurons. The approximation on train data is very good, but the model cant predict unseen data properly.

# ### $0.2x^2+5sin(5x)+4cos(x)$ approximation
# 
# Now let's try tricky one:

# In[ ]:


# f(x)=0.2x2+0.5\sin(5x)+2\cos(x)
def tricky_one(x):
    return (0.2 * x**2) + (5 * np.sin(5 * x)) + (4 * np.cos(x))

approximate_2d(tricky_one, hidden_layers = [64, 128, 256], epochs = 100, act = 'relu', init = 'he_normal') 


# After many attemts I decided to stop on this variant: relu, 3 hidden layers with 64, 128, 256 neurons. It not follow function directly, but this is still descent approximation.

# ### Approximation of multivariable function
# 
# Now lets take something more interesting - the function with 2 inputs and 1 output:
# 
# $sin(\frac{1}{2}x^2-\frac{1}{4}y^2+3)cos(2x+1-e^y)$
# 
# First - we need to edit our main function a little bit:

# In[ ]:


def approximate_3d(y_func, epochs = 10, batch_size = 4, hidden_layers = [4], test_size = 0.5, init = 'glorot_normal', act = 'sigmoid'):
    # Train/test data
    x = np.arange(-1.5, 1.6, 0.1)
    y = np.arange(-1.5, 1.6, 0.1)
    
    X, Y = meshgrid(x, y)
    Z = f(X, Y)    
    
    train_data = np.array(list(zip(X.flatten(), Y.flatten())))
    labels = Z.flatten().reshape(-1, 1)
    
    # Train test split
    train_x, test_x, train_y, test_y = train_test_split(train_data, labels, test_size = test_size, shuffle = True)
    
    # Scaling data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_x = scaler_x.fit_transform(train_x)
    test_x = scaler_x.transform(test_x)
    
    train_y = scaler_y.fit_transform(train_y)
    test_y = scaler_y.transform(test_y)
    
    # Model
    if init == 'he_normal':
        init = he_normal(seed = 666)
    elif init == 'glorot_normal':
        init = glorot_normal(seed = 666)
    
    model = Sequential()    
    for i, l in enumerate(hidden_layers):
        model.add(Dense(l, input_shape = (2, ), kernel_initializer = init, activation = act)) if i == 0 else        model.add(Dense(l, kernel_initializer = init, activation = act))
    model.add(Dense(1, kernel_initializer = init))
    model.compile(optimizer = 'sgd', loss = 'mse')
    model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_split = 0.1, verbose = 0)
    
    # Predictions
    preds = scaler_y.inverse_transform(model.predict(test_x)) # Preds on test data
    preds_train = scaler_y.inverse_transform(model.predict(train_x)) # Preds on train data
        
    # Inverse transform of data
    train_x = scaler_x.inverse_transform(train_x)
    test_x = scaler_x.inverse_transform(test_x)
    
    t_x = train_x[:, 0]
    t_y = train_x[:, 1]
    
    tr_x = test_x[:, 0]
    tr_y = test_x[:, 1]
    
    # Plotting results
    fig = plt.figure(figsize = (20, 8))
    
    # Learning curves plot
    plt.subplot(121)
    H = model.history.history
    plt.plot(H['loss'], label = 'loss')
    plt.plot(H['val_loss'], label = 'val_loss')
    plt.grid(); plt.legend()
    
    # Predictions plot    
    ax = fig.add_subplot(1, 2, 2, projection = '3d')
    ax.plot_wireframe(X, Y, Z, color = 'k', label = '$f(x)$', alpha = 0.4)
    ax.scatter(t_x, t_y, preds_train, c = 'y', label = 'Train_preds')
    ax.scatter(tr_x, tr_y, preds, c = 'g', label = 'Test_preds')
#     ax.legend()
    ax.view_init(25, 100)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()


# Now we can start experiment. After some attempts, I managed to get a good approximation with next configuration: relu, 3 hidden layers with 64, 128, 256 neurons.

# In[ ]:


def f(x, y):
    return np.sin(1/2 * x**2 - 1/4 * y**2 + 3) * np.cos(2 * x + 1 - np.e**y)

approximate_3d(f, hidden_layers = [64, 128, 256], epochs = 100, act = 'relu', init = 'he_normal')


# As we can see - the ANN can approximate different functions pretty well, but only on a specific range, that is limited by training data range.

# In[ ]:




