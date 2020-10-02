#!/usr/bin/env python
# coding: utf-8

# # Regression example

# In[29]:


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

# train test split
X_train, Y_train = X[:160], Y[:160]     
X_test, Y_test = X[160:], Y[160:]       


# Build a  neural network from 1st layer to the last layer 

# In[30]:


model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))


# Choose loss function and optimizing method
# 

# In[31]:


model.compile(loss='mse', optimizer='sgd')


# Training

# In[32]:


print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)


# Test

# In[34]:


print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


# Plotting the prediction

# In[35]:


Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

