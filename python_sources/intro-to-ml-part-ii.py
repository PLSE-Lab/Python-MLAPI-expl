#!/usr/bin/env python
# coding: utf-8

# # Part II - Using Keras and Tensorflow

# # Collect data
# 
# Use data that is linear but with gaussian noise added.

# In[ ]:


import numpy

x = numpy.linspace(0, 5, 100).reshape(-1, 1)

numpy.random.seed(7)

noise = numpy.random.normal(0, 0.1, x.size).reshape(x.shape)

y = 1.5 * x + 2.7 + noise


# # Visualize the data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as pyplot

fig1 = pyplot.figure()
axes1 = pyplot.axes(title='Vizualization of the data')
scatter1 = axes1.scatter(x, y)


# # Split the data into 60% training data and 40% test data

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)


# In[ ]:


fig2 = pyplot.figure()
axes2 = pyplot.axes(title='Split data')
scatter2_train = axes2.scatter(x_train, y_train, label='training data')
scatter2_test = axes2.scatter(x_test, y_test, label='test data')
legend2 = fig2.legend()


# # Choose a model
# Import keras functions

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from keras.models import Sequential\nfrom keras.layers import Dense')


# Define the neural network in keras.  In this case it is just a single neuron with a linear activation function.

# In[ ]:


model = Sequential()
layer1 = Dense(units=1, input_dim=1, activation='linear')
model.add(layer1)


# Compile the model into a neural network in tensorflow.  Use a "mean squared error" loss function, use the stochastic gradient descent optimizer, and keep track of the mean absolute error metric.

# In[ ]:


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])


# # Train the model

# In[ ]:


history = model.fit(x_train, y_train, epochs=500, verbose=False)


# In[ ]:


yp = model.predict(x)
axes1.plot(x, yp, color='cyan', label='fit')
axes1.legend()
fig1


# # Check the accuracy

# In[ ]:


train_loss_and_metrics = model.evaluate(x_train, y_train)
test_loss_and_metrics = model.evaluate(x_test, y_test)


# In[ ]:


print(train_loss_and_metrics)
print(test_loss_and_metrics)


# Check that w and b in the trained network are approximately 1.5 and 2.7.

# In[ ]:


print(layer1.get_weights())


# In[ ]:




