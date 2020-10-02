#!/usr/bin/env python
# coding: utf-8

# # Comparing Neural Network to Linear Regression for basic unit conversion from Farenheit to Celsius
# 
# Whilst learning on an online course about Keras, we looked at how to make a celsius to fahrenheit converting model, to start to understand how neural networks work.
# 
# Out of curiosity I created the following two models converting fahrenheit back into celsius, and annotated the notebook for learning purposes.
# 
# Feel free to take a look!

# ## Imports
# First we import the required technologies.

# In[ ]:


import keras # For our neural network

import numpy as np # To create the data

from sklearn.model_selection import train_test_split # For dividing our data
from sklearn.metrics import mean_squared_error # For evaluating our models

import matplotlib.pyplot as plt # To briefly vizualise some key points


# ## Create the basic Keras model

# In[ ]:


l0 = keras.layers.Dense(units=1, input_shape=[1])

model = keras.models.Sequential([l0])

model.compile(loss='mean_squared_error', optimizer='Adam')


# We create a simple sequential model with 1 node, as this provides sufficient complexity to model our data using the equation below.
# 
# Celsius = Fahrenheit * w1 + b1
# 
# We will train the model to figure out the ideal weight and bias variables.

# ## Fabricate the training & testing data

# In[ ]:


def to_celsius(f):
    # Applies the known equation for converting fahrenheit to celsius
    return ((f - 32) / 1.8)

# Create list of temperatures from 0 to 199 fahrenheit, and corresponding values in degrees celsius
X = np.arange(200)
y = []

for x in X:
    y.append(to_celsius(x))
    
y = np.array(y)

# Split the data in training and testing sets, for later evaluation of our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# ## Train the model

# In[ ]:


history = model.fit(X_train, y_train, epochs=5000, verbose=False) # Runs for a couple minutes


# ## Evaluating predictions

# In[ ]:


pred = model.predict(X_test)
pred_head = list(map((lambda x: x[0]), pred[:5].tolist()))

print('RMSE, Root mean squared error:', np.sqrt(mean_squared_error(y_test, pred)))

print('\nPrediction: {}\nActual: {}'.format(
    pred_head,
    y_test[:5])
)


# As our RMSE is essentially 0 you can see we are getting almost exactly correct answers, to 2 decimal places.

# In[ ]:


plt.xlabel('Actual value in Celsius')
plt.ylabel('Model`s value in Celsius')
plt.plot(y_test[:5], pred_head)
plt.show()


# After plotting the data it is clear that our model converts our temperatures pretty accurately. However it did take a minute or so to train.

# In[ ]:


plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(X,y)
plt.show()


# As you can see here, the conversion from fahrenheit to celsius is described by a simple linear function. Hence we will see later that LinearRegression just as well, and fits to the data almost instantly.

# ## Explaining how it works

# In[ ]:


print('Weights: {}'.format(l0.get_weights()))


# *The equation we are trying to model vs. the equation used by our neural network:*
# 
# Actual: y = (x - 32) / 1.8
# 
# Model: y = x * w1 - b1
#            
#           (x * 0.55554414) - 17.775848

# In[ ]:


print(10 * 0.55554414 + (-17.775848))

print(10 * l0.get_weights()[0] + l0.get_weights()[1])

print(model.predict(np.array([10])))


# * Line 1 uses the given weight & bias values
# 
# * Line 2 uses the exact weight & bias values, differing to line 1
# 
# * Line 3 uses the model's predict method, identical in result to line 2
# 
# Hence line 2 demonstrates exactly how the model functions, and line 1 shows that the given weight & bias values are not exact

# ## Comparing to standard LinearRegression model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


# Reshape data as required
lm_X_train = X_train.reshape(-1, 1)
lm_y_train = y_train.reshape(-1,1)
lm_X_test = X_test.reshape(-1,1)
lm_y_test = y_test.reshape(-1,1)


# In[ ]:


# Create model & predictions
linear_model = LinearRegression()
linear_model.fit(lm_X_train, lm_y_train) # Runs almost instantly
lm_pred = linear_model.predict(lm_X_test)


# In[ ]:


# Evaluate & compare

from decimal import Decimal

keras_rmse = '%.2E' % Decimal(np.sqrt(mean_squared_error(y_test, pred)))
lm_rmse = '%.2E' % Decimal(np.sqrt(mean_squared_error(y_test, lm_pred)))

print('Linear Model RMSE: ', lm_rmse)
print('Keras RMSE: ', keras_rmse)


# As you can see here, the root mean squared error for the linear model is many magnitudes smaller than that of the neural network. Clearly neural networks are therefore not the ideal solution to every problem.
