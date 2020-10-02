#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # **Keras basics for beginners**
# 
# 
# Hello friends,
# 
# In this kernel, I will discuss Keras and Keras fundamentals. In particular, I will show how to compile, train and evaluate the model using Keras. Also, I present a Simple Linear Regression example using Keras and visualize the results. 
# 
# So, let's get started.

# **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be very much appreciated**

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Introduction to Keras](#1)
# 1. [What is a backend](#2)
# 1. [Keras fundamentals](#3)
#    - [Keras Sequential model](#3.1)
#    - [Keras Functional API](#3.2)
# 1. [Keras layers](#4)
#    - [Sequential Model](#4.1)
#    - [Convolutional Layer](#4.2)
#    - [MaxPooling Layer](#4.3)
#    - [Dense Layer](#4.4)
#    - [Dropout Layer](#4.5)
# 1. [Compile, train and evaluate model](#5)
#    - [Compile with .compile() method](#5.1)
#    - [Train with ,fit() method](#5.2)
#    - [Evaluate with .evaluate() method](#5.3)
# 1. [Keras in action - Simple Linear Regression example](#6)
# 1. [Conclusion](#7)
# 

# ## 1. Introduction to Keras <a class="anchor" id="1"></a>
# 
# 
# [Back to Table of Contents](#0.1)
# 
# 
# 
# - Keras is an Open Source Neural Network library written in Python that runs on top of Theano or Tensorflow. 
# 
# - It is designed to be modular, fast and easy to use.
# 
# - Keras High-Level API handles the way we make models, defining layers, or set up multiple input-output models. In this level, Keras also compiles our model with loss and optimizer functions, training process with fit function. 
# 
# - Keras doesn't handle Low-Level API such as making the computational graph, making tensors or other variables because it has been handled by the "backend" engine.
# 
# - So, Keras doesn't handle low-level computation. Instead, it uses another library to do it, called the **Backend**. Thus, Keras is a high-level API wrapper for the low-level API, capable of running on top of TensorFlow, CNTK or Theano.
# 
# - Please consult the Keras Official documentation for more information on Keras:-
# 
# [Keras Official Documentation](https://keras.io/)

# ## 2. What is a backend <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)
# 
# - **Backend** is a term in Keras that performs all low-level computations such as tensor products, convolutions and many other things with the help of other libraries such as Tensorflow or Theano. 
# 
# - So, the **backend engine** will perform the computation and development of the models. Tensorflow is the default **backend engine** but we can change it in the configuration.

# ## 3. Keras fundamentals <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)
# 
# 
# - The main structure in Keras is the model which defines the complete graph of a network. 
# 
# - It is a way to organize layers.
# 
# - The simplest type of model is the **Sequential model**. It is the linear stack of layers. 
# 
# - For more complex architectures, we should use the **Keras functional API**, which allows to build arbitrary graphs of layers.

# ### 3.1 Keras Sequential model <a class="anchor" id="3.1"></a>
# 
# 
# - The **Sequential model** is a linear stack of layers.
# 
# - We can create a Sequential model by passing a list of layer instances to the constructor as follows:-
# 
# 
# `from keras.models import Sequential`
# 
# `from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout`
# 
# `model = Sequential()`
# 
# 
# - We can also simply add layers via the **.add()** method as follows:-
# 
# `model = Sequential()`
# 
# `model.add(Dense(32, input_dim=784))`
# 
# `model.add(Activation('relu'))`
# 
# 
# - For more detailed discussion on Keras Sequential model follow the link below:-
# 
# 
# [Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)

# ### 3.2 Keras Functional API <a class="anchor" id="3.2"></a>
# 
# 
# - The **Keras functional API** is used to define complex models, such as multi-output models, directed acyclic graphs, or models with shared layers.
# 
# 
# - For more detailed discussion on Keras Functional API follow the link below:-
# 
# [Keras Functional API](https://keras.io/getting-started/functional-api-guide/)
# 

# ## 4. Keras layers <a class="anchor" id="4"></a>
# 
# 
# [Back to Table of Contents](#0.1)
# 
# 
# - Keras consists of different types of layers which are fundamental to building blocks of Keras.
# 
# - In this section, we will discuss few commonly used layers in Keras.
# 

# ### 4.1 Sequential Model <a class="anchor" id="4.1"></a>
# 
# 
# - We can create a Sequential model by passing a list of layer instances to the constructor as follows:-
# 
# 
# `from keras.models import Sequential`
# 
# `from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout`
# 
# `model = Sequential()`
# 
# 

# ### 4.2 Convolutional Layer <a class="anchor" id="4.2"></a>
# 
# 
# - This is an example of convolutional layer as the input layer with the input shape of 320x320x3, with 48 filters of size 3x3 and use ReLU as an activation function.
# 
# 
# `input_shape=(320,320,3)`  #this is the input shape of an image 320x320x3
# 
# `model.add(Conv2D(48, (3, 3), activation='relu', input_shape= input_shape))`
# 
# 
# - Another example is as follows:-
# 
# `model.add(Conv2D(48, (3, 3), activation='relu'))`

# ### 4.3 MaxPooling Layer <a class="anchor" id="4.3"></a>
# 
# 
# - To downsample the input representation, use MaxPool2d and specify the kernel size.
# 
# 
# `model.add(MaxPooling2D(pool_size=(2, 2)))`

# ### 4.4 Dense Layer <a class="anchor" id="4.4"></a>
# 
# 
# - We can add a fully connected layer with just specifying the output size,
# 
# `model.add(Dense(256, activation='relu'))`

# ### 4.5 Dropout Layer <a class="anchor" id="4.5"></a>
# 
# 
# - We can add a dropout layer with 50% probability as follows:-
# 
# `model.add(Dropout(0.5))`

# ## 5. Compile, train and evaluate model <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 5.1 Compile with .compile() method <a class="anchor" id="5.1"></a>
# 
# 
# - After we have define our model, we will train them. 
# 
# - It is required to compile the network first with the loss function and optimizer function. 
# 
# - This will allow the network to change weights and minimized the loss.
# 
# - We will compile our model with **.compile()** method as follows:-
# 
# 
# `model.compile(loss='mean_squared_error', optimizer='adam')`
# 
# 

# ### 5.2 Train with .fit() method <a class="anchor" id="5.2"></a>
# 
# 
# - Now we want to train our model.
# 
# - We can use **.fit()** method to fed the training and validation data to the model. 
# 
# - This will allow you to train the network in batches and set the epochs as follows:-
# 
# `model.fit(X_train, X_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))`

# ### 5.3 Evaluate with .evaluate() method <a class="anchor" id="5.3"></a>
# 
# 
# - The final step is to evaluate the model with the test data.
# 
# - It can be done with the **.evaluate()** method as follows:-
# 
# `score = model.evaluate(x_test, y_test, batch_size=32)`

# ## 6. Keras in action - Simple Linear Regression example <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Import necessary modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

 
# Define data for the model
x = data = np.linspace(1,2,200)
y = x*4 + np.random.randn(*x.shape) * 0.3


# Create a Sequential model
model = Sequential()


# Add layers to the Sequential model
model.add(Dense(1, input_dim=1, activation='linear'))


# Compile the model
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])


# Declare initial weights and bias
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 


# Train the model
model.fit(x,y, batch_size=1, epochs=30, shuffle=False)


# Set final weights and bias
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))


# Predict the results
predict = model.predict(data)


# Visualize the results
plt.figure(figsize=(12,8))
plt.plot(data, predict, 'b', data , y, 'k.')
plt.show()


# After training the data, the output should look like the above plot.

# ## 7. Conclusion <a class="anchor" id="7"></a>
# 
# 
# [Back to Table of Contents](#0.1)
# 
# - In this kernel, I present a high level overview of Keras - the Deep Learning library of Python.
# 
# - In particular, I discuss the Keras Sequential model and Keras Functional API, common layers in Keras and how to compile, train and evaluate our model.
# 
# - Then, I present a simple linear regression example using Keras

# Thus, we come to the end of this kernel.
# 
# 
# I hope you find it useful and enjoyable.

# [Go to Top](#0)
