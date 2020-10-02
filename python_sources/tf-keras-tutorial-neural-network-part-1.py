#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Keras Tutorial - Neural Network (Part 1)
# 
# **What is Keras?** Keras is a wrapper that allows you to implement Deep Neural Network without getting into intrinsic details of the Network. It can use *Tensorflow* or *Theano* as backend. This tutorial series will cover Keras from beginner to intermediate level.
# 
# In this part we will cover:
# * Loading MNIST Digit dataset
# * Image Data basic preprocessing
# * Training Simple Neural Network
# * Validating our Model
# * How to stop when we reach desired accuracy/loss

# # Importing libraries

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset
# 
# Here we are loading mnist Dataset which is preloaded in tensorflow. <br>
# 
# >```mnist = tf.keras.datasets.mnist```<br>
# This returns the dataset object. Similarly there are 6 more datasets preloaded in keras.
# 
# >Calling the `load_data` function on this object returns splitted train and test data in form of (features, target).

# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() 


# # Overview of the dataset

# The dataset contains images, each image of 28x28px. There are 6000 images in training data and 1000 images in test data.<br>
# >The shape (6000, 28, 28) represents **6000** images each of dimension **28x28**.<br>
# The shape **(6000, )** represents (6000, 1) shape i.e. 6000 labels, each for one image.

# In[ ]:


print(f'Shape of the training data: {x_train.shape}')
print(f'Shape of the training target: {y_train.shape}')
print(f'Shape of the test data: {x_test.shape}')
print(f'Shape of the test target: {y_test.shape}')


# In[ ]:


print(y_train)


# In[ ]:


# Let's plot the first image in the training data and look at it's corresponding target (y) variable.
plt.imshow(x_train[0], cmap='gray')
print(f'Target variable is {y_train[0]}')


# # Preprocess
# 
# The image pixels in the data range from 0-255. We will scale them from 0-1 to better fit the Neural Network. This process is called **Normalization**.<br>
# > However, this is not a compulsory step. You can go ahead and remove the lines to see the effect on the final output.

# In[ ]:


# Setting custom printwidth to print the array properly
np.set_printoptions(linewidth=200)
print(x_train[0])


# In[ ]:


# Normalizing the data
# each element of nested list/array in python is divided by using a simple division operator on the list/array
x_train = x_train/255
x_test = x_test/255


# In[ ]:


print(x_train[0])


# # Modelling
# 
# There are two types of models in Tensorflow:
#  - **Sequential**
#  - **Graphical**
# 
# ## Models
# `tf.keras.model.Sequential()` lets you create a linear stack of layers providing a Sequential netural network.<br>
# `tf.model()` allows you to create arbitarary graph of layers as long as there is no cycle.
# 
# ## Flatten Layer
# `tf.keras.layes.Flatten()` flattens the input.<br>
# For input of `(batch_size, height, width)` the output converts to `(batch_size, height*width)`
# 
# ## Dense Layer
# `tf.keras.layers.Dense()` Normal dense layer of Neural Network where each node is connected to each node in the next layer. <br>
# >The two arguments passes below in dense layer are *units* and *activation* (activation function).<br>
# * **units** corresponds to the number of nodes in the layer<br>
# * **activation** is an element-wise activation function.
#     * **relu**: This activation function converts every negative value to 0 and positive remains the same
#     * **softmax**: This function takes the element with max value, converts it into 1 and rest to 0.
# 
# In the below example we've 3 dense layers with 128, 64 and 10 nodes respectively. The layer with 10 nodes is supposed to be the output layer. Since we just need single value as 1 (preferably the maximum value), therefore we apply *softmax* activation function to the final/output layer.
# 
# ## Compiling model
# `model.compile()` Sets up the optimiser, loss and metrics configuration.
# > * **optimizer**: updates the parameter of the Neural Network.
# * **loss**: Measures the error in our model.
# * **metrics**: Used to judge the model. The difference between metrics and loss is that metrics in not used to evaluate the model while training, whereas loss evaluates the model error while training and helps optimizer reduce the error.

# In[ ]:


# Creating the architecture of model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# # Training
# 
# ```model.fit``` trains the model.
# > * **x_train**: Training data/features
# * **y_train**: Target
# * **epochs**: Number of times the entire dataset is fed in the model
# 
# While training you can see the loss and accuracy calculated on the training data itself. *The number of epochs are a kind of try and test metrics. It depends on a number of factors like size of data and complexity of classification, etc.* You will slowly get a feeling of how to estimate number of epochs required for a particular model and dataset.

# In[ ]:


model.fit(x_train, y_train, epochs=3)


# # Validation
# 
# We've verified our model's accuracy for training data which comes out to be 97% approx. This accuracy is calculated on the same data on which the model is trained. Now let's see the accuracy when our model gets new data on which it is not trained. Generally it should be slightly lower than the training accuracy.<br>
# > This is done to avoid overfitting. **Overfitting** is the case when the model gives *high accuracy on training data but low accuracy on test data*. Whenever you see a hugh difference in the Train accuracy and Validation accuracy, it is an example of overfitting. 
# Overfitting can be caused due to a number of reasons and have several ways to avoid and fix. We will discuss that later in the course.<br>
# *Note that we expect the validation accuracy to be slightly lower than training accuracy but not much lower.*

# In[ ]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_acc}')


# # Stopping at reaching accuracy
# 
# Now let's suppose you need 95% accuracy for your model but you have no idea of the number of epochs required to reach that level of accuracy. You set a large number of epochs but most likely after it reaches the required accuracy it will go on training resuling into overfitting. So you just need the training to stop when it reaches to 95%. Let's see how we can do that.

# In[ ]:


# Callback class which checks on the logs when the epoch ends
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.05):
      print("\nReached Minimal loss so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Pass callbacks parameter while training
model.fit(x_train, y_train, epochs=50, callbacks=[callbacks])


# Note that we've set the epochs to 50 but as soon as the loss is below 0.05, the training is completed. You can also monitor and check ```accuracy``` parameter in place of ```loss```. Try it out yourself.
# 
# **IN THE NEXT TUTORIAL WE WILL SEE A SIMPLE CONVOLUTIONAL NEURAL NETWORK AND SEE HOW IT IS BETTER FROM SIMPLE NEURAL NETWORK.**
# 
# > # PART 2 [Convolutional Neural Network](https://www.kaggle.com/akashkr/tf-keras-tutorial-cnn-part-2)
