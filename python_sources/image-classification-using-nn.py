#!/usr/bin/env python
# coding: utf-8

# ## Image Classification
# 
# 
# Image classification is one of the important use cases in our daily life. Automotive, e-commerce, retail, manufacturing industries, security, surveillance, healthcare, farming etc., can have a wide application of image classification.
# 
# **Objective:** In this notebook, we will build a neural network to classifiy the image based on the object present in the image.
# 

# 
# ## Advanced techniques for training neural networks
# 
# Weight Initialization
# 
# Nonlinearity (different Activation functions)
# 
# Optimizers(different optimizers)
# 
# Batch Normalization
# 
# Dropout

# ### About Dataset
# 
# 
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.
# 
# #### Labels
# 
# Each training and test example is assigned to one of the following labels:
# 
# 0 T-shirt/top
# 
# 1 Trouser
# 
# 2 Pullover
# 
# 3 Dress
# 
# 4 Coat
# 
# 5 Sandal
# 
# 6 Shirt
# 
# 7 Sneaker
# 
# 8 Bag
# 
# 9 Ankle boot 

# ### Load dataset
# 
# Fashion-MNIST dataset
# 
# source: https://www.kaggle.com/zalando-research/fashionmnist
# 

# In[ ]:


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical


# In[ ]:


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[ ]:


plt.imshow(X_train[0])    # show first number in the dataset
plt.show()
print('Label: ', y_train[0])


# In[ ]:


plt.imshow(X_test[0])    # show first number in the dataset
plt.show()
print('Label: ', y_test[0])


# ### Data Pre-processing

# In[ ]:


# reshaping X data: (n, 28, 28) => (n, 784)
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))


# In[ ]:


# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Basic NN model
# 
# Naive MLP model without any alterations

# In[ ]:


from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(50, input_shape = (784, )))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[ ]:


sgd = optimizers.SGD(lr = 0.01)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, batch_size = 200, epochs = 100, verbose = 1)


# In[ ]:


results = model.evaluate(X_test, y_test)


# In[ ]:





# In[ ]:


print('Test accuracy: ', results[1])


# ### 1. Weight Initialization
# 
# Changing weight initialization scheme can significantly improve training of the model by preventing vanishing gradient problem up to some degree
# 
# Ref: https://keras.io/initializers/

# In[ ]:


# from now on, create a function to generate (return) models
def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, ), kernel_initializer='he_normal'))     # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(50, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(50, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(50, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('sigmoid'))    
    model.add(Dense(10, kernel_initializer='he_normal'))                            # use he_normal initializer
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


# In[ ]:


model = mlp_model()
history = model.fit(X_train, y_train, batch_size=200, epochs = 100, verbose = 1)


# In[ ]:


results = model.evaluate(X_test, y_test)


# In[ ]:


print('Test accuracy: ', results[1])


# ### 2. Nonlinearity (Activation function)
# 
# Sigmoid functions suffer from gradient vanishing problem, making training slower
# 
# There are many choices apart from sigmoid and tanh; try many of them!
# 
# 'relu' (rectified linear unit) is one of the most popular ones
# 
# Ref: https://keras.io/activations/

# In[ ]:


def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(Activation('relu'))    
    model.add(Dense(50))
    model.add(Activation('relu'))    
    model.add(Dense(50))
    model.add(Activation('relu'))    
    model.add(Dense(50))
    model.add(Activation('relu'))    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


# In[ ]:


model = mlp_model()
history = model.fit(X_train, y_train, epochs = 10, verbose = 1)


# In[ ]:


results = model.evaluate(X_test, y_test)


# In[ ]:


print('Test accuracy: ', results[1])


# ### 3. Batch Normalization
# 
# Batch Normalization, one of the methods to prevent the "internal covariance shift" problem, has proven to be highly effective
# 
# Normalize each mini-batch before nonlinearity
# 
# Ref: https://keras.io/optimizers/

# In[ ]:


from keras.layers import BatchNormalization, Dropout


# Batch normalization layer is usually inserted after dense/convolution and before nonlinearity
# 
# 

# In[ ]:


def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, )))
    model.add(BatchNormalization())                    
    model.add(Activation('relu'))    
    model.add(Dense(50))
    model.add(BatchNormalization())                    
    model.add(Activation('relu'))    
    model.add(Dense(50))
    model.add(BatchNormalization())                    
    model.add(Activation('relu'))    
    model.add(Dense(50))
    model.add(BatchNormalization())                    
    model.add(Activation('relu'))    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


# In[ ]:


model = mlp_model()
history = model.fit(X_train, y_train, epochs = 20, verbose = 1)


# In[ ]:


results = model.evaluate(X_test, y_test)


# In[ ]:


print('Test accuracy: ', results[1])


# ### Dropout

# In[ ]:


def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape = (784, ), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


# In[ ]:


model = mlp_model()
history = model.fit(X_train, y_train, epochs = 10, verbose = 1)


# In[ ]:


results = model.evaluate(X_test, y_test)


# In[ ]:


print('Test accuracy: ', results[1])

