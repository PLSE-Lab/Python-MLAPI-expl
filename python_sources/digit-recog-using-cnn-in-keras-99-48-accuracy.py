#!/usr/bin/env python
# coding: utf-8

# **Convolutional Neural Network with Keras**
# 
# After working on different machine learning algorithms  have started analyzing  Neural Networks and how they works.
# It is always better to have knowledge of both high level approach and then start learning mathematical derivations of algorithms we are applying.
# Coming to Image recognition  tasks what I found after using neural network is that Neural network  are really much more powerful as compared to ML algos. In Ml we first need to identify contours then subparts of contour like wheels ,window if we want to identify a car which requires applying different logics and different algos.
# In deep learning we only define layers of neurons in particular order and choose type of activation function and loss function
# I have compiled below mentioned information after studying from multiple sources on internet as reading just 1 article  we can not have clear picture about CNN.
# 
# In this kernal you will gain knowledge about - 
# 1.  Theoritical concept of Convolutional Neural Network.
# 2.  Basic Perceptron Algorithm & BackPropogation Derivation.
# 3.  Using High level library Keras for recognizing MNIST dataset.

# **First why Convolutional Neural Networks ?**
# Each neuron in layer 1 and layer 2 of traditional neural networks are interconnected which makes it  computationaly expensive task to process images using neural networks as lot of weights needed to be learned.
# In contrast CNN networks only few number of neurons are connected to neuron in 2nd layer depending on what feature we want to learn.Each combination identifying different set of features.
# 
# **Theoritical concept of Convolutional Neural Network**
# Please go through below mentioned link which gives very intuitive approach to learn - 
# https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8
# https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/

# **2. Basic Perceptron Algorithm & BackPropogation Derivation -** 
# As explaining everything is beyond the scope of this notebook so just attaching screenshot from my notes if you have any doubt please feel free to comment below or you can email me at chakshu00garg@gmail.com -
# ![image.png](attachment:image.png)
#  
# 

# **3. Using High level library Keras for recognizing MNIST dataset.**
#    In starting I was confused due to different Deep learning libraries available in market each having own drawback and benefits. I choose Keras to start with as we can build powerful models on top of popular tensorflow and other libraries using this high level library in just few minutes giving us more spare time to invest in choosing number of layers and structure of neurons in network.

# Importing all required libraries

# In[ ]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')


# Importing our training data which is 28x28 size images of Hand written digits

# In[ ]:


K.set_image_dim_ordering('th')
seed = 7
numpy.random.seed(seed)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



# In[ ]:


image_gen = ImageDataGenerator(featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,                                               
    horizontal_flip=False, shear_range=0.2)
image_gen.fit(X_train, augment=True)


# **CNN Layers -**
# * **layer 1**
# Convolution Layer (30 features, same padding) --> relu --> Convolution Layer (30 features)  --> relu --> pooling layer 
# * **layer 2**
# Convolution Layer (15 features, same padding) --> relu --> Convolution Layer (15 features) --> relu --> pooling layer --> Flatten --> DenseLayer --> Dense Layer  --> output
# 
# **Explaining few terms -** 
# 1. Relu is a non linear activation function that maps output between 0 and infinity 
# 2. Same padding keeps the matrix size same by adding zeros in additional space
# 3. Dropout helps in reducing overfitting in neural networks it turns off few neurons in layer randomly during each forward pass 
# 4. As we have multiple output classes we will use softmax function in output layer and categorical_crossentropy as loss function
# 5. Optimization algo is used to update weights in network. Adam optimizer requires less memory and is Computationally efficient with good results 

# In[ ]:


def cnn_model():
    model = Sequential()
    
    model.add(Conv2D(30, (5, 5), padding='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(30, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(15, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# **Fitting our model **
# * 1 Epoch = 1 Forward pass + 1 Backward pass for ALL training samples.I choose this to be 20 after hit and trial.Do remember to turn on gpu feature as it speeds up process.
# * Batch Size = Number of training samples in 1 Forward/1 Backward pass. (With increase in Batch size, required memory space increases.)

# In[ ]:


model = cnn_model()


#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200)

model.fit_generator(image_gen.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train) / 32,                                epochs=32)


# **Model Accuracy - 99.48 % which is pretty accurate** 

# In[ ]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(model.summary())


# Refrences ,Textbooks and Tutorials :
# *  https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8
# *  https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
# *  https://youtu.be/0e0z28wAWfg
#  
# 
# This Kernel is written by Chakshu Garg
# 
# **I will keep on adding new findings to optimize CNN models in this notebook. Do share your valuable feedback in comment section below**. **:)**
