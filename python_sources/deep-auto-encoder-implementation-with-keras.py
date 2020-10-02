#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Nanangk/Deep_Autoencoder_Implementation_with_Keras/blob/master/Deep_Autoencoder_Implementation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #Deep Autoencoder Implementation

# This project aim to implementation of Deep Autoencoder with Keras, this project use fashion mnist dataset from keras
# 
# Fashion mnist is is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST. 
# 
# The class labels are:
# 
# Label	Description
# 0.	T-shirt/top
# 1.	Trouser
# 2.	Pullover
# 3.	Dress
# 4.	Coat
# 5.	Sandal
# 6.	Shirt
# 7.	Sneaker
# 8.	Bag
# 9.	Ankle boot 

# <h3>What is Autoencoder?</h3>
# Autoencoder is one of type neural network where the input is same as the output, autoencoder learn the input data and reconstruct the input data, autoencoder used for dimensionality reduction
# 
# ![Autoencoder](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRj6teOvotqH0V03TJsL5avWnHWSUBFlJUJIgSjrdff2dz3ca5C&usqp=CAU)

# In[ ]:


#import library
from keras.datasets import fashion_mnist
from keras.layers import Input, Dense
from keras.models import Model


# In[ ]:


#Load Dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()


# In[ ]:


#Rescale  dataset
import numpy as np

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# In[ ]:


#Build Autoencoder Model

target_dimension = 16

#Encoder
input_img = Input(shape=(784,))
encoder = Dense(128, activation='relu')(input_img)
encoder = Dense(64, activation='relu')(encoder)
encoder = Dense(32, activation='relu')(encoder)

#code
coded = Dense(target_dimension, activation='relu')(encoder)

#Decoder
decoder = Dense(32, activation='relu')(coded)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dense(128, activation='relu')(decoder)
decoder = Dense(784, activation='sigmoid')(decoder)

autoencoder = Model(input_img, decoder)


# In[ ]:


#compile model
autoencoder.compile(loss = 'binary_crossentropy',
                    optimizer = 'adam')


# In[ ]:


autoencoder.summary()


# In[ ]:


#Training model
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))


# In[ ]:


#Display original data and reconstruction data
import matplotlib.pyplot as plt
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(25, 5))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()


# In[ ]:




