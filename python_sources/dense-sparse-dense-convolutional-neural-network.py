#!/usr/bin/env python
# coding: utf-8

# # Dense-Sparse-Dense Convolutional Neural Network
# A convolutional neural network is used for training on image data because of it's properties. Dense-Sparse-Training is a method used on ANY neural network(CNN,RNN,GRU,LSTM) to improve convergence. The process is pretty simple.

# In[ ]:


import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.io import imshow
from keras import backend as K
from keras.constraints import Constraint


# ## Getting the Data
# The data is encoded into Comma Separated Value(csv) files. It is essentially a table of values. Let's import that using Pandas, a data science framework and separate the images from the labels.

# In[ ]:


#Read CSV
csv = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
#Separate into matricies
X_train = csv.iloc[:,1:786].as_matrix()
Y_train = csv.iloc[:,0].as_matrix()


# ## Our Data
# The data we just got is in 2 different variables as matricies. We have 60 thousand examples of 10 different articles of clothing. The `X_train` variable contains the images in flattened form and the `Y_train`  variable contains the labeled clothing item for each flattened image.
# People have used flattened images for recognition, but it doesn't work as well. The problem with doing so is that the structure of the image is destroyed. Imagine trying to learn what a shoe or hat looks like by looking at a string of numbers instead of an image. We have 2 steps we must complete before we feed in the data.
# 
# ### 1. Convert the strings of numbers to images

# In[ ]:


# This is very simple
X_train_imgs = np.zeros([X_train.shape[0],28,28,1])
for i in range(X_train.shape[0]):
    img = X_train[i,:].reshape([28,28,1])/255.
    X_train_imgs[i] = img


# ### 2. Now we have to get the number encodings to one-hot encodings
# Like in our last club meeting, we used vectors(lists) of ones and zeros to find out what type of land it was. Kindly they had already made the labels one-hot encoded, but this time we have to do it ourselves. The good thing is, it's really easy. Let's get out of the way:

# In[ ]:


#oh stands for one-hot
#There are 60000 examples and 10 different pieces of clothing
Y_train_oh = np.zeros([Y_train.shape[0],10])
for i in range(Y_train.shape[0]):
    oh = np.zeros([10])
    oh[int(Y_train[i])] = 1.
    Y_train_oh[i] = oh


# Keep in mind there are WAY faster ways to do those 2 steps using the power of libraries like numpy. If you were to be working with Big Data(Terabytes of information), it is extremely important that you optimize for speed. However, the amount of data we are working with is tiny enough we don't have to worry about it.

# #### Let's take a look at these images now

# In[ ]:


ix = 12345 #0-41999
imshow(np.squeeze(X_train_imgs[ix]))
plt.show()
label = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print ('This is:',label[int(Y_train[ix])])


# ## How to do Dense-Sparse-Dense Training
# dense-sparse-dense(DSD) training is a method used in this paper -> https://arxiv.org/pdf/1607.04381.pdf <- for improving the convergence of neural networks. It works like this:
# 1. Train neural network to convergence
# 2. Set all weights at a certain threshold to 0
# 3. Train the neural network again while keeping those weights 0
# 4. Release the 0 weights and train at 1/10 the learning rate
# 
# This method of training improves the accuracy of every model it has been tested against on the paper(~9% relative improvements). In Keras, I think we can use a kernel constraint to keep weights at 0 to train a neural network sparsely.
# 

# In[ ]:


#Let's code our own constraint!
class Sparse(Constraint):
    '''
    We will use one variable: Mask
    After we train our model dense model,
    we will save the weights and analyze them.
    We will create a mask where 1 means the
    number is far away enough from 0 and 0
    if it is to close to 0. We will multiply
    the weights by 0(making them 0) if they
    are supposed to be masked.
    '''
    
    def __init__(self, mask):
        self.mask = K.cast_to_floatx(mask)
    
    def __call__(self,x):
        return self.mask * x
    
    def get_config(self):
        return {'mask': self.mask}


# ## Time to make the neural network
# Let's get the layers coded.
# 
#  1. Convolution with 32 5x5 filters and input_shape=(28,28,1)
#  2. relu Activation
#  3. 2x2 Pooling with a stride of 2
#  4. Convolution with 64 5x5 filters
#  5. relu Activation
#  6. 2x2 Pooling with a stride of 2
#  7. Flatten
#  8. Dense Layer with 1024 units
#  9. relu Activation
#  10. Dropout with p=0.4
#  11. Dense Layer with 10 units
#  12. softmax Activation

# In[ ]:


# Make sure you separate layers with commas!
model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])


# ## Let's compile and view the model!
# To do that we need to have a loss function, optimizer, and metrics.
# The loss function we will use is 'categorical_crossentropy'.
# The optimizer is called the adam(Adaptive Momentum).
# The metric we will use is 'accuracy' so we can see our model accuracy during training.
# 
# Then we call `model.summary()` to see some stats about our model

# In[ ]:


adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


# ## Alright! Time to Train our Dense Model!
# The first step in the process of Dense-Sparse-Dense training is to converge a model. That's pretty simple. We just have to train our model!

# In[ ]:


#We will train on 41000 examples and validate on 18999(To be quick)
model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))


# ## Now it's time to make the Sparse Model!
# First let's get the weights of the model and set the mask to be the x% closest to 0! This function I implemented is very simple. All it does is set the closest x% to 0.
# **NOTE**: Make sure the list returned by model.get_weights() goes with this:
# `[weights,biases,weights,biases.....]`
# We analyze every other variable(which is weights) for masking. If every layer that should have weights and biases has biases on, it should be fine*(IDK about BatchNorm)*.

# In[ ]:


def create_sparsity_masks(model,sparsity):
    weights_list = model.get_weights()
    masks = []
    for weights in weights_list:
        #We can ignore biases
        if len(weights.shape) > 1:
            weights_abs = np.abs(weights)
            masks.append((weights_abs>np.percentile(weights_abs,sparsity))*1.)
    return masks


# In[ ]:


masks = create_sparsity_masks(model,30)#Closest 30% to 0


# In[ ]:


sparse_model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1), kernel_constraint=Sparse(masks[0])),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3, kernel_constraint=Sparse(masks[1])),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250, kernel_constraint=Sparse(masks[2])),
    Activation('relu'),
    Dropout(0.4),
    Dense(10, kernel_constraint=Sparse(masks[3])),
    Activation('softmax')
])

adam = Adam()
sparse_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
sparse_model.summary()
#Get weights from densely trained model
sparse_model.set_weights(model.get_weights())


# ## Let's train the Sparse Model Now!
# Just use the same learning rate and stuff for the fit function!

# In[ ]:


sparse_model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))


# ## Now to train it as a Dense Network
# We will do it just like we did the last step:
# 1. Create new model without the Sparse constraint
# 2. Train the network(with 1/10 the learning rate)

# In[ ]:


redense_model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])

adam = Adam(lr=0.0001)#Default Adam lr is 0.001 so I set it to 0.0001
redense_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
redense_model.summary()
#Get weights from sparsely trained model
redense_model.set_weights(sparse_model.get_weights())

redense_model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))


# ## Phew! That took some time!
# The biggest disadvantage for DSD training is the tripled training time, but if you already have a trained and well-tuned model, I would use this method to improve accuracy.

# In[ ]:


#First, let's get all the predictions
p = redense_model.predict(X_train_imgs[41000:],verbose=1)


# ### Now we can see the outputs
# When we get the prediction out it is in a list as 10 probabilites. One for each clothing item.

# In[ ]:


ix = 300
imshow(np.squeeze(X_train_imgs[41000+ix]))
plt.show()
print ('Probabilities:')
i = 0
for i in range(10):
    correct = (Y_train[41000+ix] == i)*1
    print ('|'+'\u2588'*int(p[ix,i]*50)+' '+label[i]+' {:.5f}%'.format(p[ix,i]*100)+' <=='*correct)


# This method can be used multiple times in another sparse -> redense cycle to gain a decreasing amount of performance.
