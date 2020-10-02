#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network
# Before, we learned about a densly connected neural network. That is the base for most neural networks. However, for image recognition, there is a faster, more accuracte, and more efficient type of neural network that outperforms vanilla neural networks.
# This is called a convolutional neural network. In a nutshell, it has filters that scan over the image, trying to find a type of feature.
# First let's import the import functions and libraries we need.

# In[ ]:


import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.io import imshow


# ## Getting the Data
# The data is encoded into Comma Separated Value(csv) files. It is essentially a table of values. Let's import that using Pandas, a data science framework and separate the images from the labels.

# In[ ]:


#Read CSV
csv = pd.read_csv('../input/fashion-mnist_train.csv')
#Separate into matricies
X_train = csv.iloc[:,1:786].as_matrix()
Y_train = csv.iloc[:,0].as_matrix()


# ## Our Data
# The data we just got is in 2 different variables as matricies. We have 60 thousand examples of 10 different articles of clothing. The `X_train` variable contains the images in flattened form and the `Y_train`  variable contains the labeled clothing item for each flattened image.
# Last time we used flattened satellite images and fed those into the neural network. The problem with doing so is that the structure of the image is destroyed. Imagine trying to learn what a shoe or hat looks like by looking at a string of numbers instead of an image. We have 2 steps we must complete before we feed in the data.
# 
# ### 1. Convert the strings of numbers to images

# In[ ]:


# This is very simple
X_train_imgs = np.zeros([60000,28,28,1])
for i in range(X_train.shape[0]):
    img = X_train[i,:].reshape([28,28,1])/255.
    X_train_imgs[i] = img


# ### 2. Now we have to get the number encodings to one-hot encodings
# Like in our last club meeting, we used vectors(lists) of ones and zeros to find out what type of land it was. Kindly they had already made the labels one-hot encoded, but this time we have to do it ourselves. The good thing is, it's really easy. Let's get started:

# In[ ]:


#oh stands for one-hot
#There are 60000 examples and 10 different pieces of clothing
Y_train_oh = np.zeros([60000,10])
for i in range(Y_train.shape[0]):
    oh = np.zeros([10])
    oh[int(Y_train[i])] = 1.
    Y_train_oh[i] = oh


# Keep in mind there are WAY faster ways to do those 2 steps using the power of libraries like numpy. If you were to be working with Big Data(Terabytes of information), it is extremely important that you optimize for speed. However, the amount of data we are working with is tiny enough we don't have to worry about it.

# #### Let's take a look at these images now

# In[ ]:


ix = 59999 #0-59999
imshow(np.squeeze(X_train_imgs[ix]))
plt.show()
clothing = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print ('This is:',clothing[int(Y_train[ix])])


# ## Time to make the neural network
# Let's get the layers written out, but not coded. That's ***your job***!
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
    #layers
])


# ## Let's compile and view the model!
# To do that we need to have a loss function, optimizer, and metrics.
# The loss function we will use is 'categorical_crossentropy'.
# The optimizer is called the adam(Adaptive Momentum).
# The metric we will use is 'accuracy' so we can see our model accuracy during training.
# 
# Then we call `model.summary()` to see some stats about our model

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# ## Alright! Our model will change over 1 million parameters to increase the accuracy!
# Let's begin training! **Wait!** I have to explain one thing to you guys first. It's called a *validation set*. When we train, the model is fitting itself for the highest accuracy it can manage *on what it's training on*. There is one problem: neural networks like this tend to overfit on their data. This means they can get a very high accuracy on data it has already seen because it has memorized it. That is not good because when we show it new data, it does very poorly. We want a model that is also good at giving predictions for data it has never seen. To do this we create a validation set. The model is trained on the training set while having its accuracy checked by a validation set. It doesn't train on the validation set though. If we see the validation accuracy go down and the training accuracy continue going up, it means it has begun to overfit. That means it has begun to memorize and is not learning anything anymore.

# In[ ]:


#We will train on 59000 examples and validate on 1000
model.fit(X_train_imgs[:59000], Y_train_oh[:59000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[59001:], Y_train_oh[59001:]))


# ## Phew! That took some time!
# Now our model is trained. If we were to actually use this for something, we would train for longer. However, we have time constraints. Let's test out model on the validation data!

# In[ ]:


#First, let's get all the predictions
p = model.predict(X_train_imgs[59000:],verbose=1)


# ### Now we can see the outputs
# When we get the prediction out it is in a list as 10 probabilites. One for each clothing item.

# In[ ]:


ix = 50
imshow(np.squeeze(X_train_imgs[59000+ix]))
plt.show()
print ('Probabilities:')
for i in range(10):
    print ('|'+'\u2588'*int(p[ix,i]*50)+clothing[i]+' {:.5f}%'.format(p[ix,i]*100))


# In[ ]:




