#!/usr/bin/env python
# coding: utf-8

# # This Week's Iceberg Classification Competition
# This will be our first individual activity we will do in the club. Before I've been teaching you the basics of how a neural network works. Now, I want to see what you guys can do. You may partner up with up to 1 other person for this competition and work together. Keep in mind, you will have to share the candy!
# 
# Let's get started. A month ago, there was a competition on who could make the best algorithm that could classify if a satellite image was of a ship or of an iceberg. This was also my first competition. My best submission ranked in the top 24%, but my best model got top 17%(but it wasn't submitted). I think this would be a good activity to do something for real!
# 
# Just importing the usual stuff:

# In[1]:


import numpy as np
import pandas as pd
from skimage.io import imshow
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


# ## Let's Grab the Data
# I will do this for you since we are crunched for time. It's also pretty boring.

# In[52]:


npz = np.load('../input/input_data.npz')
X_train = npz['X_train'][:1000]
Y_train = npz['Y_train'][:1000]
del npz


# ### You can take a peek at the data here:
# 1000 examples of 75 by 75 by 3 satellite images. These are not your normal RGB images too. They use to polarizations of infrared light and the third channel is a combination of them. There are originally 4113, but we don't have the time to train it on all of them during testing.

# In[53]:


ix = 100 #0-3999
imshow(np.squeeze(X_train[ix,:,:,2]))#Looking at the combined channel
plt.show()
labels = ['Ship','Iceberg']#0 is no iceberg(ship) and 1 is iceberg
print ('This is:',labels[int(Y_train[ix])])


# ## Now for the nitty-gritty
# ***YOU*** are now going to make a neural network to predict if the images are of an iceberg or ship!
# Make sure your first layer has this argument -> `input_shape=(75,75,3)`
# Make sure you use `Flatten` before using Dense units.
# Make sure you use `Activation` after every Conv2D and Dense except the last one
# Make sure your last layer looks like this -> `Dense(1, activation='sigmoid')`
# 
# I have provided you with some more layers, each which I will describe:
# 1. **Dropout**: This layer is used A LOT. This is used to solve overfitting. This is when your training accuracy(`acc`) gets too much higher than your validation accuracy(`val_acc`). When this happens, you need to use this layer. Make sure you use this after a MaxPooling2D layer or Activation layer. The only argument, called p(between 0 and 1) is the probability a connection will be dropped. The higher this number is, the more your network is regularized, meaning it will overfit less. However, it can depend where you put it. Try to keep this layer away from the top of the model. Also, reducing the number of filters in the Conv2D or the number of units in a Dense can help.
# 2. **BatchNormalization**: This layer is also used a ton. This is a very powerful layer that works extremely well in most cases. Remember how we normalized by subtracting the mean and dividing by the standard deviation? This layer will do that to the activations inside the network. This causes faster training. I would use this, but sometimes it doesn't work.
# 3. **AveragePooling2D**: For a convolutional network to be viable, we use a pooling function. This reduces the size of the activations inside the network. Normally we use MaxPooling2D. This will just find the largest number in a box with even sides(usually 2), and use that single number to represent them all. This works pretty well, but your could give AveragePooling2D(what it does is self-explanatory). Just change the Max to Average. The code for both is the same.
# 
# Don't be afraid to use the network we used last week, but make sure you change the input size and the last 2 output layers. Good luck!
# 
# Why this is hard:
# If you get a validation log loss below 0.16, you have solved one of the hardest problems in the history of deep learning: few-shot learning.
# 

# In[130]:


model = Sequential([
    #layerrrsssss
])


# ## I'm assuming you have your network coded, but there's another step before you can train!
# When you train a neural network you need to choose a learning rate. If it's too high, your network will never converge on a good solution, and if it's too slow then you will learn too slow.

# In[131]:


learning_rate = 0.01#change it if you'd like
optimizer = Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
model.summary()


# In[132]:


#We will train on 900 examples and validate on 100
model.fit(X_train[:900], Y_train[:900],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train[900:], Y_train[900:]))


# ## Let's look at some predictions!

# In[62]:


p = model.predict(X_train[900:], verbose=1)


# In[106]:


ix = 6
imshow(np.squeeze(X_train[900+ix,:,:,2]))
plt.show()
print ('Probability:')
for i in range(1):
    print ('|'+'\u2588'*int(p[ix,i]*50)+' '*int((1-p[ix,i])*50)+'| Iceberg'+' {:.5f}%'.format(p[ix,i]*100))
print ('This is:',labels[int(Y_train[900+ix])])


# In[ ]:




