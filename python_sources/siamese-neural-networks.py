#!/usr/bin/env python
# coding: utf-8

# # Siamese Neural Networks
# For those who were present on March 5, the Monday before Spring break, you probably competed against your fellow clubmates on a competition to recognize images of icebergs from images of ships. These images were taken from space with a Sentinel-4 satellite. The goal was to create the most accurate neural network to differentiate them. However, neural networks usually only work well with A LOT of data, and I constrained you guys to 1000 examples to train on. That seems like a lot, but a CONVENTIONAL neural network needs more data. However, we are going to make a special kind of neural network: *Siamese Network*. This special network is used for face recognition and few-shot learning(learning from few examples). Let's get started by importing the usual stuff:

# In[1]:


from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


# ## Let's Grab the Data
# Same as in the competition

# In[2]:


npz = np.load('../input/input_data.npz')
X_train = npz['X_train']
Y_train = npz['Y_train']
del npz
print ('We have {} examples to work with'.format(Y_train.shape[0]-1000))


# ## Looking at the Data

# In[3]:


ix = 100 #0-4112
imshow(np.squeeze(X_train[ix,:,:,2]))#Looking at the combined channel
plt.show()
labels = ['Ship','Iceberg']#0 is no iceberg(ship) and 1 is iceberg
print ('This is:',labels[int(Y_train[ix])])


# ## Alright, let's talk about this neural network
# ![](https://i.ytimg.com/vi/6jfw8MuKwpI/maxresdefault.jpg)
# That's a visualization of a Siamese Network. It's essentially 2 networks. However, they have the same weights, making them identical. We can refer to the networks as the *left* or the *right*. The output of each network is an encoding: a string of numbers. Essentially it is expressing it's understanding of an image into a list of numbers. Let's say its 2 numbers, in 2D space. Let's say we are doing it for face recognition. We have to faces, like the ones shown above. This is how we learn: the same person should have a number similar to other pictures of the same person and different from other people's. My face could have an encoding of (-5,5). This number could change a little as I rotate my head or move it to different positions. The network is trying to increase the euclidean distance of the encoding of my face from other faces.
# 
# Let's start building this network:

# In[66]:


# We have 2 inputs, 1 for each picture
left_input = Input((75,75,3))
right_input = Input((75,75,3))

# We will use 2 instances of 1 network for this task
convnet = Sequential([
    Conv2D(5,3, input_shape=(75,75,3)),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(5,3),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(7,2),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(7,2),
    Activation('relu'),
    Flatten(),
    Dense(18),
    Activation('sigmoid')
])
# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])


# ## That's a lot more complicated than what we did before
# You just need to break it into steps:
# 1. 2 Inputs for each images
# 2. Creating a network which both images will go through individually
# 3. Couple the network to each input
# 4. Calculate the L1 distance between them. Just (x1,y1)-(x2,y2)
# 5. 1 Added layer that will say 1 if they are the same and 0 if they are different
# 
# It's not time to train though. We still have to create pairs of images to train on. There will be Positive(the same class) or Negative(different classes) for outputs. Let's construct this dataset. It is recommended by the people who published the research paper that there are equal amounts positive and negative.

# In[ ]:


# First let's separate the dataset from 1 matrix to a list of matricies
image_list = np.split(X_train[:1000],1000)
label_list = np.split(Y_train[:1000],1000)

left_input = []
right_input = []
targets = []

#Number of pairs per image
pairs = 5
#Let's create the new dataset to train on
for i in range(len(label_list)):
    for _ in range(pairs):
        compare_to = i
        while compare_to == i: #Make sure it's not comparing to itself
            compare_to = random.randint(0,999)
        left_input.append(image_list[i])
        right_input.append(image_list[compare_to])
        if label_list[i] == label_list[compare_to]:# They are the same
            targets.append(1.)
        else:# Not the same
            targets.append(0.)
            
left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))

iceimage = X_train[101]
test_left = []
test_right = []
test_targets = []

for i in range(Y_train.shape[0]-1000):
    test_left.append(iceimage)
    test_right.append(X_train[i+1000])
    test_targets.append(Y_train[i+1000])

test_left = np.squeeze(np.array(test_left))
test_right = np.squeeze(np.array(test_right))
test_targets = np.squeeze(np.array(test_targets))


# ## Hooray! We have a lot more examples now!
# Now we have pairs x 1000 examples to train the network on. Each side will have an input of an image and the output will be one if they are the same and zero if not. I'll show you how to do predictions after!

# In[ ]:


siamese_net.summary()
siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=30,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))


# In[ ]:




