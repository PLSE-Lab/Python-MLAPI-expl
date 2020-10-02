#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Recognizing Hand-written digits is a classical example of an important application of deep learning - namely convolutional neural networks. In this notebooks, I will be setting up a base pipeline for a network (it won't be that big and bad). The model gets a score of ~99%. I have chosen the layers, epochs and some parameters after messing around with it a bit. 
# 
# Just a friendly reminder to upvote the notebook (if you like it), as it helps me see the impact of my work :).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Pre-Processing Data
# 
# 
# In this dataset, the images are already black and white, mostly centered and have an appropriate resolution. Also, the image has been converted into a 1-dimensional array so that it will be easier to feed the model.

# In[ ]:


df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


y = df_train['label']
df_train.drop(['label'], axis=1, inplace=True)


# Even though the pixel values are in a 1-d arary, we have to reshape the images so that they are in the appropriate format.

# In[ ]:


df_train = df_train.values.reshape(-1,28,28,1)
df_test = df_test.values.reshape(-1,28,28,1)


# Let us now assign the train images to x, and the test images to x_predict (just to follow convention..), so that you can use this code for other networks.

# In[ ]:


x = df_train
x_predict = df_test


# Just to make sure everything is right, let us also plot the first image, and see for ourselves. Given the way that the data is organized, I would expect the first image to clearly be a 1.

# In[ ]:


g = plt.imshow(x[0][:,:,0])


# And it clearly looks like a 1 to us (cause we are amazing at such tasks). Let's see how the model handles the tilted lines.
# 
# # Modelling - Convolutional Neural Network
# 
# I will be using keras convolutional neural network for this, but you can use pytorch or raw tensorflow as well.

# In[ ]:


import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Also, we need to convert the y (target) label to a categorical type that keras can understand.

# In[ ]:


y = to_categorical(y, 10)


# 
# Here is the structure of our neural network:
# 
# Input Layer --> Convolution Layer --> Convolution Layer 2 --> MaxPool2D --> Convolution Layer --> MaxPool2D --> Flatten --> Dense Layer --> Output Layer
# 
# 3 convolutional layers will be enough to learn the general outline of the digits, and give a good result. I am also adding a dense layer at the end for some non-linear understanding of the different components of an image.
# 
# Also, remember to use the MaxPool2D and Flatten layers as well, because the convolution layers change the dimension of the data. You need to flatten it before you fit it into the standard Dense layer.
# 
# 

# In[ ]:


model = Sequential()
model.add(Conv2D(input_shape=(28,28,1), padding='same', activation='relu', filters = 64, kernel_size=3))
model.add(Conv2D(padding='same', activation='relu', filters=128, kernel_size=3))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(padding='same', activation='relu', filters=64, kernel_size=3))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.summary()


# First, I will create a validation set.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.9)


# Now, I will be fitting my data for 2 epochs (meaning that it will iterate through the data 2 times). Also, it will calibrate weights after averaging for 100 samples. This way, the model will train faster, but now loose too much information through averaging.
# 
# Also, do expect the following code block to take a lot of time to execute (9-10 mins). The reason is that I am not using GPU.

# In[ ]:


#x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8)
history = model.fit(x, y, epochs=3, validation_data=(x_valid, y_valid), verbose=1, batch_size=100)


# # Training Curves 
# 
# Let's see the training curves to understand how the model has improved over time (epochs). We will plot the accuracy for both the train and validation. This will help us see if the model is overfitting.

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# So we can see that the model's score somewhat stabilizes in the beginning itself. Also, we can see that the model is not overfitting (since the test and train recieve almost the same score). This is good, because we want our model to be able to generalize.

# # Predictions and Submission
# 
# Finally, let us make our predictions and submit it to the competition.

# In[ ]:


predictions = model.predict_classes(x_predict)


# In[ ]:


submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')


# In[ ]:


submission.head()


# In[ ]:


submission['Label'] = predictions
submission.to_csv("mysub.csv", index=False)


# # Conclusion
# 
# 
# So this was a simple pipeline for the whole project (pre-processing, modelling, training, seeing training curves, predictions). To improve the score, you can increase the number of epochs and make a more complex network.
# 
# I hope you liked it and feel free to fork it to make it even better. (Also, please upvote if you liked it).

# In[ ]:




