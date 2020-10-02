#!/usr/bin/env python
# coding: utf-8

# ### This is a basic Keras model for newbies like me to learn. Please upvote if you find it helpful.

# In[ ]:


import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

trainpath = '../input/train/'
testpath = '../input/test/'

print('# of training files: ' + str(len(os.listdir(trainpath))))
print('# of testing files: ' + str(len(os.listdir(testpath))))


# In[ ]:


# Preview labels
train_labels = pd.read_csv('../input/train_labels.csv')
print(train_labels.head())


# In[ ]:


# Preview a noninvasive plant image
from skimage import io, transform
import matplotlib.pyplot as plt

sample_image = io.imread(trainpath + '1.jpg')
print('Height:{0} Width:{1}'.format(sample_image.shape[0], sample_image.shape[1]))
plt.imshow(sample_image)


# In[ ]:


# Preview an invasive plant image
sample_image = io.imread(trainpath + '3.jpg')
plt.imshow(sample_image)


# In[ ]:


# There is one image in the test set that has different dimensions.
# It may just need a rotation, but I'm going to ignore it for now.
print(io.imread(testpath + '1068.jpg').shape)


# In[ ]:


# Check that input_shape = (batch_size, rows, columns, channels)
from keras.backend import image_data_format
print(image_data_format())


# In[ ]:


# Kernel memory is limited so I'm using 100 images each for training and validation 
# and scaling them down to 150x200 pixels to keep things simple.

x_train = np.empty(shape=(300, 150, 200, 3))
y_train = np.array(train_labels.invasive.values[0:300])
x_val = np.empty(shape=(300, 150, 200, 3))
y_val = np.array(train_labels.invasive.values[300:600])

for i in range(300):
    tr_im = io.imread(trainpath + str(i+1) + '.jpg')
    x_train[i] = transform.resize(tr_im, output_shape=(150, 200, 3))

for i in range(300):
    val_im = io.imread(trainpath + str(i+101) + '.jpg')
    x_val[i] = transform.resize(val_im, output_shape=(150, 200, 3))


# In[ ]:


# Starting architecture
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(150, 200, 3)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=SGD(lr=1e-5, momentum=0.75, nesterov=False), 
              loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# Look at how tensors affect output shape
print(model.summary())


# In[ ]:


# One epoch for demonstration purposes
model.fit(x_train, y_train, epochs=1, batch_size=20)


# In[ ]:


acc = model.evaluate(x_val, y_val)[1]
print('Evaluation accuracy:{0}'.format(round(acc, 4)))


# ### More coming soon...
