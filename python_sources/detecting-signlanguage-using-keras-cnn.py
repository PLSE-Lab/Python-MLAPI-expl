#!/usr/bin/env python
# coding: utf-8

# ## Data Augmentation
# by Gabe Wilberscheid
# 
# Often times in machine learning we will find that if we had a larger, more diverse dataset, we can build a better model. One way to get a larger dataset is by creating new data, of course, we do not want to create just random data. We need data that still makes sense. In image recognition task we can often time transform our images by slightly rotating them, shifting the brightness, or flipping the image horizontally or vertically. There are other techniques but you as the developer must choose what makes sense in your case.
# 

# In[ ]:


# load in libaries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/Sign-language-digits-dataset"))


# In[ ]:


# Lets load in the data
X = np.load('../input/Sign-language-digits-dataset/X.npy')
y = np.load('../input/Sign-language-digits-dataset/Y.npy')
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))

plt.imshow(X[700], cmap='gray')
print(y[700]) # one-hot labels starting at zero


# In[ ]:


# create a data generator using Keras image preprocessing
datagen = ImageDataGenerator(
    rotation_range=16,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12
    )


# In[ ]:


#split test and train
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=8)
# add another axis representing grey-scale
Xtest = Xtest[:,:,:,np.newaxis]
Xtrain=Xtrain[:,:,:,np.newaxis]


# In[ ]:


datagen.fit(Xtrain)


# In[ ]:


# build our CNN
model = Sequential()

# Convolutional Blocks: (1) Convolution, (2) Activation, (3) Pooling
model.add(Conv2D(input_shape=(64, 64, 1), filters=64, kernel_size=(4,4), strides=(2)))
model.add(Activation('relu'))
#outputs a (20, 20, 32) matrix
model.add(Conv2D(filters=64, kernel_size=(4,4), strides=(1)))
model.add(Activation('relu'))
#outputs a (8, 8, 32) matrix
model.add(MaxPooling2D(pool_size=4))

# dropout helps with over fitting by randomly dropping nodes each epoch
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))


# ## First Train on only the actual dataset
# Then after our model has learned, we will train the model on the dataset with our image transformations. We do this so to speed up trainning and so the model becomes more rebust to real world data.

# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adadelta(),
             metrics=['accuracy'])
model.fit(Xtrain, ytrain, batch_size=32, epochs=10)

score = model.evaluate(Xtest, ytest, verbose=0)


# In[ ]:


print('Loss: {:.4f}  Accuaracy: {:.4}%'.format(score[0],score[1]))


# As you can see we slightly overfit our trainning data. That is, we got a higher accuracy on our trainning than test data, this is normal. Now let us further train the model on transformed images.

# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adadelta(),
             metrics=['accuracy'])
model.fit_generator(datagen.flow(Xtrain, ytrain, batch_size=32),
                    steps_per_epoch=64, epochs=10)

score = model.evaluate(Xtest, ytest, verbose=0)


# In[ ]:


print('Loss: {:.4f}  Accuaracy: {:.4}%'.format(score[0],score[1]))


# ## Results
# As you can see further trainning on our augmented data increased the accuaracy by a fair amount.

# In[ ]:


test_image = Xtest[4]
test_image_array = test_image.reshape(64, 64)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

plt.imshow(test_image_array, cmap='gray')


# In[ ]:


print(np.round(result, 1))
print(ytest[4])

