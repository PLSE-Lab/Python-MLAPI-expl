#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 

# The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
# 
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
# 
# The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

# ## Content
# 1. Introduction
# 2. Data preparation
# 3. CNN
# 4. Training and Evaluating the model
# 5. Prediction and submition

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from keras import layers
from keras import regularizers
from keras import models
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


# In[ ]:


#loading data
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# ## Data preparation 

# In[ ]:


train_images=train.iloc[:,1:].values
train_labels=train.iloc[:,0:1].values
test_X=test.iloc[:,:].values


# In[ ]:


sns.countplot(train['label'])


# In[ ]:


train_images


# In[ ]:


train_images = train_images.reshape((-1, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test = test.values.reshape(-1,28,28,1)
test = test / 255.0


# In[ ]:


train_images.shape


# In[ ]:


# Some examples
g = plt.imshow(train_images[20][:,:,0])
g=plt.title(train_labels[20])


# In[ ]:


train_labels = to_categorical(train_labels)


# In[ ]:


#using data agumenteation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(train_images)


# ## CNN

# In[ ]:


model = models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])


# ## Training and Evaluating

# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(train_images,train_labels, batch_size=64),
                              epochs = 100, 
                              verbose = 2, steps_per_epoch=train_images.shape[0] // 64
                              )


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='r', label="Training accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# ## Prediction and Submission

# In[ ]:


# predict results
results = model.predict(test)


# In[ ]:


# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")


# In[ ]:


#concatinating result with series of numbers from 1 to 28000
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:


model.save('digit_clfr.h5')


# In[ ]:




