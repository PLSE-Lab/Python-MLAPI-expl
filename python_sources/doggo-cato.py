#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.


# In[ ]:


train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train'
test_dir = '../input/dogs-vs-cats-redux-kernels-edition/test'

train_dogs = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i] # get dog images
train_cats = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i] # get cat images

test_imgs = ['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)] # get test images

train_imgs = train_dogs[:2000] + train_cats[:2000] #slicing the dataset and using 2000 imges from each class

random.shuffle(train_imgs) # shuffle the images randomly

#delete the extra stuff to clean up the memory
del train_dogs
del train_cats
# collect garbage to save memory
gc.collect()



# In[ ]:


import matplotlib.image as mpimg

for ima in train_imgs[0:3]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()


# In[ ]:


# Lets declare image dimensions and reshape the images
# 3 channles for R G B
nrows = 150
ncols = 150
channels = 3

# declare a function to read and process our images to a suitable format

def read_and_process_image(list_of_images):
    X = [] #images
    y = []#labels
    
    for image in list_of_images:
        #print(image)
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncols), interpolation=cv2.INTER_CUBIC))
        if 'dog.' in image:
            y.append(1)
        elif 'cat.' in image:
            y.append(0)
        
        
    return X,y


# In[ ]:


# Processing pur images

X,y = read_and_process_image(train_imgs)


# In[ ]:


plt.figure(figsize=(20,10))
columns =5
for i in range (columns):
    plt.subplot(5 / columns+1, columns, i+1)
    plt.imshow(X[i])


# In[ ]:


# Just to check if we really have 0's and 1's labelled properly
import seaborn as sns
del train_imgs
gc.collect()

# convert list to numpy array
X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Labels for cats and dogs')


# In[ ]:


print('Shape of train images is ',X.shape )
print('Shape of labels is ',y.shape)


# In[ ]:


# Preparing train and dev sets using sklearn library
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev =  train_test_split(X, y, test_size=0.20, random_state=2)

print('Shape of train images is: ',X_train.shape)
print('Shape of dev train images is: ',X_dev.shape)
print('Shape of train labels is: ', y_train.shape)
print('Shape of dev labels is: ',y_dev.shape)


# In[ ]:


del X
del y
gc.collect()

# get the length of train and dev sets
ntrain = len(X_train)
ndev = len(X_dev)
# we will use a batch size of 32
batch_size = 32


# In[ ]:


# We will be using ConvNets (CNN's) with Keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


# In[ ]:


# We will be using the nn architecture of vggnet - Sequetial model

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(65, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) # Dropout for regularization
model.add(layers.Dense(512, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid')) # Sigmoid at the end because, binary classification
model.summary()


# In[ ]:


# We'll use RMS prop optimizer with a learning rate of 0.0001
# We'll use binary_crossentropy loss function cuz its a binary classification
# We'll use accuracy (acc) metric to evaluate the performance after training since it's a classification problem

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# In[ ]:


# we are going to create 2 ImageDataGenerators -> Train, dev # Converts IMG to RGB and floating points to tensors -> easy to feed to a NN
# Augmeting train_dataset: this helps preventingg overfitting, since we are using a relatively small dataset ## Normalization ##
train_datagen = ImageDataGenerator(rescale=1./255, #Scale the image b/w 0 and 1 --> norm
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                  )
dev_datagen = ImageDataGenerator(rescale=1./255) # No data augmentation in dev set, only rescaling


# In[ ]:


# Create Image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
dev_generator = dev_datagen.flow(X_dev, y_dev, batch_size=batch_size)


# In[ ]:


# The Training
# Lets train for epochs and 100 steps per epoch and we use .fit() for training

history = model.fit_generator(train_generator,
                             steps_per_epoch = ntrain // batch_size,
                             epochs=100,
                             validation_data = dev_generator,
                             validation_steps = ndev // batch_size)


# In[ ]:


#Save the model to reuse it later
model.save_weights('model_weights.h5')
model.save('model_keras.h5')


# In[ ]:


#lets plot graphs to get insights into accuracy and loss
#get details from history object

acc = history.history['acc']
dev_acc = history.history['val_acc']
loss = history.history['loss']
dev_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

#train and dev accuaracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, dev_acc, 'r', label='Dev accuracy')
plt.title('Training and Dev accuracy')
plt.legend()
plt.figure()

#train and dev loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, dev_loss, 'r', label='Dev loss')
plt.title('Training and Dev loss')
plt.legend()

plt.show()


# In[ ]:


#Now lets predict first 10 images from test set
X_test, y_test = read_and_process_image(test_imgs[70:80]) #y_test will be empty here cuz, test set has no label
x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[ ]:


# Iterate through the images in test set and make predictions
i=0
text_labels=[]
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(5 / columns+1, columns, i+1)
    plt.title('This is a '+text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i%10==0:
        break
plt.show()

