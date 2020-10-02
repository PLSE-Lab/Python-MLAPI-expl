#!/usr/bin/env python
# coding: utf-8

# # Using VGG 16 for Blood Cell Classification

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.getcwd())
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


from keras.models import Sequential,Model
from keras.layers import Activation
from keras.layers.core import Dense,Flatten,Dropout
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, RMSprop , SGD
import keras 
import keras.backend
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

import os
from keras import regularizers


# In[5]:


cd "../input/dataset2-master/dataset2-master/"


# In[6]:


train_path = "images/TRAIN"
test_path = "images/TEST"


# ### Sample NEUTROPHIL images belonging to class 0

# In[13]:


plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    path = os.listdir(train_path + '/NEUTROPHIL')[i]
    img = cv2.imread(train_path + '/NEUTROPHIL' + '/' + path)
    plt.imshow(img)
    plt.title('NEUTROPHIL: 0')
    plt.tight_layout()
plt.show()


# ### Sample MONOCYTE images belonging to class 1

# In[14]:


plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    path = os.listdir(train_path + '/MONOCYTE')[i]
    img = cv2.imread(train_path + '/MONOCYTE' + '/' + path)
    plt.imshow(img)
    plt.title('MONOCYTE: 1')
    plt.tight_layout()
plt.show()


# ### Sample EOSINOPHIL images belonging to class 2

# In[15]:


plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    path = os.listdir(train_path + '/EOSINOPHIL')[i]
    img = cv2.imread(train_path + '/EOSINOPHIL' + '/' + path)
    plt.imshow(img)
    plt.title('EOSINOPHIL: 2')
    plt.tight_layout()
plt.show()


# ### Sample LYMPHOCYTE images belonging to class 3

# In[16]:


plt.figure(figsize = (12,12))
for i in range(4):
    plt.subplot(1, 4, i+1)
    path = os.listdir(train_path + '/LYMPHOCYTE')[i]
    img = cv2.imread(train_path + '/LYMPHOCYTE' + '/' + path)
    plt.imshow(img)
    plt.title('LYMPHOCYTE: 3')
    plt.tight_layout()
plt.show()


# ### Set the dimensions of the input images

# In[ ]:


inputs = (240,320,3)


# ### Loading the VGG16 Model Pre trained on ImageNet weights

# In[ ]:


vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs)


# ### Freeze the last 4 layers

# In[ ]:


model = Sequential()
for layer in vgg.layers[:-4]:
  layer.trainable=False


# ### Add 4 dense fully connected layers at the end

# In[ ]:


model.add(vgg)
 
# Add new layers
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32, activation='relu' ))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu' ))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu' ))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))
model.summary()


# ### Create a data Image Generator

# In[ ]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest', validation_split=0.25)


validation_datagen = ImageDataGenerator(rescale=1./255 )
 
# Change the batchsize according to your system RAM
train_batchsize = 32
val_batchsize = 32


# ### Split data into train and test data

# In[ ]:


train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(240, 320),
        batch_size=train_batchsize,
        class_mode='categorical', subset = "training")
 
validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(240, 320),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False,
    subset='validation')


# In[ ]:


print(len(train_generator[0][1]))


# ### Compile the model with an appropriate loss function

# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])


# ### Fit the compiled model on our training data

# In[ ]:


history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples//train_generator.batch_size ,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples//validation_generator.batch_size,
      verbose=1)


# ### Print the graph of accuracy vs epochs and loss vs epochs

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# ### Resuts of the best epoch

# In[ ]:


print("The best Training accuracy {}".format(max(acc)*100))
print("The best validation accuracy {} ".format(max(val_acc)*100))


# ### Printing the predicted labels for each image

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict_generator(validation_generator, 2487 // 20 + 1)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

