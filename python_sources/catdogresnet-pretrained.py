#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os,shutil

in_path="../input"

print(os.listdir(in_path))


# In[2]:


IMAGE_SIZE=80
BATCH_SIZE=128


label=[]
data=[]
counter=0
path="../input/train/train"
for file in os.listdir(path):
    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_COLOR)
    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
    try:
        data.append(image_data/255)
    except:
        label=label[:len(label)-1]
    counter+=1
    if counter%1000==0:
         print (counter," image data retreived")

data=np.array(data)
data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],3)
label=np.array(label)
print (data.shape)
print (label.shape)


# In[3]:


sns.countplot(label)


# In[4]:


from sklearn.model_selection import train_test_split
train_data, valid_data, train_label, valid_label = train_test_split(
    data, label, test_size=0.2, random_state=42)
print(train_data.shape)
print(train_label.shape)
print(valid_data.shape)
print(valid_label.shape)


# In[5]:


from keras import Sequential
from keras.layers import *
import keras.optimizers as optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
import keras.backend as K


# In[6]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D




# In[7]:


from keras import applications
resnet_model = applications.ResNet50(weights='imagenet',
                               include_top=False,
                               input_shape=(80, 80, 3))

resnet_model.summary()


# In[8]:


from keras import layers
from keras import models
model = models.Sequential()
model.add(resnet_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation  = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


# In[9]:


model.summary()


# In[10]:


from keras import backend as K
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')
from keras import layers
from keras import models
from keras import optimizers
from keras.layers import GlobalAveragePooling2D

model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(lr=1e-4),metrics=['acc'])
# vgg_model.summary()
# train_history=vgg_model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=40,batch_size=BATCH_SIZE)


# In[11]:


train_history=model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=20,batch_size=BATCH_SIZE)


# In[12]:


import matplotlib.pyplot as plt
acc = train_history.history['acc']
val_acc = train_history.history['val_acc']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'blue', label='Training acc')
plt.plot(epochs, val_acc, 'red', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'blue', label='Training loss')
plt.plot(epochs, val_loss, 'red', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[13]:


from keras import Sequential
from keras.layers import *
import keras.optimizers as optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
import keras.backend as K
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory("../input/test1",target_size=(80, 80),batch_size=32,class_mode='binary')


# In[14]:


from tensorflow.python.keras.models import Sequential
from keras.models import load_model

print("-- Evaluate --")

scores = model.evaluate_generator(
            test_generator, 
            steps = 100)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


# In[15]:


Y_pred = model.predict(valid_data)
predicted_label=np.round(Y_pred,decimals=2)


# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mlxtend.plotting import plot_confusion_matrix

# Get the confusion matrix

CM = confusion_matrix(valid_label, Y_pred.round())
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(12, 12))
plt.xticks(range(2), ['Cat', 'Dog'], fontsize=16)
plt.yticks(range(2), ['Cat', 'Dog'], fontsize=16)
plt.show()


# In[ ]:


test_data=[]
id=[]
counter=0
for file in os.listdir("../input/test1/test1"):
    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_COLOR)
    try:
        image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))
        test_data.append(image_data/255)
        id.append((file.split("."))[0])
    except:
        print ("ek gaya")
    counter+=1



test_data=np.array(test_data)
print (test_data.shape)
test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],3)
dataframe_output=pd.DataFrame({"id":id})


# In[ ]:




