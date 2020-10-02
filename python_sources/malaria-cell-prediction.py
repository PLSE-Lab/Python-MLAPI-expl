#!/usr/bin/env python
# coding: utf-8

# In[52]:


#Importing all necesary library

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import fnmatch
import keras
from time import sleep
import itertools
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Dense,Input, TimeDistributed, GlobalAveragePooling2D,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D, MaxPooling1D,Activation
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as k

get_ipython().run_line_magic('matplotlib', 'inline')


# **1. Process the data**

# In[70]:


base_dir = os.path.join("../input/cell_images/cell_images")
imagePatches_0 = glob(base_dir+'/Uninfected/*.png', recursive=True)
imagePatches_1 = glob(base_dir+'/Parasitized/*.png', recursive=True)
print(len(imagePatches_0))
print(len(imagePatches_1))


# **2. Image Process**

# In[74]:


infected = os.listdir('../input/cell_images/cell_images/Parasitized/')
uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')
plt.figure(figsize=(12,12))
for i in range(6):
    plt.subplot(1,6,i+1)
    img = cv2.imread('../input/cell_images/cell_images/Parasitized/'+infected[i])
    plt.imshow(img)
    plt.title('infected')
plt.figure(figsize=(12,12))
for i in range(6):
    plt.subplot(1,6,i+1)
    img2 = cv2.imread('../input/cell_images/cell_images/Uninfected/'+uninfected[i])
    plt.imshow(img2)
    plt.title('uninfected')


# In[3]:


x =  []
y = []

for img in imagePatches_0:
    full_size_image = cv2.imread(img)
    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)
    x.append(im)
    y.append(0)
    
for img in imagePatches_1:
    full_size_image = cv2.imread(img)
    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)
    x.append(im)
    y.append(1)
    
x = np.array(x)
y = np.array(y)


# In[4]:


print(x.shape)
print(y.shape)


# **3. Split the data into tain and test**

# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state=101)
y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

del x, y


# In[12]:


x_train.shape


# **4. Design the model**

# In[22]:


def my_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(7,7), input_shape=(224,224,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model
model = my_model()
model.summary()


# **5. Train the model for 20 epochs**

# In[24]:


mcp = ModelCheckpoint(filepath= 'model_check_path.hdf5', monitor='val_acc', save_best_only=True,
                      save_weights_only=False)
hist = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2, callbacks=[mcp])


# **6. Define some visualization function**

# In[59]:


def plot_model_acc(val_acc, train_acc):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_facecolor('w')
    ax.grid(b=False)
    ax.plot(train_acc, color='red')
    ax.plot(val_acc, color = 'green')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc = 'lower right')
    plt.show()

def plot_model_loss(val_loss, train_loss):   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_facecolor('w')
    ax.grid(b=False)
    ax.plot(train_loss, color='red')
    ax.plot(val_loss, color='green')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm)/float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.grid(b=False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max()/1.5 if normalize else cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment='center',
                    color='white' if cm[i,j] > thresh else 'black')
        else:
            plt.text(j, i, "{:,}".format(cm[i,j]),
                    horizontalalignment='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# **7. Result Analisis**

# In[55]:


from sklearn.metrics import classification_report
pred = model.predict(x_valid)
print(classification_report(np.argmax(y_valid, axis=1), np.argmax(pred, axis=1)))


# In[47]:


val_acc = hist.history['val_acc']
train_acc = hist.history['acc']
Visualize.plot_model_acc(val_acc, train_acc)


# In[48]:


val_loss = hist.history['val_loss']
train_loss = hist.history['loss']
plot_model_loss(val_loss=val_loss, train_loss=train_loss)


# In[49]:


model.load_weights('model_check_path.hdf5')


# In[63]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(pred, axis=1))
plot_confusion_matrix(cm=cm, normalize=False, cmap='Reds',
                      target_names=['Uninfected', 'Parasitized'],
                      title = 'Confusion Matrix')

