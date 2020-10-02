#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.listdir("../input"))


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
#sets matplotlib inline and display graphs below the carasponding cell
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)
# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image


# In[3]:


train_path = '../input/training_set/training_set'
test_path = '../input/test_set/test_set'


# In[4]:


from keras.preprocessing.image import ImageDataGenerator
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['cats','dogs'],  batch_size=50)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224), classes=['cats','dogs'],batch_size=50)


# In[5]:


#plot image with labels
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
            f = plt.figure(figsize=figsize)
            cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
            for i in range(len(ims)):
                sp = f.add_subplot(rows, cols, i+1)
                sp.axis('Off')
                if titles is not None:
                    sp.set_title(titles[i], fontsize=16)
                    plt.imshow(ims[i], interpolation= None if interp else 'none')


# In[6]:


imgs, labels = next(train_batches)


# In[7]:


plots(imgs, titles=labels)


# # Build a Train Model

# In[8]:


#preprocessing
from keras.preprocessing.image import ImageDataGenerator

#dl libraries
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

#specify for cnn
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


# In[9]:


model = Sequential([
    Conv2D(32,(3,3), activation = 'relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(2, activation='softmax')
])
#224 size, 3 = rgb color scale
#classifier = sequential
#32,3,3--> 32 filter with 3*3 for each filter.
#start with 32 filters, and then create more layers with 64,128,256.
#224,224,3---> 3 color channel(RGB), 224*224 pixel. 
# but when use cpu, 3,64 use, due to computational limit 
# flatten: creating a long vector. 


# In[10]:


model.summary()


# In[11]:


#optimization
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# loss =''how to find loss


# In[ ]:


model.fit_generator(train_batches, steps_per_epoch=50, #(80*100)
                   validation_data=test_batches, validation_steps=50, epochs=20, verbose=2)


# # Prediction

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


test_imgs, test_labels = next(test_batches)


# In[ ]:


plots(test_imgs, titles=test_labels)


# In[ ]:


test_labels = test_labels[:,0]
test_labels


# In[ ]:


predictions = model.predict_generator(test_batches, steps=1, verbose=0)
  # 10 sample of test images then steps=1


# In[ ]:


predictions


# In[ ]:


# how mapping done?
test_batches.class_indices


# # Confusion matrix

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


# In[ ]:


cm = confusion_matrix(test_labels, predictions[:,0])# only gives first array


# In[ ]:


def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap = plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.Sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
        print(cm)
        
        thresh = cm.max() /2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i,j],
                    horizontalalignment='center',
                    color = 'white' if cm[i,j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Prediction label')


# In[ ]:


cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm, cm_plot_labels, title= 'Confusion_matrix')


# # VGG16 Model

# In[ ]:


import keras
from keras.applications.vgg16 import VGG16, preprocess_input


# In[ ]:


vgg16_model = keras.applications.VGG16()


# In[ ]:


vgg16_model.summary()


# In[ ]:


type(vgg16_model)


# In[ ]:


model = Sequential()
for layer in vgg16_model.layers[:-1]:   #exclude last output layer
    model.add(layer)


# In[ ]:


model.summary()


# In[ ]:


model.layers.pop(0)   # last layer is gone : because in theie output is 1000 we requred only 2


# In[ ]:


model.summary()


# In[ ]:


for layer in model.layers:
    layer.trainable = False


# In[ ]:


model.add(Dense(2, activation= 'softmax'))


# In[ ]:


model.summary()


# # Train the Fine-Tuned VGG-16 Model

# In[ ]:


model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit_generator(train_batches, steps_per_epoch=4,
                   validation_data = valid_batches, validation_steps=4, epochs=5, verbose=2)


#  # Visualising Accuracy and loss w.r.t. the Epochs

# In[ ]:


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)


# In[ ]:


test_labels = test_labels[:,0]
test_labels


# In[ ]:


predictions = model.predict_generator(test_batches, steps=1, verbose=0)


# In[ ]:


cm = confusion_matrix(test_labels, np.round(predictions[:,0]))


# In[ ]:


def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap = plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.Sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
        print(cm)
        
        thresh = cm.max() /2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i,j],
                    horizontalalignment='center',
                    color = 'white' if cm[i,j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Prediction label')


# In[ ]:


cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm, cm_plot_labels, title='confusion_matrix')


# In[ ]:




