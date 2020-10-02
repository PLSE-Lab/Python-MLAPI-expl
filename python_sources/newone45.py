#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import matplotlib.pyplot as plt
import itertools
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import matplotlib.image as mpimg
from keras.preprocessing import image
import imageio as im
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
from keras import models
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing import image
from keras.layers import Dropout
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[2]:


train_path = '/kaggle/input/augdata/train/train'


# In[3]:


pwd


# In[4]:


train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',
                                                                                                 'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',
                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',
                                                                                                 'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10)


# In[5]:


def plots(ims, figsize=(12,6), rows=5, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3): ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# In[6]:


img,labels = next(train_batches)


# In[7]:


plots(img,titles=labels)


# In[8]:


model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=(128,128,3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu')) 

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Conv2D(64,(3,3),padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu')) 

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(38,activation='softmax'))
model.build()


# In[9]:


model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_batches,steps_per_epoch=3536,epochs=20)


# In[ ]:


a = model.get_weights()
model.save("NEW45.h5")


# In[ ]:


test_path = '/kaggle/input/plantvillagepredictions/val/val'


# In[ ]:


test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',
                                                                                                 'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',
                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',
                                                                                                 'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10,shuffle=False)


# In[ ]:


output = model.predict_generator(test_batches,steps=440)


# In[ ]:


for i in range(0,4396):
    for j in range(0,38):
        if output[i][j] == max(output[i]):
            print(j)


# In[ ]:


model.save('newone11.h5')


# In[ ]:


model.save_weights('any.json')


# In[ ]:


a = model.get_weights()


# In[ ]:


b = test_batches.classes
b


# In[ ]:


count = 0
for i in range(0,4396):
    for j in range(0,38):
        if output[i][j] == max(output[i]):
            if j == b[i]:
                count+=1
print(count)


# In[ ]:


count/4396


# In[ ]:


model.save('newone.h5')


# In[ ]:


ls


# In[ ]:


from keras.models import load_model


# In[ ]:


model = Sequential()


# In[ ]:


model = load_model('newone.h5')


# In[ ]:


ls -al


# In[ ]:


cd input


# In[ ]:


ls


# In[ ]:


cd augdata/


# In[ ]:


ls


# In[ ]:


cd train/train


# In[ ]:


ls


# In[ ]:


a = test_batches.classes
a


# In[ ]:


out = np.zeros((4396,1))


# In[ ]:


count = 0
for i in range(0,4396):
    for j in range(0,38):
        if output[i][j] == max(output[i]):
            out[i] = j
            if j == a[i]:
                
                count+=1
print(count)


# In[ ]:


out


# In[ ]:


count/4396


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[ ]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


# class_names = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
# class_names.shape


# In[ ]:


# type(class_names[0])


# In[ ]:


# plot_confusion_matrix(out, a, classes=class_names,title='Confusion matrix, without normalization')


# In[ ]:


class_names = np.zeros((38,1))
type(class_names)


# In[ ]:


for i in range(0,38):
    class_names[i] = int(i)


# In[ ]:


class_names


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[ ]:


print(f1_score(out,a, average="macro"))


# In[ ]:


print(precision_score(out, a, average="macro"))
print(recall_score(out,a, average="macro"))  


# In[ ]:


ls


# In[ ]:


model.save('99%Accurate.h5')


# In[ ]:


model.get_weights()


# In[ ]:


ls


# In[ ]:


get_ipython().system('wget NEW40.h5')


# In[ ]:


get_ipython().system('pip install kaggle-cli')


# In[ ]:


# !kg download NEW40.h5


# In[ ]:


get_ipython().system('chmod 777 newonw.h5')


# In[ ]:


ls -l


# In[ ]:


get_ipython().system('wget newone.h5')


# In[ ]:


ls

