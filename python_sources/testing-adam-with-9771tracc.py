#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


cd ..


# In[3]:


import numpy as np
import tensorflow 
import keras
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
from keras.models import load_model


# In[4]:


model = Sequential()


# In[5]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[6]:


model = load_model('/kaggle/input/finalmodel/NewOne5084acc993v121.h5')


# In[7]:


train_path = '/kaggle/input/augdata/train/train'


# In[9]:


train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(128,128),classes =['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10',
                                                                                                 'c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20',
                                                                                                    'c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30',
                                                                                                 'c_31','c_32','c_33','c_34','c_35','c_36','c_37'],batch_size=10,shuffle=False)


# In[11]:


output = model.predict_generator(train_batches,steps=3536)


# In[14]:


import numpy as np
a = train_batches.classes
a.shape


# In[21]:


out = np.zeros((35360,1))
out.shape


# In[22]:


count = 0
for i in range(0,35360):
    for j in range(0,38):
        if output[i][j] == max(output[i]):
            out[i] = j
            if j == a[i]:    
                count+=1
print(count)


# In[16]:


count/35360


# In[17]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[18]:


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


# In[19]:


class_names = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
class_names.shape


# In[25]:


plot_confusion_matrix(out, a, classes=class_names,title='Confusion matrix, without normalization')


# In[24]:


for i in range(0,38):
    class_names[i] = int(i)


# In[26]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[27]:


print(f1_score(out,a, average="macro"))


# In[28]:




print(precision_score(out, a, average="macro"))
print(recall_score(out,a, average="macro"))  


# In[ ]:




