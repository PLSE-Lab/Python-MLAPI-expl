#!/usr/bin/env python
# coding: utf-8

# # Cactus Classification

# In[ ]:


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


# ## Data Prep

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


from tqdm import tqdm
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Activation,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from PIL import Image


# In[ ]:


import pandas as pd
import cv2
import os

def load_imgs(path):
    imgs = {}
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        imgs[f] = cv2.imread(fname)
    return imgs

img_train = load_imgs('../input/train/train/')
img_test = load_imgs('../input/test/test/')


# Reading from the train folder, we normalized the images

# In[ ]:


train_csv = pd.read_csv('../input/train.csv')
import numpy as np

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    X_train.append(img_train[row['id']]/255)
    Y_train.append(int(row['has_cactus']))

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array([img_test[f] for f in img_test])

print('Training data shape:', X_train.shape, '=>', Y_train.shape)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["axes.grid"] = False


# Plotting the normalized images- showing three images with cactus and two without cactus

# In[ ]:


fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(X_train[0])
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(X_train[1])
axes[1].set_title("Has cactus:" + str(Y_train[0]))
axes[2].imshow(X_train[2])
axes[2].set_title("Has cactus:" + str(Y_train[0]))
axes[3].imshow(X_train[1000])
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(X_train[1050])
axes[4].set_title("Has cactus:" + str(Y_train[1050]))


# ## Image preprocessing using Guassian filtering
# For sharpening the edges of cactus in the pixelated images, we applied Gaussian filtering

# In[ ]:


from scipy.ndimage import gaussian_filter

def img_sharpen(img):
    blurred_f = gaussian_filter(img, 2)

    filter_blurred_f = gaussian_filter(blurred_f, 2)

    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return sharpened


# In[ ]:


#Sharpen the low quality cactus images
sharp_img_xtrain = []

for im in X_train:
    sharp_img_xtrain.append(img_sharpen(im))


# Plotting the same images as above except with the sharpening filter using Gaussian filtering

# In[ ]:


fig, axes = plt.subplots(1, 5, figsize=(15, 4))
axes[0].imshow(sharp_img_xtrain[0])
axes[0].set_title("Has cactus:" + str(Y_train[0]))
axes[1].imshow(sharp_img_xtrain[1])
axes[1].set_title("Has cactus:" + str(Y_train[0]))
axes[2].imshow(sharp_img_xtrain[2])
axes[2].set_title("Has cactus:" + str(Y_train[0]))
axes[3].imshow(sharp_img_xtrain[1000])
axes[3].set_title("Has cactus:" + str(Y_train[1000]))
axes[4].imshow(sharp_img_xtrain[1050])
axes[4].set_title("Has cactus:" + str(Y_train[1050]))


# ## Train/Test Split for CNN

# In[ ]:


from sklearn.model_selection import train_test_split
from numpy import array

sharp_xtrain = array(sharp_img_xtrain)
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)


# Defining a class for pooling a layer using Mulitple Learning Instance. 
# Inspire by article on classifying images using deep multiple instance learning: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/pdf/btw252.pdf

# In[ ]:


import tensorflow as tf

class noisyand(tf.keras.layers.Layer):
    def __init__(self, num_classes, a = 20, **kwargs):
        self.num_classes = num_classes
        self.a = max(1,a)
        super(noisyand,self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name = "b",shape = (1,input_shape[-1].value), initializer = "uniform",trainable = True)
        super(noisyand,self).build(input_shape)

    def call(self,x):
        mean = tf.reduce_mean(x, axis = [1,2])
        return (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


# Defining the Deep CNN model

# In[ ]:


def define_model(input_shape= (32,32,3), num_classes=1):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding = 'same',
                     input_shape=input_shape))
    
    model.add(Conv2D(64, (3, 3), padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    
    model.add(noisyand(num_classes+1))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    return model


# In[ ]:


model = define_model()
model.summary()


# In[ ]:


model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                  optimizer=tensorflow.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])


# In[ ]:


epoch=15
history = model.fit(x_train, y_train,
         batch_size=32,
         epochs=epoch,
         verbose=1,
         validation_data=(x_test, y_test))


# Plotting the accuracy scores of the model

# In[ ]:


acc=history.history['acc']
epochs_=range(0,epoch)
plt.plot(epochs_,acc,label='training accuracy')

acc_val=history.history['val_acc']
plt.scatter(epochs_,acc_val,label="validation accuracy")
plt.ylim([0.85,1.0])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot of Model')

plt.legend()


# Plotting the loss scores of the model

# In[ ]:


acc=history.history['loss']
epochs_=range(0,epoch)
plt.plot(epochs_,acc,label='training loss')

acc_val=history.history['val_loss']
plt.scatter(epochs_,acc_val,label="validation loss")
plt.ylim([0,0.5])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot of Model')

plt.legend()


# ## Submission Part

# In[ ]:


submission_set=pd.read_csv('../input/sample_submission.csv')
submission_set.head()


# Predicting images has cactus for submission

# In[ ]:


predictions=np.empty((submission_set.shape[0],))
    
for n in tqdm(range(submission_set.shape[0])):
    data=np.array(Image.open('../input/test/test/'+submission_set.id[n]))
    data=data.astype(np.float32)/255
    #Sharpen the low quality cactus images
    data=img_sharpen(data)
    predictions[n]=model.predict(data.reshape((1,32,32,3)))[0]

    
submission_set['has_cactus']=predictions
submission_set.to_csv('sample_submission.csv',index=False)

submission_set.head()


# ## ROC Curve

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

clf=model
y_pred_proba = clf.predict_proba(x_test)
y_pred = clf.predict_classes(x_test)

fpr,tpr,_= roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

