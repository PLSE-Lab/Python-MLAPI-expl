#!/usr/bin/env python
# coding: utf-8

# # **Deep Learning of Xray Images Using Transfer Learning (ImageNetv2) **
# 
#    Deep learning in the medical field is a fast growing field. This is an important stepping stone to move to a new era of practising medicine, and that it by the help of machines and computers to increase physician, radiologist or even surgeons' accuracy in areas such as medical prognosis, medical diagnosis, treatment planning and public health intervention. The basic units of machine learning actually closely resemble the long-used biostatistics modelling of medical researches such as logistic regression, linear regression and even Cox time-based regression. 
#     
#    Some of the more interesting researches on Deep Learning diagnosing diseases can be read here:
# *     [CheXNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686) - Chest XRay 
# *     [Dermatology](https://www.nature.com/articles/nature21056) - Detecting skin cancers
# *     [Opthamology](https://www.ncbi.nlm.nih.gov/pubmed/27898976) - Fundus image 
# *     [Pathology](https://jamanetwork.com/journals/jama/fullarticle/2665774) - Microscope images 

#  However it is important to note that a machine is not always perfect. Therefore, machine learning must only be used as an adjunct by the medical profesionals instead of relying solely on it (for the time being, that is). Hopefully in the future we will these models being deployed to real world situation in order to reduce the burden of the physician, increase patient-care quality, reduce medical expenditure as well as making healthcare processes such as diagnosing and calculating risks faster. 
#    
#    Credits to [deeplearning.ai](http://deeplearning.ai) team for great courses.

# A little bit more on the model that I have used. I am using pre-trained ImageNet/MobileNetv2 archictecture that has been trained with millions of images. I am only using the weightage from the earlier layers that learns the simpler parameters of an image, and using those learned weightages/biases on the Xray images. 
# 
# Here is the ImageNet model:
# ![ImageNet layers](https://www.researchgate.net/profile/Sehla_Loussaief/publication/325772980/figure/fig2/AS:673660541079552@1537624597820/ImageNet-CNN-layers-Fig-2-demonstrates-the-different-network-layers-required-by-the.png)
# 
# Please note that I have removed the last few layers and replaced it with AveragePooling2D and Dense layers with L2 regularization. I needed to use the regularizer because there was too much discrepancy between traning set accuracy and validation/test set accuracy.
# 
# Here are the accuracy I have achieved. (Note that I ran into some RAM issues and had to stop the learning early, if more resources were available, or if there are ways to get pass this/more tuning of hyperparameters was done,maybe the accuracy can be improved)
# 
# **Traning set:** 0.9053 (the higher the epochs, the higher the accuracy - true for all sets - up to 0.9500 // num epochs limited by ram)
# 
# **Validation set:**  0.8356
# 
# **Test set:** 0.8500
# 
# 
# This is my first Kaggle Notebook! Comment any improvements you might want to see, thank you!
# 
# *This is a work in progress, v1.00
# *

# Lets get to the code:
# 
# Load all the functions needed

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('fivethirtyeight')
import pickle 
import os 
import cv2 
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Getting data directiories,
# the train, val and test directory has been splitted.

# In[ ]:


data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
test_dir = data_dir + '/test'


# Data loading code was borrowed and modified from [therealcyberlord](http://https://www.kaggle.com/therealcyberlord)

# Basically we get the files, convert it into RGB (changes 1 channel to 3), and append the file and label to data array

# In[ ]:


labels = ['NORMAL', 'PNEUMONIA']
img_size = 200
def get_files(file_dir):
  data = []
  for label in labels: 
      path = os.path.join(file_dir, label)
      class_num = labels.index(label)
      for img in os.listdir(path):
          try:
              img_arr = cv2.imread(os.path.join(path, img))
              img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
              resized_arr = cv2.resize(img_arr, (img_size, img_size))
              data.append([resized_arr, class_num])
          except Exception as e:
              print(e)
  return np.array(data)


# In[ ]:


train = get_files(train_dir)
val = get_files(val_dir)
test = get_files(test_dir)

print('Done loading data')


# Lets see some example of xrays:

# In[ ]:


plt.imshow(train[1000][0])
plt.axis('off')
print(labels[train[1000][1]])


# In[ ]:


Splitting to training, dev, test sets


# In[ ]:


X = []
y = []

for i,j in train:
  X.append(i)
  y.append(j)

for i,j in val:
  X.append(i)
  y.append(j)
  
for i,j in test:
  X.append(i)
  y.append(j)


# In[ ]:


X = np.array(X).reshape(-1, 200, 200, 3)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=32)


# In[ ]:


#normalize

X_train = X_train/255
X_val = X_val/255
X_test = X_test/255


# 1. Need to delete these because to optimize RAM usage.

# In[ ]:


del train
gc.collect()
del val
gc.collect()
del test
gc.collect()
del X
gc.collect()
del y
gc.collect()


# Datagenerator.

# In[ ]:


datagen = ImageDataGenerator(rotation_range=90, shear_range=0.1, zoom_range=0.1, width_shift_range=0.1,height_shift_range=0.1)

#no vertical and horizontal flip, because it will cause false real world results.

datagen.fit(X_train)


# Here's the fun part, loading MobileNet v2, note that we set include_top = False to not include the top layers

# In[ ]:


IMG_SHAPE = (200,200,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# Set trainable = False because we want to use the weights from the model.

# In[ ]:


base_model.trainable = False

base_model.summary()


# Adding a few stuffs here.
# 1. GlobalAveragePooling2D layer with regularizer.
# 2. Dense layer also with regularize.
# 3. I have printed the model summary to see the whole layers.

# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D(activity_regularizer=tf.keras.regularizers.l2(0.05))
prediction_layer = tf.keras.layers.Dense(1, activity_regularizer=tf.keras.regularizers.l2(0.05), activation= tf.keras.activations.sigmoid)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer, 
])

model.summary()


# In[ ]:


base_learning_rate = 0.0005
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


len(model.trainable_variables)


# Here's where the RAM problem start, if you have any suggestions please drop in the comments section below.
# 
# The model stopped after 11 epochs, so I had to run it again another time due to the RAM issues.
# 
# Maybe greater accuracy could be achieved if I could run higher number of epochs, also please not that I could include the confusion matrix and the curves as well due to this issue. I will try to update these in future versions.

# In[ ]:


initial_epochs = 20


# In[ ]:


early_stop = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
#history = model.fit(datagen.flow(X_train, y_train, batch_size=30), callbacks=[early_stop], validation_data=(X_val, y_val), epochs=initial_epochs)


# In[ ]:


additional_epochs = 5


# In[ ]:


#history = model.fit(datagen.flow(X_train, y_train, batch_size=30), callbacks=[early_stop], validation_data=(X_val, y_val), epochs=additional_epochs)


# In[ ]:


#this cannot be run, however I have done it and it ranges around 85 to 90, depending on the epoch num.
#model.evaluate(X_test, y_test)


# In[ ]:




