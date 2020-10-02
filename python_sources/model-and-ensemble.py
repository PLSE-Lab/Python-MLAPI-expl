#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# HAM10000 dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. 
# 
# Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: 
# 
# Actinic keratoses  
# 
# Bowen's disease (akiec), 
# 
# basal cell carcinoma (bcc), 
# 
# benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl),
# 
# dermatofibroma (df),
# 
# melanoma (mel), 
# 
# melanocytic nevi (nv) and 
# 
# vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
# 
# **Original Data Source
# **
# 
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018). doi: 10.1038/sdata.2018.161

# In this notebook, I aim to use the HAM10000 Dataset to classify 7 types of skin diseases.
# 
# The accuracy reached 84-85% quite more than several previous work

# In[ ]:


os.listdir('../input/resnet50')


# In[ ]:


import os
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
import datetime
import sys
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,Dropout,MaxPool2D
from keras.applications import ResNet50
from keras import optimizers
import math


# # 1 Constructing a model**

# Here we construct a model based on pretrained Resnet50 model. In  order to prevent overfitting we had 2 Dropout layers. 
# 
# Since we have 7 class, we use a Dense of 7 neurons

# In[ ]:


base_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      include_top=False, input_shape=(224, 224, 3))
model=Sequential()
model.add(base_model)
# Freeze the layers except the last 4 layers
model.add(Dropout(0.40))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# In[ ]:


model.summary()


# # 2: Constructing Data Flow

# Reading filenames of images and their label

# In[ ]:


df=pd.read_csv('../input/mnist1000-with-one-image-folder/HAM10000_metadata.csv')
df['file_name']=df['image_id']+'.jpg'
df=df[['file_name','dx','lesion_id']]
df.head()


# Next, we split the data into training, validation and test set with stratitied methods

# In[ ]:


from sklearn.model_selection import train_test_split
label_dataframe=df.pop('dx').to_frame()
X_train, X_test, y_train, y_test = train_test_split(df, label_dataframe, test_size=0.2, random_state=42)
X_train,X_val,y_train,y_val=train_test_split(X_train, y_train, test_size=0.25, random_state=42)
print(X_val.shape)
print(X_train.shape)
print(X_test.shape)


# In[ ]:





# Concat labels with filenames

# In[ ]:


train=pd.concat([X_train,y_train],axis=1)
train.head()
val=pd.concat([X_val,y_val],axis=1)
val.head()
test=pd.concat([X_test,y_test],axis=1)
test.head()


# Convert string labels to int labels

# In[ ]:


from sklearn import preprocessing
vle = preprocessing.LabelEncoder()
vle.fit(val['dx'])
label=vle.transform(val['dx']) 
print(list(vle.classes_))
val['label']=label
print(train.head())
le_name_mapping = dict(zip(vle.classes_, vle.transform(vle.classes_)))
print(le_name_mapping)


# In[ ]:


trle = preprocessing.LabelEncoder()
trle.fit(train['dx'])
label=trle.transform(train['dx']) 
print(list(trle.classes_))
train['label']=label
print(train.head())
le_name_mapping = dict(zip(trle.classes_, trle.transform(trle.classes_)))
print(le_name_mapping)


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(test['dx'])
label=le.transform(test['dx']) 
print(list(le.classes_))
test['label']=label
print(test.head())
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


# Data Augmentation and prepare the flow of data

# In[ ]:


train_generator = ImageDataGenerator(
rescale = 1./255,
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
vertical_flip=False)  # randomly flip images)

train_data= train_generator.flow_from_dataframe(
dataframe=train,
x_col="file_name",
y_col="dx",
batch_size=64,
seed=311,
directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",
shuffle=True,
class_mode="categorical",
target_size=(224,224))


# In[ ]:


test_generator=ImageDataGenerator(
rescale = 1./255)
test_data= test_generator.flow_from_dataframe(
dataframe=test,
x_col="file_name",
y_col="dx",
seed=45,
directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",
shuffle=False,
batch_size=1,
class_mode=None,
target_size=(224,224))
val_data=test_generator.flow_from_dataframe(
dataframe=val,
directory="../input/mnist1000-with-one-image-folder/ham1000_images/HAM1000_images",
x_col="file_name",
y_col="dx",
batch_size=64,
seed=45,
shuffle=False,
class_mode="categorical",
target_size=(224,224))


# # 3.Combine and run

# Adding classweights

# In[ ]:


from sklearn.utils import class_weight
class_weight = np.round(class_weight.compute_class_weight('balanced',np.unique(y_train),y_train['dx']))
print(class_weight)
print(train_data.class_indices)
print(val_data.class_indices)
print(train['dx'].value_counts())


# Automatically reducing the LR after 3 "patient epochs", then run

# In[ ]:


from keras.metrics import top_k_categorical_accuracy

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                    patience=3, 
                                    verbose=1, 
                                    factor=0.5, 
                                    min_lr=0.00001)

model.compile(optimizer=optimizers.adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
history=model.fit_generator(generator=train_data,
                    steps_per_epoch=train_data.samples//train_data.batch_size,
                            validation_data=val_data,
                            verbose=1,
                            validation_steps=val_data.samples//val_data.batch_size,
                    epochs=35,class_weight=class_weight,callbacks=[learning_rate_reduction])


# # 4. Evaluation and testing on validation set

# In[ ]:


val_data.reset()
predictions = model.predict_generator(val_data, steps=val_data.samples/val_data.batch_size,verbose=1)


# In[ ]:


y_pred= np.argmax(predictions, axis=1)
print(y_pred)
ground_truth=val_data.classes


# In[ ]:


from sklearn.metrics import classification_report
print('Classification Report')
target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv','vasc']
print(classification_report(val_data.classes, y_pred, target_names=target_names))


# In[ ]:





# # 5 Evaluation and testing on test set

# In[ ]:


test_data.reset()
predictions = model.predict_generator(test_data, steps=test_data.samples/test_data.batch_size,verbose=1)
y_pred= np.argmax(predictions, axis=1)


# In[ ]:


print(y_pred)
ground_truth=test['label']
from sklearn.metrics import classification_report
print('Classification Report')
target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv','vasc']
print(classification_report(ground_truth, y_pred, target_names=target_names))

