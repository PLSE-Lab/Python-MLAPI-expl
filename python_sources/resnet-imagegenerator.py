#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import json
from tqdm import tqdm,tqdm_notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df=pd.read_csv("../input/train.csv")
train_df["has_cactus"]=train_df["has_cactus"].map(lambda x:str(x))
print(train_df.shape)

import cv2
image=cv2.imread("../input/train/train/01e30c0ba6e91343a12d2126fcafc0dd.jpg"
                )
plt.imshow(image)
print(image.shape)


# In[13]:


from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense,MaxPooling2D
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers


# In[17]:


base_model=ResNet50(weights="imagenet",include_top=False,input_shape=(32,32,3))


# In[22]:


def add_new_layer(base_model):
    x=base_model.output
    x=Flatten()(x)
    x=Dense(512,activation="relu")(x)
    predictions = Dense(1, activation='sigmoid',activity_regularizer=regularizers.l1(0.05))(x)
    model = Model(input=base_model.input, output=predictions)
    return model
def transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',
                  metrics=['accuracy'])


# In[9]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 validation_split=0.1,
                                 rotation_range=30,
                                 shear_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 zoom_range=0.2)
train_generator=train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../input/train/train",
    x_col="id",
    y_col="has_cactus",
    batch_size=32,
    shuffle=True,
    class_mode="binary",
    target_size=(32,32),
    subset='training')

validation_generator=train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../input/train/train",
    x_col="id",
    y_col="has_cactus",
    batch_size=32,
    shuffle=True,
    class_mode="binary",
    target_size=(32,32),
    subset='validation')


# In[23]:


model=add_new_layer(base_model)


# In[24]:


print(model.summary())


# In[26]:


transfer_learn(model, base_model)


# In[27]:


history=model.fit_generator(generator=train_generator,
                            validation_data=validation_generator,
                            validation_steps=int(train_df.shape[0]/32),
                            steps_per_epoch=int(train_df.shape[0]/32),
                            epochs=200,
                            verbose=2)


# In[28]:


history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[29]:


train_dir=r"../input/train/train/"
test_dir=r"../input/test/test/"
X_test=[]
X_image=[]
for image in tqdm_notebook(os.listdir(test_dir)):
    im=cv2.imread(test_dir+image)
    X_test.append(im)
    X_image.append(image)
X_test=np.array(X_test)
X_test=X_test/255.0


# In[30]:


print(X_test.shape)


# In[31]:


test_predictions=model.predict(X_test)


# In[32]:


submission=pd.DataFrame(test_predictions,columns=['has_cactus'])


# In[33]:


submission['id'] = ''
cols=list(submission.columns)
cols = cols[-1:] + cols[:-1]
submission=submission[cols]
for i, img in enumerate(X_image):
    submission.set_value(i,'id',img)
print(submission)


# In[34]:


submission.to_csv('submission.csv',index=False)


# In[35]:


print(submission.head())

