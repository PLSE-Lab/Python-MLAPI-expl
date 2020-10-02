#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import math
import h5py
import random
import nilearn as nl
from nilearn import image, datasets, plotting
from nilearn.image import get_data
from random import randint
from tqdm.notebook import tqdm
from sklearn.impute import KNNImputer


# In[ ]:


source = "../input/trends-assessment-prediction"


# In[ ]:


img_train = []
for name in os.listdir(source+'/fMRI_train'):
    img_train.append(int(name[:5]))
img_train.sort()


# In[ ]:


t = h5py.File(source+'/fMRI_train/10001.mat', 'r')['SM_feature'][()]
for i in range(53):
    print(i)
    x_axis = t[:,:,i].transpose(1,2,0)
    plt.imshow(x_axis[:, :,28], cmap=plt.cm.Set1)
    plt.show()


# #### I think the 17th picture is the most representative. But you can try another one.

# # Let's create a dataset

# In[ ]:


X_train_x = np.array([])
for id_img in tqdm(img_train):
    t = h5py.File(source+f'/fMRI_train/{id_img}.mat', 'r')['SM_feature'][()]
    x_axis = t[:,:,17].transpose(1,2,0)
    
    if len(X_train_x) == 0:
        X_train_x = x_axis[:, :,28]
    elif X_train_x.shape[1] == 53:
        X_train_x = np.append([X_train_x], [x_axis[:, :,28]], axis=0)
    else:
        X_train_x = np.append(X_train_x, [x_axis[:, :,28]], axis=0)


# In[ ]:


img_test = []
for name in os.listdir(source+'/fMRI_test'):
    img_test.append(int(name[:5]))
img_test.sort()


# In[ ]:


X_test_x = np.array([])
for id_img in tqdm(img_test):

    t = h5py.File(source+f'/fMRI_test/{id_img}.mat', 'r')['SM_feature'][()]
    x_axis = t[:,:,17].transpose(1,2,0)
    
    if len(X_test_x) == 0:
        X_test_x = x_axis[:, :,28]
    elif X_test_x.shape[1] == 53:
        X_test_x = np.append([X_test_x], [x_axis[:, :,28]], axis=0)
    else:
        X_test_x = np.append(X_test_x, [x_axis[:, :,28]], axis=0)


# In[ ]:


X_train_x = X_train_x.reshape(X_train_x.shape[0], 1, 52, 53)
X_test_x = X_test_x.reshape(X_test_x.shape[0], 1, 52, 53)


# In[ ]:


train_scores = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")
loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")
sample_submission = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")


# In[ ]:


train_scores = train_scores.drop(['Id'], axis=1)


# In[ ]:


train_scores.head()


# In[ ]:


impute = KNNImputer(n_neighbors=20)
y_train = impute.fit_transform(train_scores)


# # Let's create CNN

# In[ ]:


import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau


# In[ ]:


def weighted_NAE(yTrue,yPred):
    weights = K.constant([.3, .175, .175, .175, .175], dtype='float32')
    
    return K.sum(weights*K.sum(K.abs(yTrue-yPred))/K.sum(yPred))


# In[ ]:


model = Sequential()
model.add(BatchNormalization()) 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,52,53), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization()) 
model.add(Dropout(0.4))
model.add(Convolution2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4)) 
model.add(Flatten())
model.add(BatchNormalization()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='relu'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=[weighted_NAE])


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode='min',
                              patience=3, min_lr=0.001)


# In[ ]:


history = model.fit(X_train_x, y_train, epochs=20, batch_size=1024, validation_split=0.1, shuffle=True, callbacks=[reduce_lr])


# In[ ]:


model.summary()


# In[ ]:


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 21)
plt.plot(epochs, loss_values, 'b', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure(figsize=(15,10))
plt.show()


# In[ ]:


pred=pd.DataFrame()
pred["Id"]=sample_submission.Id
pred["Predicted"]=model.predict(X_test_x).flatten()
pred.to_csv('out.csv', index=False)


# # You can also improve the model or use other axes when cutting pictures. Good luck!

# In[ ]:




