#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import np_utils
import itertools
import numpy as np
import seaborn as sns
sns.set_style('ticks') 


# In[ ]:


df = pd.read_csv("/kaggle/input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df_new = df.iloc[:, 3:8]
df_new['redshift'] = df['redshift']
df_new['class'] = df['class']
df_new.info()


# In[ ]:


sns.countplot(x= 'class', data = df_new)
plt.show()


# In[ ]:


#Resampling
star = df_new[df_new['class'] == 'STAR']
galaxy = df_new[df_new['class'] == 'GALAXY']
qso = df_new[df_new['class'] == 'QSO']


# In[ ]:


star = star.sample(qso['class'].count())
galaxy = galaxy.sample(qso['class'].count())


# In[ ]:


df_resampling = pd.concat([star, galaxy, qso], axis = 0)
df_resampling.head()


# In[ ]:


sns.countplot(x= 'class', data = df_resampling)
plt.show()


# In[ ]:


def calc_flux(x):
    return -2.5*np.log(x)


# In[ ]:


aux = df_resampling['class']
df_resampling.drop('class', axis=1, inplace=True)
df_resampling['flux_u'] = calc_flux(df_resampling['u'])
df_resampling['flux_g'] = calc_flux(df_resampling['g'])
df_resampling['flux_r'] = calc_flux(df_resampling['r'])
df_resampling['flux_i'] = calc_flux(df_resampling['i'])
df_resampling['flux_z'] = calc_flux(df_resampling['z'])
df_resampling['class'] = aux

df_resampling = df_resampling.drop(['u','g', 'r', 'i', 'z'], axis =1) 


# In[ ]:


df_resampling.head()


# In[ ]:


#Correlation
plt.figure(figsize=(8, 6))
df_corr = df_resampling.corr()
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdBu')
plt.show()


# In[ ]:


# Pairplot
plt.figure(figsize=(8, 8))
sns.pairplot(df_resampling, hue ="class")
plt.plot()


# In[ ]:


#Transform numpy array
X = df_resampling.iloc[:, :-1].to_numpy()
y = df_resampling['class'].to_numpy()
y = y.reshape(-1, 1)


# In[ ]:


# encode class values as integers
encoder = OneHotEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y).toarray()


# In[ ]:


X_train, X_test_val, y_train, y_test_val = train_test_split(X, encoded_Y, test_size=0.33)


# In[ ]:


X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size = 0.25)


# In[ ]:


#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


# Neural network
model = Sequential()
model.add(Dense(16, input_dim=6, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)

#Optimizer
adam = Adam(lr=1e-3, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64, 
                    callbacks= [reduce_lr])


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()


# In[ ]:


plt.plot(history.history['lr']) 
plt.title('Model Learning rate') 
plt.ylabel('LR') 
plt.xlabel('Epoch') 
plt.show()


# In[ ]:


y_prob = model.predict(X_val) 
y_classes = y_prob.argmax(axis=-1)
y_class = []
for i in range(len(y_classes)):
    y_class.append(encoder.categories_[0][y_classes[i]])
y_class = np.array(y_class)
result = encoder.transform(y_class.reshape(-1, 1)).toarray()


# In[ ]:



fig= plt.figure(figsize=(8,6))

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
sns.heatmap(confusion_mtx, annot=True, fmt="d")


# In[ ]:


acc = accuracy_score(Y_true, Y_pred_classes)
acc

