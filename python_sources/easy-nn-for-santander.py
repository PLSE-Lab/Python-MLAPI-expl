#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ver = 'nn_bl_14'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use(['seaborn-darkgrid'])
plt.rcParams['font.family'] = 'DejaVu Sans'
import time
from datetime import datetime

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# **Load datasets**

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


train['target'].value_counts().sort_index(ascending=False).plot(kind='barh', 
                                                                          figsize=(15,6))
plt.title('Target', fontsize=18);


# **Data preparation**

# In[ ]:


X = train.iloc[:,2:].values
y = train.iloc[:,1].values
test = test.iloc[:,1:].values


# In[ ]:


sc0 = StandardScaler()
sc1 = RobustScaler()


# In[ ]:


X_train0 = sc0.fit_transform(X)
X_test0 = sc0.transform(test)
X_train1 = sc1.fit_transform(X)
X_test1 = sc1.transform(test)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (24, 12))
pca = PCA()
X_reduced0 = pca.fit_transform(X_train0)
X_reduced1 = pca.fit_transform(X_train1)


ax[0].scatter(X_reduced0[:, 0], X_reduced0[:, 1], c=y,
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('bwr', 2))
ax[0].set_title('PCA projection StdScalar')

ax[1].scatter(X_reduced1[:, 0], X_reduced1[:, 1], c=y,
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('bwr', 2))
ax[1].set_title('PCA projection Robust')

print(pca.n_components_)


# **Model**

# In[ ]:


def modelir():
    model = Sequential()

    #input 
    model.add(Dense(200, input_dim=200, kernel_initializer = 'uniform'))
    model.add(Activation("relu"))
    #model.add(Dropout(0.8))

    #2 
    model.add(Dense(1024, kernel_initializer = 'uniform'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    for _ in range(5):
        model.add(Dense(10, kernel_initializer = 'uniform'))
        model.add(Activation('relu'))
    
    model.add(Dense(10, kernel_initializer = 'uniform'))
    model.add(Activation('relu'))
    
    

    #output
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
#print(model.summary())


# In[ ]:


def simple_blend1(X, y, test):
    model = modelir()
    pred = pd.DataFrame()
    for i in range(1, 3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
        earlystopper = EarlyStopping(patience=5, verbose=1)
        history = model.fit(X_train, y_train, batch_size=512, epochs=500, validation_split=0.2, verbose=2, callbacks=[earlystopper], shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("accuracy for test data: %.2f%%" % (scores[1]*100))
        plt.plot(history.history['acc'], label='accuracy for train data')
        plt.plot(history.history['val_acc'], label='validation data accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.7)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        y_pred_t = model.predict(test)
        print(y_pred_t.T[0])
        pred[i] = y_pred_t.T[0]
    return pred


# In[ ]:


pr1 = simple_blend1(X, y, test)


# In[ ]:


pr1['mean'] = pr1.mean(axis=1)
pr1.head()


# In[ ]:


def simple_blend2(X, y, test):
    pred = pd.DataFrame()
    for i in range(1, 3):
        model = modelir()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
        earlystopper = EarlyStopping(patience=5, verbose=1)
        history = model.fit(X_train, y_train, batch_size=512, epochs=500, validation_split=0.2, verbose=2, callbacks=[earlystopper], shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("accuracy for test data: %.2f%%" % (scores[1]*100))
        plt.plot(history.history['acc'], label='accuracy for train data')
        plt.plot(history.history['val_acc'], label='validation data accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.7)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        y_pred_t = model.predict(test)
        print(y_pred_t.T[0])
        pred[i] = y_pred_t.T[0]
    return pred


# In[ ]:


pr2 = simple_blend2(X, y, test)


# In[ ]:


pr2['mean'] = pr2.mean(axis=1)
pr2.head()


# **Submission**

# In[ ]:


filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))
target['target'] = pr1['mean']
target.to_csv(filename+'1'+'.csv', index=False)
target['target'] = pr2['mean']
target.to_csv(filename+'2'+'.csv', index=False)


# In[ ]:





# In[ ]:




