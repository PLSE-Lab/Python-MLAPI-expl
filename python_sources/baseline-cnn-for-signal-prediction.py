#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPooling1D
from keras.layers import Conv1D
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
import plotly.graph_objs as go
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
gc.collect()
plt.rcParams['figure.figsize'] = [20, 8]


# In[11]:


train = pd.read_csv('../input/X_train.csv')
y = pd.read_csv('../input/y_train.csv')
test = pd.read_csv('../input/X_test.csv')
ss = pd.read_csv('../input/sample_submission.csv')


# In[12]:


tmp = dict(zip(y.series_id, y.surface))
train['target'] = train['series_id'].map(tmp)
train.head()
train.head()


# In[13]:


train.head()


# In[14]:


le = LabelBinarizer()
lb = LabelEncoder()
target = le.fit_transform(y['surface'])
target = np.array(target)
y['s']=lb.fit_transform(y['surface'])


# In[15]:


def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def elr(train):
    x, y, z, w = train['orientation_X'].tolist(), train['orientation_Y'].tolist(), train['orientation_Z'].tolist(), train['orientation_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    train['euler_x'] = nx
    train['euler_y'] = ny
    train['euler_z'] = nz
    return(train)    

train = elr(train)
test = elr(test)
train.head()


# In[16]:


def create_seg(train):
    segments = []

    for i in range(0, len(train), 128): 
        ox = train.orientation_X.values[i:i+128]
        oy = train.orientation_Y.values[i:i+128]
        oz = train.orientation_Z.values[i:i+128]
        ow = train.orientation_W.values[i:i+128]
        ax = train.angular_velocity_X.values[i:i+128]
        ay = train.angular_velocity_Y.values[i:i+128]
        az = train.angular_velocity_Z.values[i:i+128]
        lx = train.linear_acceleration_X.values[i:i+128]
        ly = train.linear_acceleration_Y.values[i:i+128]
        lz = train.linear_acceleration_Z.values[i:i+128]
        ex = train.euler_x.values[i:i+128]
        ey = train.euler_y.values[i:i+128]
        ez = train.euler_z.values[i:i+128]

        segments.append([ox, oy, oz, ow, ax, ay,az,lx, ly, lz, ex, ey, ez])
    segments = np.asarray(segments, dtype= np.float32).reshape(-1, 128, 13)
    return segments


# In[17]:


seg = create_seg(train)
test = create_seg(test)


# In[18]:


model = Sequential()
model.add(Conv1D(128, 1, input_shape = (128,13)))
model.add(Activation("relu"))
model.add(Conv1D(128,1))
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))

model.add(Conv1D(64,3))
model.add(Activation("relu"))
model.add(Conv1D(64,3))
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))

model.add(Conv1D(32,5))
model.add(Activation("relu"))
model.add(Conv1D(32,5))
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Dense(9, activation='sigmoid'))


# In[19]:


model.summary()


# In[20]:


filepath = "skynet.hdf5"
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), 
                 keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]

model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

history = model.fit(seg,
                      target,
                      batch_size=50,
                      epochs=100,
                      callbacks=callbacks_list,
                      validation_split=0.3,
                      shuffle=True,
                      verbose=1)


# In[21]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[22]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[23]:


model.load_weights(filepath)
predictions = model.predict(test)
prd = le.inverse_transform(predictions)
ss['surface'] = prd
ss.to_csv('submission.csv', index=False)

