#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow.compat.v1 as tf


# In[ ]:


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy

from tqdm import tqdm
import keras
from keras.datasets import mnist

import datetime

get_ipython().run_line_magic('matplotlib', 'inline')

BATCH_SIZE = 32
IMG_SIZE = 224


# In[ ]:


np.random.seed(2019)
tf.set_random_seed(2019)


# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
print(train_df.shape)
train_df.head()


# In[ ]:


train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()


# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# resize image

# In[ ]:


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=IMG_SIZE):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im




N = train_df.shape[0]
x_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    )


# In[ ]:


y_train = pd.get_dummies(train_df['diagnosis']).values
#one hot encode

print(x_train.shape)
print(y_train.shape)


# In[ ]:


y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# In[ ]:


x_train_val, x_test, y_train_val, y_test = train_test_split(
    x_train, y_train_multi, 
    test_size=0.1, 
    random_state=2019
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, 
    test_size=0.1, 
    random_state=2019
)
print("shape of train images is :",x_train.shape)
print("shape of test images is :",x_test.shape)
print("shape of valid images is :",x_val.shape)
print("shape of train labels is :",y_train.shape)
print("shape of test labels is :",y_test.shape)
print("shape of valid labels is :",y_val.shape)


# In[ ]:




def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)


# kappa

# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# model effientnet

# In[ ]:


get_ipython().system('pip install keras_efficientnets')
#!pip install -U efficientnet


# In[ ]:


#import efficientnet.keras as efn 
from keras_efficientnets import *
effnet = EfficientNetB4(
         weights=None,
         include_top=False,
         input_shape=(IMG_SIZE,IMG_SIZE,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(effnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:



model = build_model()
model.summary()

training
# In[ ]:


def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1



startdate = datetime.datetime.now() 
startdate = startdate.strftime("%Y-%m-%d %H:%M:%S") 


# In[ ]:


from keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
kappa_metrics = Metrics()
est=EarlyStopping(monitor='val_loss',patience=5, min_delta=0.005)
call_backs=[est,kappa_metrics]

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    validation_data=(x_val,y_val),
     epochs=15,
    callbacks=call_backs)


# In[ ]:


enddate = datetime.datetime.now() 
enddate = enddate.strftime("%Y-%m-%d %H:%M:%S") 

print('start date ',startdate)
print('end date ',enddate)
print('Time ',subtime(startdate,enddate)) 


# In[ ]:


import numpy
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

with open('history.json', 'w') as f:
    json.dump(history.history, f,cls=MyEncoder)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['accuracy', 'val_accuracy']].plot()


# In[ ]:


plt.plot(kappa_metrics.val_kappas)


# In[ ]:


model.load_weights('model.h5')
results = model.evaluate(x_test,  y_test, verbose=0)

print("test loss:",results[0])
print("test accuracy:",results[1])


# In[ ]:


def get_preds_and_labels(model):
    preds = []
    labels = []
   
    preds.append(model.predict(x_test))
    labels.append(y_test)
   
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()
test_kappas = []
y_pred, labels = get_preds_and_labels(model)
y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
_test_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
test_kappas.append(_test_kappa)
print(f"test_kappa: {round(_test_kappa, 4)}")

