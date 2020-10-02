#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


# In[ ]:


# get data from '.scv' files.
train = pd.read_csv('../input/digit-recognizer/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = train.ix[:,1:].values.astype('float32')
X_test = (pd.read_csv('../input/digit-recognizer/test.csv').values).astype('float32')
#convert list of labels to binary class matrix
y_train = train.iloc[:,0].values.astype('int32')


# In[ ]:


# reshape data to [28,28],like a picture! You know, CNN can deal with photos.
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x):
    return (x-mean_px)/std_px
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
seed = 43
np.random.seed(seed)


# In[ ]:


from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam ,RMSprop
from keras.preprocessing import image
gen = image.ImageDataGenerator()
# create CNN model
def get_bn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'],
                  )
    return model


# In[ ]:


from sklearn.model_selection import train_test_split
X = X_train
y = y_train
# you can try other parameters and see the results.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)


model= get_bn_model()
# model.optimizer.lr=0.01
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=5,
                    validation_data=val_batches, validation_steps=val_batches.n)

predictions = model.predict_classes(X_test, verbose=0)


# In[ ]:


# output the '.scv' file to submit.
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv('submission.scv',index=False,header=True)

# by the way! I love KAGGLW kernel free GPU! It is quite friendly to students!

