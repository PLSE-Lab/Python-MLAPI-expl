#!/usr/bin/env python
# coding: utf-8

# This kernel is used to show a kaggle public kernel area bug. More details see [link](https://www.kaggle.com/product-feedback/142959). NOTE: the error is showing on the plant pathology end(public kernel area), if you are at the digit recognizer end, please ignore this.
# 

# In[ ]:


import numpy as np
import pandas as pd
from keras.optimizers import Adam ,RMSprop
from keras.utils.np_utils import to_categorical
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.optimizers import RMSprop
from keras.preprocessing import image
test=pd.read_csv('../input/digit-recognizer/test.csv')
train=pd.read_csv('../input/digit-recognizer/train.csv')

X_train = (train.iloc[:,1:].values).astype('float32')
y_train = train.iloc[:,0].values.astype('int32')
y_train= to_categorical(y_train)
X_test = test.values.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px

model = Sequential([
        Lambda(standardize, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(),
        Convolution2D(32,(5,5), activation='relu',padding='SAME'),
        BatchNormalization(),
        Dropout(0.4),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(),
        Convolution2D(64,(5,5), activation='relu',padding='SAME'),
        BatchNormalization(),
        Dropout(0.4),
        Flatten(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
model.compile(Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
gen = image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)
X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=256)
val_batches=gen.flow(X_val, y_val, batch_size=256)
history=model.fit_generator(generator=batches, steps_per_epoch=len(batches), epochs=5, 
                    validation_data=val_batches, validation_steps=len(val_batches))
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submissionhh.csv", index=False, header=True)

