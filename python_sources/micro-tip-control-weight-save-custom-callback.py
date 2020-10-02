#!/usr/bin/env python
# coding: utf-8

# # [micro tip]Control weight save(Keras callback)

# When use Kaggle kernel, file I/O takes lots of time to save model weights. In this, I make custom `ModelCheckpoint` callbaks to control the point save model weights using `epoch` or `accuracy` etc...
# 
# I apply APTOS model and work without error. In the kerenl, just make simple Resnet model and training mnist dataset to check fast.
# 
# Reference code is @Keeplearning's [[APTOS] resnet50 baseline](https://www.kaggle.com/mathormad/aptos-resnet50-baseline/output). 

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger

print(os.listdir('../input'))


# ## Make custom ModelCheckpoint callback

# Basic `ModelCheckpoint` callback make using some parameters like that([Keras callback doc](https://keras.io/callbacks/#modelcheckpoint)):
#     
#     keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# Let make custom `ModelCheckpoint` callback. I override `on_epoch_end` function simply adding 'epoch condition'. I use some `print` to check `epoch` and `logs`. Delete this part after using.
# 
# Original Keras `callback` code is in [keras.callback.ModelCheckpoint](https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/keras/callbacks.py#L805-L1139)

# In[ ]:


from keras.callbacks import ModelCheckpoint


# In[ ]:


class Custom_checkpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        print('epoch : ', epoch) #Just check 
        print('logs : ', logs)  #Just check
        #Set this epoch
        if epoch <= 2: 
            return print('Low epoch')
        super().on_epoch_end(epoch, logs)


# In[ ]:


cp = Custom_checkpoint('../working/Resnet50.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)


# In[ ]:


#Create other callbacks.
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                   verbose=1, mode='min', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=9)

csv_logger = CSVLogger(filename='../working/training_log.csv',
                       separator=',',
                       append=True)


# ## Model training

# mnist model training code is in [Keras mnist cnn](https://keras.io/examples/mnist_cnn/).

# In[ ]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[cp, reduceLROnPlat])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Model weights saved after `epoch` over 2(setting number). We can see that `epoch` start 0 and `logs` is a dict writen 'val_loss', 'val_acc' etc.
# 
# I make other `ModelCheckpoint` to use 'val_loss'.

# In[ ]:


class Custom_checkpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_acc'] <= 0.99:
            return print('Low val_acc')
        super().on_epoch_end(epoch, logs)


# In[ ]:


cp2 = Custom_checkpoint('../working/Resnet50.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[cp2, reduceLROnPlat])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In [APTOS] resnet50 baseline, make `QWKEvaluation` callback option. We can apply control option like above.
# 
# I make `QWKEvaluation` after epoch is over 20.

# In[ ]:


from keras.callbacks import Callback
class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), batch_size=64, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        ###Add epoch condition
        if epoch <= 20:
            return print('Low epoch : ', epoch)
        
        ###Org code###
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(generator=self.valid_generator,
                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                                  workers=1, use_multiprocessing=True,
                                                  verbose=1)
            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)
                # return np.sum(y.astype(int), axis=1) - 1
            
            score = cohen_kappa_score(flatten(self.y_val),
                                      flatten(y_pred),
                                      labels=[0,1,2,3,4],
                                      weights='quadratic')
#             print(flatten(self.y_val)[:5])
#             print(flatten(y_pred)[:5])
            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history):
                print('save checkpoint: ', score)
                self.model.save('../working/Resnet50_bestqwk.h5')

qwk = QWKEvaluation(validation_data=(valid_generator, valid_y),
                    batch_size=batch_size, interval=1)

