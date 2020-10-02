#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output. 


# In[ ]:


import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
import sys
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D, MaxPool2D
from keras.optimizers import RMSprop, Adam, sgd, Adamax
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import skimage
from skimage.transform import rescale
from keras.utils import np_utils
from scipy.misc import imresize
import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing.image import load_img
import shutil
from glob import glob
from keras import regularizers
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU
from keras.callbacks import CSVLogger
def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
tf.keras.backend.clear_session()
from keras.datasets import mnist


# In[ ]:


# MODEL_NUMBER = 1
NB_EPOCH = 40
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# OPTIMIZE = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
N_HIDDEN = 128
VALIDATION_SPLIT = 0.05
DROPOUT = 0.3
RESHAPED = (28, 28, 1)


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


x_train = train.drop(labels=["label"], axis=1)
x_train = x_train / 255.0
x_train = x_train.values.reshape(-1, 28, 28, 1)
y_train = train["label"]
y_train = np.array(to_categorical(y_train, num_classes=10))
x_test = test / 255.0
x_test = x_test.values.reshape(-1, 28, 28, 1)
print(x_train.shape, y_train.shape, x_test.shape)


# In[ ]:


get_ipython().system('wc -l ../input/train.csv')


# In[ ]:


(x_train_mnist, y_train_mnist), (x_val_mnist, y_val_mnist) = mnist.load_data()
x_train_mnist = np.concatenate((x_train_mnist, x_val_mnist))
y_train_mnist = np.concatenate((y_train_mnist, y_val_mnist))

x_train_mnist = x_train_mnist.reshape((x_train_mnist.shape[0], 28, 28, 1))
X_train_mnist = x_train_mnist / 255
y_train_mnist = np.array(to_categorical(y_train_mnist.reshape((y_train_mnist.shape[0], 1)),num_classes=10))
print(x_train_mnist.shape, y_train_mnist.shape)


# In[ ]:


x_train = np.concatenate((x_train, x_train_mnist))
y_train = np.concatenate((y_train, y_train_mnist))


# In[ ]:


datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# In[ ]:


np.random.seed(0)


# In[ ]:


input_shape = (28,28,1)
model = Sequential()
model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 96,kernel_size = (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters = 192, kernel_size = (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 256, kernel_size = (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


def create_log_dir():
    if os.path.exists('./log_dir'):
        shutil.rmtree('./log_dir')
    os.mkdir('./log_dir')


# In[ ]:


create_log_dir()
print(os.listdir("./"))


# In[ ]:


learning_rate = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=1.0e-5)


# In[ ]:


EarlyStopping_cb = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')


# In[ ]:


csv_logger = CSVLogger('model.log')


# In[ ]:


def history_plot():
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train, y_train, test_size=VALIDATION_SPLIT, random_state=0)


# In[ ]:


print(x_train2.shape)
print(x_val2.shape)


# In[ ]:


def fit_model():
    model_file = './log_dir/model_file' + '.hdf5'
    checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, monitor='val_acc', save_best_only=True)
    history = model.fit_generator(datagen.flow(x_train2, y_train2, batch_size=BATCH_SIZE),
                              validation_data=(x_val2, y_val2),
                              epochs = NB_EPOCH, 
                              verbose = 2, 
                              steps_per_epoch=x_train2.shape[0] // BATCH_SIZE ,
                              callbacks=[learning_rate, checkpointer, csv_logger, EarlyStopping_cb])
#                               steps_per_epoch=x_train2.shape[0],                                     
#                              callbacks=[learning_rate, checkpointer, csv_logger])
    return(history)


# In[ ]:


history = fit_model()


# In[ ]:


history_plot()


# In[ ]:


results = np.zeros((x_test.shape[0], NB_CLASSES))
model_file = './log_dir/model_file' + '.hdf5'
model.load_weights(model_file)
results = results + model.predict(x_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().system('wc -l ./submission.csv')


# In[ ]:


get_ipython().system('head ./submission.csv')

