#!/usr/bin/env python
# coding: utf-8

# ## **MNIST Digit Recognition**
# 
# Digit-MNIST is the most fundamental and easiest dataset possible for deep learning purposes. And we have seen [here](https://www.kaggle.com/gauravsharma99/getting-started-with-digit-mnist-using-keras) that without much efforts we achieved almost 99.5% accuracy on this dataset (which is not even the standard 60k MNIST). Also with this same model and configuration we may have achieved even higher accuracy with the 60k MNIST.
# 
# Although we have achieved pretty good accuracy in such an easy dataset but for such we an easy dataset our model has parameters in the range of 100k-200k, which is way more for such an easy task. So can we do better i.e., reducing the model size significantly and simultaneously retaining the accuracy in the same range.
# 
# Recently I was given a task by some organization to achieve an accuracy of atleast 99.4% on the 60k MNIST. You might think what's good about that, even handicapped models can achieve an accuracy of around 98-99% in MNIST without doing anything. But the constraint was to achieve such an high accuracy using a model having **atmost 8k parameters**. Although I wasn't able to touch the 99.4% bar but I got an best accuracy of around 99.2% having less than 8k parameters in the given time limit. And I was sure if the model have given enough time then 99.4% accuracy is achievable afterall no one design a task which is un-achievable. The organization may already achieved that and that's why gave us such a task.
# 
# But the idea here is that before jumping to deep networks we first try to achieve the goal in minimal model possible, just like we say in ML i.e., never jump to complex models but first try simple models like linear models. So using such low number of parameters we achieved pretty good accuracy. The benefits are many like easy to inspect, debug, load, train etc. Because the number of layers & parameters are now very less so we can inspect and debug(if needed) our network very easily.
# 
# I thought it's worth sharing the notebook, it's very simple and I didn't do much.
# 
# 
# #### Model Architecture
# I used **only convolutional layers with no dense layer** due to the constraint on number of trainable parameters. Due to this the network has only **7968 trainable parameters**. Also **MaxPooling** helped a lot in reducing the trainable parameters.
# 
# #### Model Generalization
# For model generalization I used **BatchNormalization** and **Dropout** in the architecture. While during the training of network I used **ReduceLROnPlateau** and **EarlyStopping**.<br><br>
# 
# I used various different settings like cropping 28x28 image down to 20x20 and using ImageDataGenerator but didn't get any significant improvements for this small network and hence finally retained to these configurations.
# Under all these configurations I achieved the best **validation accuracy of almost 99.2%** for the given
# neural network. The training accuracy is also in the range of 99%.<br><br>

# In[ ]:


import os
import random

import numpy as np
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


# settings to get reproducible results, still the results are not entirely reproducible.
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# In[ ]:


def preprocess_data(X, normalize=True):
    X = X.astype('float32')

    if normalize:
        X /= 255.

    return X.reshape(*X.shape, 1)


# In[ ]:


(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = preprocess_data(X_train)
X_valid = preprocess_data(X_valid)

y_train = np_utils.to_categorical(y_train)
y_valid = np_utils.to_categorical(y_valid)

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]


# In[ ]:


def build_cnn_8k(optim='adam', loss='categorical_crossentropy'):
    '''
    Total Trainable params: 7,968 (< 8k)
    '''
    net = Sequential(name='cnn_8k')

    net.add(
        Conv2D(
            filters=32,
            kernel_size=(3,3),
            input_shape=(img_width, img_height, img_depth),
            name='conv2d_1'
        )
    )
    net.add(LeakyReLU(name='leaky_relu_1'))
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(MaxPooling2D(pool_size=(2,2), name='max_pool_1'))

    net.add(
        Conv2D(
            filters=14,
            kernel_size=(3,3),
            name='conv2d_2'
        )
    )
    net.add(LeakyReLU(name='leaky_relu_2'))
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(MaxPooling2D(pool_size=(2,2), name='max_pool_2'))

    net.add(Flatten(name='flatten_layer'))
    net.add(Dropout(0.2, name='dropout_1'))
    net.add(Dense(num_classes, activation='softmax', name='dense_out'))
    
    net.compile(
        loss=loss,
        optimizer=optim,
        metrics=['accuracy']
    )
    
    net.summary()
    return net


# In[ ]:


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    min_delta=0.00005,
    baseline=0.98,
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=4,
    factor=0.5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [
    early_stopping,
    lr_scheduler,
]


# In[ ]:


loss = 'categorical_crossentropy'
optim = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
batch_size = 128
epochs = 40

model = build_cnn_8k(optim, loss)
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True
)


# In[ ]:


sns.set()


# In[ ]:


fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label='train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label='train')
sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
plt.title('Loss')
plt.tight_layout()

plt.show()


# In[ ]:


import pandas as pd
df_accu = pd.DataFrame({'train': history.history['accuracy'], 'valid': history.history['val_accuracy']})
df_loss = pd.DataFrame({'train': history.history['loss'], 'valid': history.history['val_loss']})

fig = plt.figure(0, (14, 4))
ax = plt.subplot(1, 2, 1)
sns.violinplot(x="variable", y="value", data=pd.melt(df_accu), showfliers=False)
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.violinplot(x="variable", y="value", data=pd.melt(df_loss), showfliers=False)
plt.title('Loss')
plt.tight_layout()

plt.show()


# The outliers in the plots are of `initial epochs`.

# In[ ]:




