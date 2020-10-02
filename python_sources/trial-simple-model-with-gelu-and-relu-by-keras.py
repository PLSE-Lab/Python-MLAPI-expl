#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, DepthwiseConv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.activations import relu

from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# define constants
BASE_PATH = '/kaggle/input/Kannada-MNIST/'

# set random seed
np.random.seed(1973)


# ## Load data files 
# 
# Loading datase by pandas dataframe.
# 
# - train.csv : for Training data 
# - test.csv : for Test data 
# - Dig-MNIST.csv : additional labeled data. This file can use for validation and test model. very kindly data for me (us?).
# - sample_submission.csv : sample data for submission.

# In[ ]:


train_src_df = pd.read_csv(filepath_or_buffer=BASE_PATH+'train.csv')
test_src_df = pd.read_csv(filepath_or_buffer=BASE_PATH+'test.csv')
dm_src_df = pd.read_csv(filepath_or_buffer=BASE_PATH+'Dig-MNIST.csv')
submission = pd.read_csv(filepath_or_buffer=BASE_PATH+'sample_submission.csv')


# ### train.csv 
# train.csv contain 60,000 labeled data. each data have label and 1d pixcel data.  
# Pixcel data may be necessary convet to 2d pixcel data.  
# Pixel data takes a value in the range of 0-255.  
# Label data takes a value in the range of 0-9.  

# In[ ]:


print(train_src_df.info())
print(train_src_df.head())
print(train_src_df.describe())


# ### test.csv  
# test.csv has 5,000 non labeled data.  
# first column is for id. The second and subsequent columns are pixel data.  
# data range is same as train.csv.

# In[ ]:


print(test_src_df.info())
print(test_src_df.head())
print(test_src_df.describe())


# ### Dig-MNIST  
# Dig-MNIST.csv has 10,240 labeled data.  
# It's enough values for validation and test.

# In[ ]:


print(dm_src_df.info())
print(dm_src_df.head())
print(dm_src_df.describe())


# ### preprocess  
# separate label and pixcel data. and, convert shape of pixcel data, 1d to 3d (height , width, channel).  
# and create ImageDataGenerator for data augumentation, for avoid overfitting. 

# In[ ]:


all_data = pd.concat([train_src_df, dm_src_df])
'''
X_train = train_src_df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28, 1)
y_train = to_categorical(train_src_df.iloc[:, 0])
X_additional = dm_src_df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28, 1)
y_additional = to_categorical(dm_src_df.iloc[:, 0])
X_valid, X_test, y_valid, y_test = train_test_split(X_additional, y_additional, test_size=0.5, shuffle=True)
'''
X_all_data = all_data.iloc[:, 1:].to_numpy().reshape(-1, 28, 28, 1)
y_all_data = to_categorical(all_data.iloc[:, 0])

X_train, X_valid, y_train, y_valid = train_test_split(X_all_data, y_all_data, test_size=0.07, shuffle=True)
test_data = test_src_df.iloc[:, 1:].to_numpy().reshape(-1, 28, 28, 1)
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(test_data.shape)


# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(15,15))
for idx in range(9):
    i = idx % 3 # Get subplot row
    j = idx // 3 # Get subplot column
    axes[i, j].imshow(X_train[idx].reshape(28, 28))

plt.show()


# In[ ]:


train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7,1.0],
    shear_range=0.2,
    zoom_range=0.2, 
    channel_shift_range=0.0, 
    fill_mode='constant', 
    cval=0.0, 
    horizontal_flip=False, 
    vertical_flip=False, 
    rescale=1./255, 
    preprocessing_function=None, 
    data_format='channels_last', 
    validation_split=0.0, 
    dtype='float32')

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(X_train)


# ## Define model 
# define model like vgg 16. 
# I use activation function, 'gelu', used in BERT.  
# ... but, gelu did not perform as well as relu. The function may be a bit too complex.
# following code is refered from follwoing web page.  
# [gelu in BERT](https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264)  
# 

# In[ ]:


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    refer : https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


# In[ ]:


def create_model(fig_size=28, channel_num=1, class_num=10, activation=relu):
    model = Sequential()
    
    # 28*2 -> 14
    model.add(Conv2D(64, (5, 5), padding='same', activation=activation, input_shape=(fig_size, fig_size, channel_num)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (5, 5), padding='same', activation=activation))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(2))

    # 14*2 -> 7
    model.add(Conv2D(128, (3, 3), padding='same', activation=activation))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same', activation=activation))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(2))

    # 7*3 -> 3
    model.add(Conv2D(256, (3, 3), padding='same', activation=activation))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), padding='same', activation=activation))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), padding='same', activation=activation))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(2))

    # 3 -> 1 -> flatten
    model.add(Conv2D(1024, (1, 1), activation=activation))
    model.add(Dropout(0.5))
    model.add(DepthwiseConv2D((3, 3), padding='valid', activation=activation))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    # fully connection and classify
    model.add(Dense(512, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    # compile
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# ### Fit and evaluate 
# fit and evaluate both relu and gelu, and compare validation loss and acc.

# In[ ]:


model_relu = create_model(activation=relu)
model_gelu = create_model(activation=gelu)
model_relu.summary()


# In[ ]:


# Define callbacks
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1.0e-8, verbose=1)
mcp = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)


# In[ ]:


# Define training parameters
BATCH_SIZE = 64
EPOCH_NUM = 30


# In[ ]:


#training relu
history_relu = model_relu.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE) , 
        validation_data=test_datagen.flow(X_valid, y_valid, batch_size=BATCH_SIZE), 
        epochs=EPOCH_NUM, 
        callbacks=[rlr, mcp]
        )
model_relu.load_weights('weights.hdf5')


# In[ ]:


#training gelu
history_gelu = model_gelu.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE) , 
        validation_data=test_datagen.flow(X_valid, y_valid, batch_size=BATCH_SIZE), 
        epochs=EPOCH_NUM, 
        callbacks=[rlr, mcp]
        )
model_gelu.load_weights('weights.hdf5')


# ### Check result 
# compare result of relu and gelu. 
# on val_loss and val_acc, relu's score is better than gelu.   
# so, I use relu activation model.

# In[ ]:


# Plot training & validation accuracy values
plt.plot(history_relu.history['acc'])
plt.plot(history_relu.history['val_acc'])
plt.plot(history_gelu.history['acc'])
plt.plot(history_gelu.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train:relu', 'Test:relu', 'Train:gelu', 'Test:gelu'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_relu.history['loss'])
plt.plot(history_relu.history['val_loss'])
plt.plot(history_gelu.history['loss'])
plt.plot(history_gelu.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train:relu', 'Test:relu', 'Train:gelu', 'Test:gelu'], loc='upper left')
plt.show()


# In[ ]:


print('val_acc\nrelu:{}\ngelu:{}\nval_loss\nrelu:{}\ngelu:{}'.format(max(history_relu.history['val_acc']), max(history_gelu.history['val_acc']), min(history_relu.history['val_loss']), min(history_gelu.history['val_loss'])))


# ## Predict by test data and create submission data.  
# create submission data by relu model. 

# In[ ]:


pred_ret = None
if min(history_relu.history['val_loss']) < min(history_gelu.history['val_loss']):
    pred_ret = model_relu.predict_generator(test_datagen.flow(test_data, None, shuffle=False, batch_size=BATCH_SIZE))
else:
    pred_ret = model_gelu.predict_generator(test_datagen.flow(test_data, None, shuffle=False, batch_size=BATCH_SIZE))
    
pred_ids = np.argmax(pred_ret, axis=1)
pred_ids.shape


# In[ ]:


submission['label'] = pred_ids
submission.head()


# In[ ]:


submission.to_csv(path_or_buf='submission.csv', index=False)

