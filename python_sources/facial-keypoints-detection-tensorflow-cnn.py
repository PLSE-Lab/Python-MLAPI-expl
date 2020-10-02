#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import functools, datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


TRAIN_PATH = '/kaggle/input/facial-keypoints-detection/training/training.csv'
TEST_PATH = '/kaggle/input/facial-keypoints-detection/test/test.csv'
LOOKUP_PATH = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'
BATCH_SIZE = 256


# In[ ]:


train_df = pd.read_csv(TRAIN_PATH)


# In[ ]:


print('num of cols {}'.format(len(train_df.columns)))
train_df.columns


# In[ ]:


sample_img = np.array(train_df['Image'][0].split(' '), dtype=np.float32).reshape(96, 96)


# In[ ]:


sample_img


# In[ ]:


y = train_df.drop('Image', axis=1)
t = y.iloc[0].values


# In[ ]:


plt.imshow(sample_img, cmap='gray')
plt.scatter(t[0::2], t[1::2], c='red', marker='x')


# In[ ]:


train_df.isnull().any(axis=0).value_counts()
train_df.fillna(method='ffill', inplace=True)


# In[ ]:


train_df.isnull().any(axis=0).value_counts()


# In[ ]:


training_data = train_df['Image'].values
labels = train_df.drop('Image', axis=1)


# In[ ]:


FEATURES = list(labels.columns)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(training_data, labels, random_state=42, test_size=0.1)


# In[ ]:


len(X_train)


# ## Data pipeline

# In[ ]:


def process_img(X, y=None):
    imgs = [np.array(i.split(' '), dtype=np.float32).reshape(96, 96, 1) for i in X]
    imgs = [img / 255.0 for img in imgs]
    return np.array(imgs), y


# In[ ]:


def data_pipeline(X, y, shuffle_size):
    dataset = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(shuffle_size)
        .batch(BATCH_SIZE)
        .prefetch(1)
        .repeat()
    )
#     print('Dataset element spec {}'.format(dataset.element_spec))
    iterator= dataset.make_one_shot_iterator()
    return iterator


# In[ ]:


train_shuffle_size = len(X_train)
test_shuffle_size = len(X_test)


# In[ ]:


X_train, y_train = process_img(X_train, y_train.values)
X_test, y_test = process_img(X_test, y_test.values)


# In[ ]:


X_train.shape


# In[ ]:


y_test.shape


# In[ ]:


train_iterator = data_pipeline(X_train, y_train, train_shuffle_size)


# In[ ]:


validation_iterator = data_pipeline(X_test, y_test, test_shuffle_size)


# ## Define model

# In[ ]:


Activation='elu'
Input = tf.keras.layers.Input
Conv2d = functools.partial(
            tf.keras.layers.Conv2D,
            activation=Activation,
            padding='same'
        )
BatchNormalization = tf.keras.layers.BatchNormalization
AveragePooling2D = tf.keras.layers.AveragePooling2D
MaxPooling2D = tf.keras.layers.MaxPool2D
Dense = functools.partial(
            tf.keras.layers.Dense,
            activation=Activation
        )
Flatten = tf.keras.layers.Flatten


# In[ ]:


def prepare_model():
    input = Input(shape=(96, 96, 1,))
    conv_1 = Conv2d(16, (2, 2))(input)
    batch_norm_1 = BatchNormalization()(conv_1)
    
    conv_2 = Conv2d(32, (3, 3))(batch_norm_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    
    conv_3 = Conv2d(64, (4, 4))(batch_norm_2)
    avg_pool_1 = AveragePooling2D((2,2))(conv_3)
    batch_norm_3 = BatchNormalization()(avg_pool_1)
    
    conv_128 = Conv2d(128, (4, 4))(batch_norm_2)
    avg_pool_128 = AveragePooling2D((2,2))(conv_3)
    batch_norm_128 = BatchNormalization()(avg_pool_1)
    
    conv_4 = Conv2d(64, (7, 7))(batch_norm_128)
    avg_pool_1 = AveragePooling2D((2, 2))(conv_128)
    batch_norm_4 = BatchNormalization()(avg_pool_128)
    
    conv_5 = Conv2d(32, (7, 7))(batch_norm_4)
    flat_1 = Flatten()(conv_5)
    
    dense_1 = Dense(30)(flat_1)
    outputs = Dense(30)(dense_1)
    
    model = tf.keras.Model(input, dense_1)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


# In[ ]:


model = prepare_model()


# In[ ]:


model.summary()


# ## Callbacks

# In[ ]:


logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True),
    tensorboard_callback,
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
]


# In[ ]:


model.fit(train_iterator, steps_per_epoch=24, epochs=50, validation_data=validation_iterator, validation_steps=6,callbacks=callbacks)


# In[ ]:





# ## Performance

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[ ]:





# ## Inferance

# In[ ]:


test_df = pd.read_csv(TEST_PATH)
lookup_df = pd.read_csv(LOOKUP_PATH)


# In[ ]:


test_imgs, y = process_img(test_df['Image'])


# In[ ]:


test_imgs.shape


# In[ ]:


predictions = model.predict(test_imgs)


# In[ ]:


predictions.shape


# In[ ]:


plt.imshow(test_imgs[0].reshape(96, 96), cmap='gray')
plt.scatter(predictions[0][0::2], predictions[0][1::2], c='red', marker='x')


# In[ ]:





# In[ ]:


locations = []
rows = []


# In[ ]:


for row_id, img_id, feature_name, loc in lookup_df.values:
    fi = FEATURES.index(feature_name)
    loc = predictions[img_id - 1][fi]
    locations.append(loc)
    rows.append(row_id)


# In[ ]:


row_id_series = pd.Series(rows, name='RowId')
loc_series = pd.Series(locations, name='Location')


# In[ ]:


sub_csv = pd.concat([row_id_series, loc_series], axis=1)


# In[ ]:


sub_csv.to_csv('face_key_detection_submission.csv',index = False)

