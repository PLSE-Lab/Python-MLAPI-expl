#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# Any results you write to the current directory are saved as output.


# In[ ]:


'''
def build_net():
   
    net = Sequential(name='DCNN')
    net.add(TimeDistributed(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            input_shape=(img_width, img_height, img_depth),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )))
    #48*48*64
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(TimeDistributed(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )))
    #48*48*64
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')))
    #24*24*64
    net.add(Dropout(0.4, name='dropout_1'))
    net.add(TimeDistributed(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )))
    #24*24*128
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(TimeDistributed(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )))
    net.add(BatchNormalization(name='batchnorm_4'))
    #24*24*128
    net.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')))
    #12*12*128
    net.add(Dropout(0.4, name='dropout_2'))
    net.add(TimeDistributed(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        ) ))
     #12,12,256
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(TimeDistributed(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )  ))
    
    #12,12,256
    net.add(BatchNormalization(name='batchnorm_6'))
    net.add(TimeDistributed(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')))
    
    #6,6,256
    net.add(Dropout(0.5, name='dropout_3'))
    net.add(TimeDistributed(Flatten(name='flatten')))
    
    '''
'''
    #(6*6*256)
    net.add(TimeDistributed(
        Dense(
            64*64,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )))
    
    net.add(BatchNormalization(name='batchnorm_7'))
    net.add(Reshape((64,64)))
    '''
'''
    net.add(LSTM(100,input_shape=(64,64)))
    net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )
        
    #net.summary()

    return net

'''


# In[ ]:




get_ipython().system(' pip install neural_structured_learning')


# In[ ]:


import math
import numpy as np
import pandas as pd

import scikitplot
import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import neural_structured_learning as nsl

from keras.utils import np_utils


# In[ ]:


import pandas as pd
df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')
print(df.shape)
df.head()


# In[ ]:


df.emotion.unique()


# In[ ]:


emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}


# In[ ]:


df.emotion.value_counts()


# In[ ]:


INTERESTED_LABELS = [3, 4, 6]


# In[ ]:


df = df[df.emotion.isin(INTERESTED_LABELS)]
df.shape


# In[ ]:


img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))


# In[ ]:


img_array.shape


# In[ ]:


img_array = np.stack(img_array, axis=0)


# In[ ]:


img_array.shape


# In[ ]:


le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)

img_labels = np_utils.to_categorical(img_labels)
img_labels.shape


# In[ ]:


le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape



# In[ ]:


img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]

num_classes = 3

X_train = X_train / 255.
X_valid = X_valid / 255.


# In[ ]:


input_shape=(img_width,img_height,img_depth)


# In[ ]:


# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator()


# In[ ]:


def build_net(input_shape):
    net = Sequential(name='DCNN')
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            input_shape=input_shape,
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        ))
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))

    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    net.add(BatchNormalization(name='batchnorm_4'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))

    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        ))
    net.add(BatchNormalization(name='batchnorm_6'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.5, name='dropout_3'))
    net.add(Flatten(name='flatten'))      
    net.add(
        Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        ))
    net.add(BatchNormalization(name='batchnorm_7'))
    net.add(Dropout(0.6, name='dropout_4'))
    net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        ))
    net.summary()
    return net


# In[ ]:


model = build_net(input_shape)

adv_config = nsl.configs.make_adv_reg_config(multiplier=0.3, adv_step_size=0.05)
adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)


# In[ ]:


batch_size = 32
epochs = 20

adv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
)


# In[ ]:


X_train.shape,y_train.shape,X_valid.shape,y_valid.shape


# In[ ]:


'''
adv_model.fit_generator(
    trainAug.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=valAug.flow(X_valid, y_valid),
    validation_steps=len(X_valid) // 32,epochs=epochs)
'''


# In[ ]:


def convert_to_dict_generator(image_data_gen):
    for image, label in image_data_gen:
        yield {'image': image, 'label': label} 


# In[ ]:



adv_model.fit_generator(
    convert_to_dict_generator(trainAug.flow(X_train, y_train, batch_size=batch_size)),
    validation_data=tf.data.Dataset.from_tensor_slices({'image': X_valid, 'label': y_valid}).batch(10),
    steps_per_epoch=10,
    epochs=5
)
'''
dataset = tf.data.Dataset.from_tensor_slices((x_test)) # I need to provide y_test also until version 1.12.0
dataset = dataset.batch(batch_size=10)
data = dataset.make_one_shot_iterator()
'''


# In[ ]:


out = model.predict_classes(X_valid)
print("total wrong  predictions", np.sum(np.argmax(y_valid, axis=1)) != out)


# In[ ]:


len(df)


# In[ ]:


print(classification_report(np.argmax(y_valid, axis=1), out))


# In[ ]:


import matplotlib.pyplot as plt
np.random.seed(0)
indices = np.random.choice(range(X_valid.shape[0]), size=15, replace=False)

fig = plt.figure(1, (9,30))

i = 0
for idx in indices:
    true_emotion = emotion_label_to_text[np.argmax(y_valid[idx])]
    pred_emotion = emotion_label_to_text[model.predict_classes(np.expand_dims(X_valid[idx], axis=0))[0]]
    
    for j in range(3):
        i += 1
        ax = plt.subplot(15,3,i)
        sample_img = X_valid[idx,:,:,0]
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t:{true_emotion}, p:{pred_emotion}")


# In[ ]:





# In[ ]:




