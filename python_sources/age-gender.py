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

import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tnrange
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from keras.models import Model
from keras.utils import plot_model,to_categorical
from keras.layers import Conv2D,Dense,Flatten,BatchNormalization,Add,Activation,Dropout,            GlobalAveragePooling2D,MaxPooling2D,AveragePooling2D
from keras.engine.input_layer import Input
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import backend as K


# In[ ]:


df = pd.read_csv("../input/celeba-dataset/list_attr_celeba.csv",usecols=['image_id','Male','Young'])


# In[ ]:


path = "../input/celeba-dataset/img_align_celeba/img_align_celeba/"
# images = os.listdir("../input/celeba-dataset/img_align_celeba/img_align_celeba/")
# images = images[:5000]


# In[ ]:


df.columns


# In[ ]:


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None : 
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[ ]:


image_list = []
label1 = []
label2 = []
images = list(df.values)
for i in tnrange(10000):
    image_list.append(convert_image_to_array(path+str(images[i][0])))
    if images[i][1]==-1:
        label1.append(0)
    else:
        label1.append(1)  
        
    if images[i][2]==-1:
        label2.append(0)
    else:
        label2.append(1)  


# In[ ]:


x_train = image_list[:8000]
x_test = image_list[8000:]

y_train1 = label1[:8000]
y_test1 = label1[8000:]

y_train2 = label2[:8000]
y_test2 = label2[8000:]


# In[ ]:


# print("[INFO] Spliting data to train, test")
# # x_train, x_test, y_train, y_test = train_test_split(image_list, labels, test_size=0.2, random_state = 42) 


# In[ ]:


y_train1 = to_categorical(y_train1)
y_test1 = to_categorical(y_test1)

y_train2 = to_categorical(y_train2)
y_test2 = to_categorical(y_test2)


# In[ ]:


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)
y_train2 = np.array(y_train2)
y_test2 = np.array(y_test2)


# In[ ]:


inputShape = (218, 178, 3)
inputs = Input(shape=inputShape)
x = inputs
x = Conv2D(32,(3,3),padding='SAME',kernel_initializer='random_uniform',input_shape=inputShape,activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))
x = MaxPooling2D((2,2))(x)
x = Conv2D(32,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))

x = Conv2D(64,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))

x = Conv2D(128,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))

x = Conv2D(192,(5,5),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))
x = MaxPooling2D((2,2))(x)
x = Conv2D(192,(5,5),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))

x = Conv2D(256,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))
x = MaxPooling2D((3,3))(x)
x = Conv2D(256,(3,3),padding='SAME',kernel_initializer='random_uniform',activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# model.add(Activation('relu'))

out1 = Flatten()(x)
out1 = Dense(512)(out1)
out1 = BatchNormalization()(out1)
out1 = Activation('relu')(out1)
out1 = Dropout(0.3)(out1)
out1 = Dense(2)(out1)
out1 = Activation('softmax')(out1)

out2 = Flatten()(x)
out2 = Dense(512)(out2)
out2 = BatchNormalization()(out2)
out2 = Activation('relu')(out2)
out2 = Dropout(0.3)(out2)
out2 = Dense(2)(out2)
out2 = Activation('softmax')(out2)

model = Model(inputs,[out1,out2])


# In[ ]:


# aug = ImageDataGenerator(
#     rotation_range=25, width_shift_range=0.1,
#     height_shift_range=0.1, shear_range=0.2, 
#     zoom_range=0.2, 
#     fill_mode="nearest")


# In[ ]:


# history = model.fit_generator(
#     aug.flow(x_train, y_train, batch_size=64),
#     validation_data=(x_test, y_test),
#     steps_per_epoch=len(x_train) // 64,
#     epochs=20, verbose=1
#     )
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
best_model = ModelCheckpoint('cnn_weights.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
h = model.fit(x_train,[y_train1,y_train2],validation_data=(x_test,[y_test1,y_test2]),epochs=15,callbacks=[best_model])


# In[ ]:


inc_model = InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   include_top = False,
                   input_shape=(218,178,3))


# In[ ]:


x = inc_model.output
# x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(2048, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)


# In[ ]:


model1 = Model(inc_model.input,predictions)
best_model = ModelCheckpoint('inc_weights.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# In[ ]:


for layer in model1.layers[:311]:
    layer.trainable = False


# In[ ]:


model1.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[ ]:


model1.fit(x_train,y_train1,epochs=20,validation_data=(x_test,y_test1),batch_size=32,callbacks=[best_model])

# history = model.fit_generator(
#     aug.flow(x_train, y_train, batch_size=64),
#     validation_data=(x_test, y_test),
#     steps_per_epoch=len(x_train) // 64,
#     epochs=20, verbose=1
#     )


# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:




model_json = model1.to_json()
with open("model_inc.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:




