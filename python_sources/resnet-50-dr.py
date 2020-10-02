#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv")
data.head()


# In[ ]:


data.head()


# In[ ]:


data['image_name'] = [i+".jpeg" for i in data['image'].values]
data.head()


# In[ ]:


data['level'].hist()
data['level'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train, val = train_test_split(data, test_size=0.15)


# In[ ]:


train.shape, val.shape


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


import cv2
def load_ben_color(image):
    IMG_SIZE = 224
    sigmaX=10
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image


# In[ ]:


data_gen = ImageDataGenerator(rescale=1/255.,
                              zoom_range=0.15,
                              fill_mode='constant',
                              cval=0.,
                              horizontal_flip=True,
                              vertical_flip=True,
                              preprocessing_function=load_ben_color)


# In[ ]:


# batch size
bs = 32

train_gen = data_gen.flow_from_dataframe(train, 
                                         "../input/diabetic-retinopathy-resized/resized_train/resized_train/",
                                         x_col="image_name", y_col="level", class_mode="raw",
                                         batch_size=bs,
                                         target_size=(224, 224))
val_gen = data_gen.flow_from_dataframe(val,
                                       "../input/diabetic-retinopathy-resized/resized_train/resized_train/",
                                       x_col="image_name", y_col="level", class_mode="raw",
                                       batch_size=bs,
                                       target_size=(224, 224))


# ### ResNet50

# In[ ]:


from keras.applications.resnet50 import ResNet50
import keras.layers as L
from keras.models import Model


# In[ ]:


base_model = ResNet50(weights='../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   include_top=False,
                   input_shape=(224, 224, 3))
x = base_model.output
x = L.GlobalMaxPooling2D()(x)
x = L.BatchNormalization()(x)
x = L.Dropout(0.2)(x)
x = L.Dense(1024, activation="relu")(x)
x = L.Dropout(0.1)(x)
x = L.Dense(64, activation="relu")(x)
predictions = L.Dense(5, activation='softmax')(x)


# In[ ]:


model = Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


for layer in base_model.layers[:-20]: layer.trainable = False


# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
model_chk = ModelCheckpoint("vgg16_model.h5", save_best_only=True, monitor="val_accuracy")
reduce_lr = ReduceLROnPlateau()


# In[ ]:


model.fit_generator(train_gen, train_gen.n // bs,
                    validation_data=val_gen, validation_steps=val_gen.n // bs,
                    epochs=30, workers=4, callbacks=[model_chk])


# In[ ]:


model.evaluate_generator(val_gen, steps=val_gen.n/bs, verbose=1)


# In[ ]:


from keras.models import load_model
model = load_model("vgg16_model.h5")


# test some instances

# In[ ]:


from PIL import Image
im = Image.open("../input/diabetic-retinopathy-resized/resized_train/resized_train/" + val.iloc[0].image_name)
im = np.array(im.resize((224, )*2, resample=Image.LANCZOS))
im.shape


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(im)


# In[ ]:


plt.imshow(load_ben_color(im))


# In[ ]:


print("predicted:", np.argmax(model.predict(load_ben_color(im).reshape(1, *im.shape))[0]))
print("actual:", val.iloc[0].level)


# In[ ]:




