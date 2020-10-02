#!/usr/bin/env python
# coding: utf-8

# Here i crated a model that predicts cat breed with 82% accuracy

# In[ ]:


import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D,Dropout
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


df = pd.read_csv('../input/cats-and-dogs-breeds-classification-oxford-dataset/annotations/annotations/list.txt')
info = df.head(4).copy()
df = df.loc[5:,]
df[['CLASS-ID','SPECIES','BREED','ID']] = df['#Image CLASS-ID SPECIES BREED ID'].str.split(expand=True) 
df = df.drop('#Image CLASS-ID SPECIES BREED ID',axis=1)
df = df.rename(columns={"CLASS-ID": "image", "SPECIES": "CLASS-ID", 'BREED' : "SPECIES", "ID":"BREED ID"})
df[["CLASS-ID","SPECIES","BREED ID"]] = df[["CLASS-ID","SPECIES","BREED ID"]].astype(int)
df= df[df['SPECIES']==1] #cats
df['image'] = df['image'].apply(lambda x : str(x)+'.jpg')
df = df[['image','BREED ID']]
df = df.reset_index()
df = df.drop('index',axis=1)
df['BREED ID'] = df['BREED ID'].astype('str')


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    validation_split=0.2,
                                    rotation_range=90,
                                    width_shift_range=0.2, 
                                   height_shift_range=0.2)


validation_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255)


# In[ ]:


train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory='../input/cats-and-dogs-breeds-classification-oxford-dataset/images/images',
        x_col='image',
        y_col='BREED ID',
        target_size=(350, 350),
        batch_size=64,
        class_mode="categorical",
        seed=123,
        #validate_filenames = False,
        subset='training'
)

valid_gen_flow = validation_datagen.flow_from_dataframe(
        dataframe=df,
        directory='../input/cats-and-dogs-breeds-classification-oxford-dataset/images/images',
        x_col='image',
        y_col='BREED ID',
        target_size=(350, 350),
        batch_size=64,
        class_mode="categorical",
        seed=123,
        #validate_filenames = False,
        subset='validation')


# In[ ]:


features, target = next(train_gen_flow)

fig = plt.figure(figsize=(10,10))
for i in range(10):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(features[i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


# In[ ]:


optimizer= Adam()

backbone = ResNet50(input_shape=(350,350,3),weights='imagenet', include_top=False)
#backbone.trainable = False
model = Sequential()
model.add(backbone)
model.add(GlobalAveragePooling2D())
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['acc'])
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=25),
]


# In[ ]:


def train_model(model, train_data, test_data,batch_size=None,epochs=250, steps_per_epoch =None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(train_data,
          validation_data=test_data,
          steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps,
          verbose=1, epochs=epochs, callbacks=my_callbacks)
    return model


# In[ ]:


train_model(model,train_gen_flow,valid_gen_flow)


# chek it up

# In[ ]:


result = model.predict(features[[22]]) 
result = result.tolist()


# In[ ]:


breeds = { "1": "Abyssinian", "2":"Bengal", "3":"Birman", "4":"Bombay","5":"British Shorthair", "6":"Egyptian Mau", "7":"Maine Coon","8":"Persian", "9":"Ragdoll","10":"Russian Blue", "11":"Siamese","12":"Sphynx"}


# In[ ]:


indices = valid_gen_flow.class_indices
indices = indices.keys()
final_breed = []
for i in indices:
    final_breed.append(breeds.get(i))
final_breed


# In[ ]:


sorted(zip(result, final_breed), reverse=True)[:3]


# In[ ]:


def predict_breed(imgage):
    img = cv2.imread(image)
    img = cv2.resize(img,(350,350))
    img = np.reshape(img,[1,350,350,3])
    img = img/255.
    classes = model.predict([img])


# In[ ]:


model.save("model.h5")


# In[ ]:


import cv2
img = cv2.imread('../input/myface/photo_2020-06-26_14-35-07.jpg')
img = cv2.resize(img,(300,300))
img = np.reshape(img,[1,300,300,3])
img = img/255.

result = model.predict([img])


# In[ ]:


sorted(zip(result[0], final_breed), reverse=True)[:3]

