#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as npimg


# In[ ]:


train_dir = "/kaggle/input/food11-image-dataset/training"
validation_dir  = "/kaggle/input/food11-image-dataset/validation"
evaluation_dir = "evaluation"


# In[ ]:


train_data = []
for path,d,files in os.walk(train_dir):
    for file in files:
        train_data.append(path+"/"+file)


# In[ ]:


img = cv2.imread(train_data[2])
plt.imshow(img)
plt.show()


# In[ ]:


train_df = pd.DataFrame({"images":train_data}) 


# In[ ]:


def get_cat(img_path):
    return (img_path.partition('/')[-1].rpartition('/')[0]).lower()

train_df['category'] = train_df['images'].apply(get_cat)


# In[ ]:


num_classes = len(train_df['category'].unique())


# In[ ]:


def image_preprocess(img):
    img = npimg.imread(img)
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(128,128))
    img = img/255
    return img


# In[ ]:


image  = train_df['images'][0]
original_image = npimg.imread(image)
preprocessed_image=image_preprocess(image)

fig,axs=plt.subplots(1,2,figsize=(6,3))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")

axs[1].imshow(preprocessed_image)
axs[1].set_title("Preprocessed Image")


# In[ ]:


le  = LabelEncoder()
train_df['labels'] = le.fit_transform(train_df['category'])
X = train_df['images'].values
y = to_categorical(train_df['labels'])


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)
print("Training samples: {}\nValid Samples: {}".format(len(X_train),len(X_test)))


# In[ ]:


X_train = np.array(list(map(image_preprocess,X_train)))
X_test = np.array(list(map(image_preprocess,X_test)))


# In[ ]:





# In[ ]:


plt.imshow(X_train[random.randint(0,len(X_train)-1)])
plt.axis("off")
print(X_train.shape[1:])


# In[ ]:


def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# In[ ]:


model = createModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 150
epochs = 50

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(X_test, y_test))


# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["traning data","validation data"])
plt.title("Loss")
plt.xlabel("epoch")


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["traning data","validation data"])
plt.title("Accuracy")
plt.xlabel("epoch")


# In[ ]:


### With data argumentation


# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=45,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


# In[ ]:


img_height, img_width,channels = 128,128,3


# In[ ]:


train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=50
)


# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = validation_datagen.flow_from_directory(
    validation_dir, 
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=50
)


# In[ ]:


history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=3430//batch_size,
    steps_per_epoch=9866//batch_size)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["traning data","validation data"])
plt.title("Loss")
plt.xlabel("epoch")


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["traning data","validation data"])
plt.title("Accuracy")
plt.xlabel("epoch")


# In[ ]:




