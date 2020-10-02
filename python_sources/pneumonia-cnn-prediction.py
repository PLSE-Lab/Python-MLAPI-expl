#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Inspired from https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification/data#Import-Library
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../input/chest-xray-pneumonia"))


# In[ ]:


#Defining constants
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# In[ ]:


#Preparing training data
train_folders=["../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA","../input/chest-xray-pneumonia/chest_xray/train/NORMAL"]
categories={}
for i in train_folders:
    filenames = os.listdir(i)
    for filename in filenames:
        category = filename.split('.')[0]
        if "bacteria" in str(category) or "virus" in str(category):
            categories[filename]=1
        else:
            categories[filename]=0
 

train_df = pd.DataFrame(categories.items(), columns=['filename','category'])
print(train_df)


# In[ ]:


#Preparing test data
test_folders=["../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA","../input/chest-xray-pneumonia/chest_xray/test/NORMAL"]
categories={}
for i in test_folders:
    filenames = os.listdir(i)
    for filename in filenames:
        category = filename.split('.')[0]
        if "bacteria" in str(category) or "virus" in str(category):
            categories[filename]=1
        else:
            categories[filename]=0
 

test_df = pd.DataFrame(categories.items(), columns=['filename','category'])
print(test_df)


# In[ ]:


sample = random.choice(os.listdir("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA"))
image = load_img("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"+sample)
plt.imshow(image)


# Input Layer: It represent input image data. It will reshape image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.
# Conv Layer: This layer will extract features from image.
# Pooling Layer: This layerreduce the spatial volume of input image after convolution.
# Fully Connected Layer: It connect the network from a layer to another layer
# Output Layer: It is the predicted values layer.

# In[ ]:


#Building the CNN model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have normal and pneumonia

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[ ]:


#Setting callbacks

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#To prevent over fitting we will stop the learning after 10 epochs
earlystop = EarlyStopping(patience=10) 


#We will reduce the learning rate when then accuracy not increase for 2 steps

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]


# Because we will use image genaretor with class_mode="categorical". We need to convert column category into string. Then imagenerator will convert it one-hot encoding which is good for our classification.
# 
# So we will convert 1 to pneumonia and 0 to normal

# In[ ]:


train_df["category"] = train_df["category"].replace({0: 'normal', 1: 'pneumonia'}) 
test_df["category"] = test_df["category"].replace({0: 'normal', 1: 'pneumonia'}) 


# In[ ]:


train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# In[ ]:


print(train_df)


# In[ ]:


train_df['category'].value_counts().plot.bar()


# In[ ]:


total_train = train_df.shape[0]
test_train = test_df.shape[0]
batch_size=15


# In[ ]:


print(train_df)


# In[ ]:


df1=train_df.copy()
print(df1)


# In[ ]:


print(len(df1))


# I cannot pass multiple directories flow_from_dataframe function.
# we are creating a dataframe that has all the images with their absolute paths defined.

# In[ ]:




for i in range(len(df1)):
    if "bacteria" in str(df1.iloc[i,0]) or "virus" in str(df1.iloc[i,0]):
        df1.iloc[i,0]=os.path.join("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/",df1.iloc[i,0])
    else:
        df1.iloc[i,0]=os.path.join("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/",df1.iloc[i,0])
        


# In[ ]:


print(df1.iloc[5215,0])    


# In[ ]:


print(os.getcwd())


# In[ ]:


#Training generator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
        df1, 
        #"../input/chest-xray-pneumonia/chest_xray/train/NORMAL/", 
        directory=None,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size   
    )


# In[ ]:


df2=test_df.copy()
for i in range(len(df2)):
    if "bacteria" in str(df2.iloc[i,0]) or "virus" in str(df2.iloc[i,0]):
        df2.iloc[i,0]=os.path.join("../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/",df2.iloc[i,0])
    else:
        df2.iloc[i,0]=os.path.join("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/",df2.iloc[i,0])
        


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    df2, 
    directory=None,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[ ]:


example_df = df1.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    directory=None, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[ ]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[ ]:


#Fit the model
FAST_RUN = False 
#setting it to true since it would take more time for 50 epochs. But 50 would give the best accuracy rate
#trying with 20 epochs
epochs=3 if FAST_RUN else 20
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_train//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[ ]:


#Prepare Validation Data
val_normal_filenames = os.listdir("../input/chest-xray-pneumonia/chest_xray/val/NORMAL")
val_df = pd.DataFrame({
    'filename': val_normal_filenames
})


val_pneumonia_filenames = os.listdir("../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA")
val_df1 = pd.DataFrame({
    'filename': val_pneumonia_filenames
})

val_df=val_df.append(val_df1,ignore_index = True) 
nb_samples = val_df.shape[0]
print(val_df)


# In[ ]:



for i in range(len(val_df)):
    if "bacteria" in str(val_df.iloc[i,0]) or "virus" in str(val_df.iloc[i,0]):
        val_df.iloc[i,0]=os.path.join("../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/",val_df.iloc[i,0])
    else:
        val_df.iloc[i,0]=os.path.join("../input/chest-xray-pneumonia/chest_xray/val/NORMAL/",val_df.iloc[i,0])

print(val_df)


# In[ ]:


val_gen = ImageDataGenerator(rescale=1./255)
val_generator = val_gen.flow_from_dataframe(
    val_df, 
    directory=None,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=8,
    shuffle=False
)


# In[ ]:



predict = model.predict_generator(val_generator, steps=np.ceil(nb_samples/batch_size))


# For categoral classication the prediction will come with probability of each category. So we will pick the category that have the highest probability with numpy average max

# In[ ]:


print(len(np.argmax(predict, axis=-1)))


# In[ ]:


val_df['category'] = np.argmax(predict, axis=-1)


# In[ ]:


print(val_df)


# *We will convert the predict category back into our generator classes by using train_generator.class_indices. It is the classes that image generator map while converting data into computer vision*

# In[ ]:


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
val_df['category'] = val_df['category'].replace(label_map)


# In[ ]:


val_df['category'] = val_df['category'].replace({ 'Pneumonia': 1, 'Normal': 0 })


# In[ ]:


print(val_df)


# In[ ]:


#See predicted result with images


plt.figure(figsize=(12, 24))
for index, row in val_df.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel('(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# In[ ]:




