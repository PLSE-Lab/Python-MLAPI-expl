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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Importing The REquired Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Lambda,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import RMSprop,Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau


# In[ ]:


train=pd.read_csv('../input/indian-dance-form-recognition/dataset/train.csv')
test=pd.read_csv('../input/indian-dance-form-recognition/dataset/test.csv')


# In[ ]:


base='../input/indian-dance-form-recognition/dataset'
train_dir=os.path.join(str(base)+'/train/')
test_dir=os.path.join(str(base)+'/test/')


# In[ ]:


train_fnames=os.listdir(train_dir)
test_fnames=os.listdir(test_dir)


# In[ ]:


print(train_fnames[:9])
print(test_fnames[:9])


# In[ ]:


img_width=224
img_height=224


# In[ ]:


def train_data_preparation(list_of_images,train,train_dir):
    x=[]#Array of images
    y=[]# labels
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(train_dir+image),(img_width,img_height),interpolation=cv2.INTER_CUBIC))
        if image in list(train['Image']):
            y.append(train.loc[train['Image']==image,'target'].values[0])
    return x,y


# In[ ]:


import cv2
training_data,training_labels=train_data_preparation(train_fnames,train,train_dir)


# In[ ]:


def test_prepare_data(list_of_images,test_dir):
    x=[]
    for image in list_of_images:
        
        x.append(cv2.resize(cv2.imread(test_dir+image),(224,224),interpolation=cv2.INTER_CUBIC))
    return x    


# In[ ]:


testing_data=test_prepare_data(test_fnames,test_dir)


# In[ ]:


def show_batch(image_batch,image_label):
    plt.figure(figsize=(12,12))
    for n in range(30):
        ax=plt.subplot(6,6,n+1)
        plt.imshow(image_batch[n])
        plt.title(image_label[n].title())
        plt.axis('off')


# In[ ]:


show_batch(training_data,training_labels)


# In[ ]:


le=LabelEncoder()
training_labels=le.fit_transform(training_labels)


# In[ ]:


training_labels


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(training_data,training_labels,test_size=0.33,random_state=42)


# In[ ]:


#Data Augmentation
train_datagenerator=ImageDataGenerator(rescale=1./255,
                                      featurewise_center=False,
                                      samplewise_center=False,
                                      rotation_range=40,
                                      zoom_range=0.20,
                                      width_shift_range=0.10,
                                       height_shift_range=0.10,
                                       horizontal_flip=True,
                                       vertical_flip=False)

test_datagenerator=ImageDataGenerator(rescale=1./255)


train_datagenerator.fit(X_train)
test_datagenerator.fit(X_test)
test_datagenerator.fit(testing_data)

X_train=np.array(X_train)
testing_data=np.array(testing_data)
X_test=np.array(X_test)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#Training Using Transfer Learning
vggmodel=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3),pooling='max')


# In[ ]:


vggmodel.summary()


# In[ ]:


vggmodel.trainable=False
model=Sequential([
    vggmodel,
    Dense(units=1024,activation='relu',kernel_initializer='uniform'),
    Dropout(0.25),
    Dense(units=512,activation='relu'),
    Dropout(0.25),
    Dense(units=8,activation='softmax')
])


# In[ ]:


reduce_learning_rate=ReduceLROnPlateau(monitor='loss',
                                      factor=0.1,
                                      patience=2,
                                      cooldown=2,
                                      min_lr=0.01,
                                      verbose=1)


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history=model.fit_generator(train_datagenerator.flow(X_train,to_categorical(y_train,8),batch_size=16),validation_data=test_datagenerator.flow(X_test,to_categorical(y_test,8),batch_size=16),verbose=2,epochs=15)


# In[ ]:


plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.legend(loc='best')
plt.show()


# In[ ]:


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.legend(loc='best')
plt.show()


# In[ ]:


predictions=model.predict(testing_data)


# In[ ]:


predictions


# In[ ]:


predictions=[np.argmax(i) for i in predictions]


# In[ ]:


predictions


# In[ ]:


target=le.inverse_transform(predictions)


# In[ ]:


target


# In[ ]:


submission = pd.DataFrame({ 'Image': test.Image, 'target': target })


# In[ ]:


submission.to_csv('output2.csv', index=False)


# In[ ]:


submission


# In[ ]:




