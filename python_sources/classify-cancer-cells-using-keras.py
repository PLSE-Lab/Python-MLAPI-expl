#!/usr/bin/env python
# coding: utf-8

# In[28]:


# initiating gpu using tensorflow.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


# In[29]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras


# In[31]:


train = '../input/train/'
test = '../input/test/'
img_size1 = 96
img_size2 = 96


# In[32]:


cancer_files_train = pd.read_csv('../input/train_labels.csv')


# In[33]:


cancer_files_train.head()


# In[34]:


for i in range(len(cancer_files_train)):
        img_array = cv2.imread(os.path.join(train,cancer_files_train['id'][i]+'.tif'),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break


# In[35]:


# creating a training dataset.
training_data = []
i = 0
def create_training_data():
    for i in range(len(cancer_files_train)):
            img_array = cv2.imread(os.path.join(train,cancer_files_train['id'][i]+'.tif'),cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array,(img_size2,img_size1))
            training_data.append([
                new_img,cancer_files_train['label'][i]])


# In[36]:


# Creating a test dataset.
testing_data = []
i = 0
def create_testing_data():        
    for img in os.listdir(test):
        img_array = cv2.imread(os.path.join(test,img),cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array,(img_size2,img_size1))
        testing_data.append([img,
            new_img])


# In[37]:


create_training_data()


# In[38]:


create_testing_data()


# In[39]:


print(len(training_data))
print(len(testing_data))


# In[40]:


random.shuffle(training_data)


# In[41]:


x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1,img_size2,img_size1,1)


# In[42]:


X.shape


# In[43]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50)

Y_train = np_utils.to_categorical(y_train,num_classes=2)
Y_test = np_utils.to_categorical(y_test,num_classes=2)


# In[57]:


model = Sequential()


# In[58]:


model.add(Conv2D(32,kernel_size=(3,3),strides=1,activation='relu',input_shape=(img_size1,img_size1,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))


# In[59]:


model.add(Conv2D(64,kernel_size=(3,3),strides=2,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))


# In[60]:


model.add(Conv2D(128,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))


# In[61]:


model.add(Conv2D(256,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))


# In[62]:


model.add(Conv2D(512,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))


# In[63]:


model.add(Conv2D(1024,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))


# In[64]:


model.add(Flatten())
model.add(Dense(units = 512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units = 128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2,activation='softmax'))


# In[65]:


model.summary()


# In[66]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[67]:


callbacks = [EarlyStopping(monitor='val_acc',patience=5)]


# In[70]:


batch_size = 25
n_epochs = 100
results = model.fit(x_train,Y_train,batch_size=batch_size,epochs=n_epochs,verbose=1,validation_data=(x_test,Y_test))


# In[77]:


model.save_weights('./cell_classification_lr_weights.h5', overwrite=True)


# In[78]:


model.save('./cell_classification_lr.h5')


# In[73]:


model = load_model('../working/cell_classification_lr.h5')


# In[74]:


test_data = np.array(testing_data[0][1]).reshape(-1,img_size2,img_size1,1)


# In[75]:


preds = model.predict(test_data)


# In[76]:


preds


# In[ ]:




