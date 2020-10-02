#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[ ]:


import tensorflow as tf
import os,cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# # Loading data

# In[ ]:


f = []
for dits,_,filenames in os.walk('/kaggle/input/stanford-dogs-dataset/images/Images'):
    f.append(dits)


# In[ ]:


def load_data(label,data_dir,imgsize):
    for img in tqdm(os.listdir(data_dir)):
        path = os.path.join(data_dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(imgsize,imgsize))
        
        X.append(np.array(img))
        y.append(str(label))


# # Loading images of first 20 dog breeds

# In[ ]:


X = []
y = []
imgsize = 150
for a in f[1:21]:
    load_data(a[37:].replace('-','_'),a,imgsize)


# In[ ]:


pd.Series(y).unique()


# In[ ]:


len(pd.Series(y).unique())


# In[ ]:


le= LabelEncoder()
y = le.fit_transform(y)


# # One hot encoding target variables

# In[ ]:


from keras.utils.np_utils import to_categorical
y = to_categorical(y,20)


# # Normalizing the data

# In[ ]:


X = np.array(X)
X = X/255


# # Train test splitting the data

# In[ ]:



x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)


# In[ ]:


X.shape


# In[ ]:


def load_model():
  model = Sequential()
  model.add(Conv2D(32,3,padding='same',activation='relu',input_shape=[150, 150, 3]))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Dropout(0.5))
  model.add(Conv2D(32,3,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Dropout(0.5))
  model.add(Conv2D(64,3,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Conv2D(64,3,padding='same',activation='relu'))
  model.add(Dropout(0.3))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Conv2D(64,3,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(units=128,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(units=64,activation='relu'))
  model.add(Dense(units=20,activation='softmax'))

  print(model.summary())
  return model


# In[ ]:


def plot_accuracy(history):
 plt.plot(history.history['accuracy'])
 plt.plot(history.history['val_accuracy'])
 plt.title('model accuracy')
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 plt.legend(['train', 'test'], loc='upper left')
 plt.show()
def plot_losses(history):
 plt.plot(history.history['loss'])
 plt.plot(history.history['val_loss'])
 plt.title('model loss')
 plt.ylabel('loss')
 plt.xlabel('epoch')
 plt.legend(['train', 'test'], loc='upper left')
 plt.show()


# # Adam

# In[ ]:


model = load_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
history = model.fit(x_train,y_train,validation_split=0.2,epochs=150)


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


plot_accuracy(history)


# In[ ]:


plot_losses(history)


# # SGD

# In[ ]:


model = load_model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])

history = model.fit(x_train,y_train,validation_split=0.2,epochs=200)


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


plot_accuracy(history)


# In[ ]:


plot_losses(history)


# # Trainsfer Learning

# In[ ]:


tf_model = tf.keras.applications.ResNet50(include_top = False,input_shape=(150,150,3),weights='imagenet')
tf_model.trainable = True


# In[ ]:


tf_model.output


# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(tf_model.output)
prediction_layer = Dense(units=20, activation='softmax')(global_average_layer)
model_tf = tf.keras.models.Model(inputs=tf_model.input, outputs=prediction_layer)
model_tf.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


history = model_tf.fit(x_train,y_train, epochs=50, validation_split=0.2)


# In[ ]:


plot_accuracy(history)


# In[ ]:


plot_losses(history)


# In[ ]:


model_tf.evaluate(x_test,y_test)


# Any Suggestions are welcome.. 
# 
# Thank you.
