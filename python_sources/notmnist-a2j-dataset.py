#!/usr/bin/env python
# coding: utf-8

# # This is Alphabates Prediction From (A-J) respect to (0-9) numbers.
# 
# 
# *   here, I have traied 112 images for each and every alphabate.
# *   This is not Mnist data set.
# 
# 

# **All Required Libraries to work done.**

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import pickle


# **i all cleaned my dataset so i just load it back
# if you want to know more about the cleaning,reshaping,and Type Conversion you can check out the
# https://github.com/rizwan777/ML-DS-DA/blob/master/Kaggle_work/NotMnist_A2J.ipynb file it will help you to batter understanding.

# In[ ]:


X_features = pickle.load(open("../input/cleandata/X_features.pickle","rb"))
Y_labels = pickle.load(open("../input/cleandata/Y_labels.pickle","rb"))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_features, Y_labels, test_size=0.2, random_state=13)


# In[ ]:


len(X_train),len(X_val),len(y_train),len(y_val)
# totally 1120 images From 896 image for X_train, 224 imge for X_val


# **Just Verifying the dataset is loaded properly and also its check for the shape.**

# In[ ]:


X_train[10].shape, y_train[10]


# **lets open any random image from the dataset**

# In[ ]:


plt.imshow(np.squeeze(X_train[10]),cmap="binary")
plt.show()
# yeah features of array with label match correctly .. here [A-J] match with[0-9]


# **In First Line we need to reshape the size of the image to fit well as input.
# we need to pass again three args overhear also otherwise its just make it single vector array.**

# In[ ]:


X_features = np.array(X_train).reshape(-1,100,100,1) 
# we need to pass again three args overhear also otherwise its just make it single vector array.
Y_labels = np.array(y_train).reshape(-1)
X_features.shape, Y_labels.shape


# In[ ]:


np.max(X_features[10])


# * **Below code will inable the GPU processing and it will used half of the memory to perform the operation.**

# In[ ]:


for i in Y_labels[20:30]:
    print(i)


# In[ ]:


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
# normalization of data for easy to calculations.
X_features = X_features/255.0


# In[ ]:


# i dont want to run and test now its take time to train the data.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Activation,MaxPool2D,Dropout,Flatten
model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',input_shape=X_features.shape[1:]))
model.add(Activation('tanh'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('tanh'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('tanh'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('tanh'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss="sparse_categorical_crossentropy",optimizer="RMSprop",metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history=model.fit(X_features,Y_labels,batch_size=64,epochs=20,validation_split=0.3)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


predict = model.predict([X_features])
predict, predict.shape


# In[ ]:


np.argmax(predict[666])  # lets test any number from the Prediction is accurate or not


# In[ ]:


plt.imshow(np.squeeze(X_features[666]),cmap="binary")
plt.show()


# In[ ]:




