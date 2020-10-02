#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognization | MNIST Data
# 
# A model to recognize and predict handwritten digits using **Tensorflow**, **Keras** and **CNN**.

# ## MNIST Data
# ----
# The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets.
# 
# ![Mnist Data](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fanalyticsindiamag.com%2Fwp-content%2Fuploads%2F2017%2F12%2FMNIST-dataset.jpg&f=1&nofb=1)
# 
# 

# **Tensorflow**:- 
# 
# TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks
# 
# **Keras**:-
# 
# Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow. It focuses on being user-friendly, modular, and extensible

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


df_train.head()


# - Label:- it contain the actual number.

# # Normalizing our Data
# ---
# In statistics and applications of statistics, normalization can have a range of meanings. In the simplest cases, normalization of ratings means adjusting values measured on different scales to a notionally common scale, often prior to averaging.
# 
# 
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.kdnuggets.com%2Fwp-content%2Fuploads%2Fconvolutional-neural-net-architecture-2.jpg&f=1&nofb=1)

# In[ ]:


x_train = np.array(df_train.iloc[:,1:])
x_train = np.array([np.reshape(i, (28, 28, 1)) for i in x_train])
y_train = np.array(df_train.iloc[:,0])


# In[ ]:


x_train = x_train/255.0
y_train = keras.utils.to_categorical(y_train)


# In[ ]:


x_test = np.array(df_test)
x_test = np.array([np.reshape(i, (28, 28, 1)) for i in x_test])
x_test = x_test/255.0


# # Let's Visualize what do we have here.
# 
# Sample plots.

# In[ ]:


# 6 random plots

img_indices = [random.randint(0,33600) for i in range(6)] 
n=0
fig = plt.figure(figsize=[15,10])
axes = fig.subplots(2, 3)
for row in range(2):
    for col in range(3):
        axes[row,col].imshow((x_train[img_indices[n]]).reshape((28,28)), cmap='Accent')
        n += 1


# ## Splitting data for training and testing purpose

# # Lets start with building the model
# 
# <img src="https://blog.imarticus.org/wp-content/uploads/2020/04/deep.gif" height=400 width=700>
# 
# ## Convolutional Neural Networks
# ___
#     In deep learning, a convolutional neural network is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks, based on their shared-weights architecture and translation invariance characteristics.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)


# In[ ]:


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='random_uniform', padding='same', activation='relu', input_shape=(X_train.shape[1:])))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters=128, kernel_size=(7,7), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(7,7), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))

model.add(keras.layers.Conv2D(filters=256, kernel_size=(7,7), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))


model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=y_train.shape[1], activation='softmax'))


print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
filepath = "model.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rlp = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.2)


# In[ ]:


# Configure and train the model
history = model.fit(X_train, Y_train,  batch_size=500, callbacks=[es, ckpt, rlp], epochs=100, validation_data=(X_test,Y_test))


# # Predicted Number Vs Actual Number
# 
# Lets plot and see how well is our model's prediction.

# In[ ]:


# 40 random plots to test our model

img_indices = [random.randint(0,1000) for i in range(40)]
n=0
fig = plt.figure(figsize=[30,50])
axes = fig.subplots(10, 4)
for row in range(10):
    for col in range(4):
        axes[row,col].imshow((X_test[img_indices[n]][:,:,0]).reshape((28,28)), cmap='Accent')
        predicted_num = np.argmax(model.predict(X_test[img_indices[n]:img_indices[n]+1]))
        actual_num = np.argmax(Y_test[img_indices[n]])
        axes[row,col].set_title("{}. Predicted = {} | Actual = {}".format(n+1, predicted_num, actual_num), fontsize=15)
        n += 1


# # Let's Save and submit what we made.

# * Saving out predictions to `submission.csv`

# In[ ]:


id_img = []
label = []
for i in range(len(x_test)):
    id_img.append(i+1)
    label.append(np.argmax(model.predict(x_test[i:i+1])))
    
img_id = np.array(id_img)
label = np.array(label)


# In[ ]:


op_df = pd.DataFrame()
op_df['ImageId'] = img_id
op_df['Label'] = label
op_df.to_csv("submission.csv", index=False)


# # Request: - 
# If you find this kernel interesting and learns something from it please don't forget to upvote. Also, write your question in comments below lets start the discussion.
# ![](https://nelottery.com/media/email_alerts/2019/12_19/research/thankyou.gif)
