#!/usr/bin/env python
# coding: utf-8

# # Data Preparation
# Load the training data with pandas and explore it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(f'The train set contain {train.shape[0]} examples')
train.head(3)


# In[ ]:


y_train = train['label']
X_train = train.drop(labels = ["label"],axis = 1)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


digits = y_train.unique()
count = y_train.value_counts()

plt.bar(digits, count)
plt.title('Train set')
plt.xlabel('Digit')
plt.ylabel('Count')


# # Normalization

# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0


# # Reshape

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[ ]:


get_ipython().system('pip install tensorflow==2.0.0-alpha0')


# In[ ]:



import tensorflow as tf


# In[ ]:


print(tf.__version__)


# In[ ]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


# In[ ]:


y_train = to_categorical(y_train, num_classes = 10)


# In[ ]:


plt.imshow(X_train[100][:,:,0])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state= 101)


# # CNN Model 

# In[ ]:


from tensorflow import keras


# In[ ]:


from tensorflow.keras import layers, models


# In[ ]:


model = models.Sequential()


# In[ ]:


model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = "softmax"))


# In[ ]:


model.summary()


# In[ ]:


optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


## Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# Fit the model
history = model.fit(X_train, y_train, batch_size = 100, epochs = 10, validation_data = (X_test, y_test))


# # Validation

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
plt.figure(figsize=(24,8))
plt.subplot(1,2,1)
plt.plot(history.history["val_accuracy"], label="validation_accuracy", c="red", linewidth=4)
plt.plot(history.history["accuracy"], label="training_accuracy", c="green", linewidth=4)
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["val_loss"], label="validation_loss", c="red", linewidth=4)
plt.plot(history.history["loss"], label="training_loss", c="green", linewidth=4)
plt.legend()
plt.grid(True)

plt.suptitle("ACC / LOSS",fontsize=18)

plt.show()


# In[ ]:


print('Train accuracy of the model: ',history.history['accuracy'][-1])


# In[ ]:


print('Train loss of the model: ',history.history['loss'][-1])


# In[ ]:


print('Validation accuracy of the model: ',history.history['val_accuracy'][-1])


# In[ ]:


print('Validation loss of the model: ',history.history['val_loss'][-1])


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="BuPu",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


results = model.predict(test)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist.csv",index=False)

