#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer

# ###### CNN on classic dataset of handwritten images

# ### Importing important libraries

# In[ ]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf


# ### Reading Datasets

# ##### Train Dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


train.describe()


# ##### Test Dataset

# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


test.describe()


# ### Exploratory Data Analsysis

# #### Checking for NULL values

# In[ ]:


train.melt(id_vars="label")['value'].isnull().sum()


# In[ ]:


test.melt()['value'].isnull().sum()


# #### Count of each labels

# In[ ]:


train['label'].value_counts().sort_index()


# In[ ]:


# Plot
plt.figure(figsize=(8, 4))
sns.set_style("whitegrid")
sns.countplot(x="label", data=train)
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()


# #### Dividing the training dataset into X and y

# In[ ]:


y = train['label']
X = train.drop(['label'], axis = 1)


# #### Normalize the pixel data, i.e. converting values from 0 - 254 to 0 - 1 

# In[ ]:


X = X / 255
test = test / 255


# #### Converting labels to numpy array

# In[ ]:


y = np.array(y)


# #### Reshaping image to 28px X 28px dimension

# In[ ]:


X = X.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[ ]:


plt.imshow(X[0][:,:,0])


# ### Modelling CNN

# #### Creating train and test datasets

# In[ ]:


random_seed = 4


# In[ ]:


# Split the train and test set for the fitting
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state=random_seed)


# ### Tensorflow Model

# In[ ]:


# Tensorflow Keras CNN Model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = train_X.shape[1:]))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu"))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# #### Optimizer and loss function

# In[ ]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# In[ ]:


model.summary()


# #### Fitting the train dataset

# In[ ]:


model.fit(train_X, train_y, epochs=3)


# #### Finding the loss and accuracy of the model

# In[ ]:


val_loss, val_acc = model.evaluate(test_X, test_y) 


# In[ ]:


val_acc


# ### Predicting the submission dataframe

# #### Fitting the full train data

# In[ ]:


model.fit(X, y, epochs=3)


# #### Predicting on given test data

# In[ ]:


test_pred = model.predict(test)


# In[ ]:


submission = pd.DataFrame()
submission['ImageId'] = range(1, (len(test)+1))
submission['Label'] = np.argmax(test_pred, axis=1)


# In[ ]:


submission.head()


# In[ ]:


submission.shape


# #### Saving in a csv file

# In[ ]:


submission.to_csv("submission.csv", index=False)

