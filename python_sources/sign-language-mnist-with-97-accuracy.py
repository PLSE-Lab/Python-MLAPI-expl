#!/usr/bin/env python
# coding: utf-8

# ### MNIST Image Classification on Sign Language
# * In this project I am using MNIST dataset of hand made sign language images to classify each image into a class of 26 alphabets from A-Z.
# * KAGGLE LINK: https://www.kaggle.com/datamunge/sign-language-mnist

# ![hand sign](https://storage.googleapis.com/kagglesdsdata/datasets%2F3258%2F5337%2Famer_sign2.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1592174518&Signature=RrlCY6TrfnUgr9AxNjlEyWsp845fJfKHr9ohA6GCgKLQISPdxPXttB9JunDC%2BRHjPBVSyVZQKk6dAPtnxNvqnZHr%2FVucF7Jjpjzfd83N0CHs%2BpVrxZ%2FvqDRq8I4ijBXFz%2F2XxWTDEIAWzt2%2FCx0EMrgvBCWguuLSXHlWLuNVX3luSUS4zHlPCUwNsejkecU88Gu%2BpHJCyz0R9F7pUk1wvhhZvbUrkr0au2%2BXa%2BGzH4VmE0VDzJmjr8sSx7imFh%2BfR%2BJUGnCM5gtLTFKS6nZCEI2WCgSwC%2B4oXKzWEbiOPT39K5%2Bab0jT6EIqmELgbvLv2IMTQdT3MM29B1k0Rc3z%2Fg%3D%3D)

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


mnist_train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")
mnist_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test.csv")


# In[ ]:


print(mnist_train.shape, mnist_test.shape)


# In[ ]:


mnist_train.head()


# In[ ]:


mnist_test.head()


# In[ ]:


print(mnist_train.isna().any().any(), mnist_test.isna().any().any())
# Data is completely clean without any missing values.


# In[ ]:


mnist_train_data = mnist_train.loc[:, "pixel1":]
mnist_train_label = mnist_train.loc[:, "label"]

mnist_test_data = mnist_test.loc[:, "pixel1":]
mnist_test_label = mnist_test.loc[:, "label"]


# In[ ]:


# Data Normalization
mnist_train_data = mnist_train_data/255.0
mnist_test_data = mnist_test_data/255.0


# ### Data Visualization

# In[ ]:


data_array = np.array(mnist_train_data.loc[2, :])
shaped_data = np.reshape(data_array, (28, 28))
sign_img = plt.imshow(shaped_data, cmap=plt.cm.binary)
plt.colorbar(sign_img)
print("IMAGE LABEL: {}".format(mnist_train.loc[2, "label"]))
plt.show()


# In[ ]:


sns.countplot(mnist_train.label)
print(list(mnist_train.label.value_counts().sort_index()))


# In[ ]:


mnist_train_data = np.array(mnist_train_data)
mnist_test_data = np.array(mnist_test_data)

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)
mnist_test_data = mnist_test_data.reshape(mnist_test_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D
from tensorflow.keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler


# In[ ]:


nclasses = mnist_train_label.max() - mnist_train_label.min() + 1
mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)
print("Shape of ytrain after encoding: ", mnist_train_label.shape)


# In[ ]:


nclasses = mnist_test_label.max() - mnist_test_label.min() + 1
mnist_test_label = to_categorical(mnist_test_label, num_classes = nclasses)
print("Shape of ytest after encoding: ", mnist_test_label.shape)


# ### Training model

# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(25, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model_history = model.fit(mnist_train_data, mnist_train_label, batch_size=500, shuffle=True, epochs=20, validation_split=0.1)


# In[ ]:


mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 784)
print(mnist_train_data.shape, mnist_train_label.shape)

vc_loss, vc_accuracy = model.evaluate(mnist_test_data, mnist_test_label)
print("\nLOSS: {}\nACCURACY: {}".format(vc_loss, vc_accuracy))


# In[ ]:


plt.plot(model_history.history['accuracy'],label = 'ACCURACY')
plt.plot(model_history.history['val_accuracy'],label = 'VALIDATION ACCURACY')
plt.legend()


# In[ ]:


plt.plot(model_history.history['loss'],label = 'TRAINING LOSS')
plt.plot(model_history.history['val_loss'],label = 'VALIDATION LOSS')
plt.legend()


# ### Final Result
# * LOSS: 13%
# * ACCURACY: 97%

# ### Please upvote if you find it hekpful! :)
