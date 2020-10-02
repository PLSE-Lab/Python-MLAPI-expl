#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


# # Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print("Training set dimension: {}".format(train.shape))
print("Test set dimension: {}".format(test.shape))
print(train.info())
print(train.head())


# In[ ]:


test.head()


# In[ ]:


X_train = train.loc[:,'pixel0':'pixel783']
y_train = train.loc[:,'label']
X_test = test.copy()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# In[ ]:


print("Number of categories: {}".format(len(y_train.value_counts())))
print(y_train.value_counts().sort_index())


# # Visualization

# Each image has 28 pixels height and width. Each pixel indicates brightness, 0 to 255, with higher number meaning darker.<br>
# pixel0 to pixel27 is the first row of an image, and pixel28 to pixel 55 is the second row of an image.

# In[ ]:


temp = X_train.loc[0,:]
temp = temp.values.reshape((28, 28)) 

label = y_train[0]

plt.figure()
plt.imshow(temp)
plt.colorbar()
plt.grid(False)
plt.title("True label is {}".format(label))
plt.show()


# # Preprocess

# In[ ]:


print("Maximum pixel value is {}".format(max(X_train.max())))
print("Minimum pixel value is {}".format(min(X_train.min())))


# Scale the values to be 0 to 1.

# In[ ]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[ ]:


print("Maximum pixel value is {}".format(max(X_train.max())))
print("Minimum pixel value is {}".format(min(X_train.min())))


# # Model

# In[ ]:


# build
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu, input_shape = (28*28,)))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))


# In[ ]:


# compile
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


# # Training

# In[ ]:


EPOCH = 5

model.fit(X_train, y_train, epochs = EPOCH, verbose = 1, validation_split = 0.3)


# # Prediction of training set

# In[ ]:


pred_train = model.predict(X_train)


# In[ ]:


pred_train[0]


# In[ ]:


np.argmax(pred_train[0])


# In[ ]:


# true label
y_train[0]


# In[ ]:


pred_train[1]


# In[ ]:


np.argmax(pred_train[1])


# In[ ]:


# true label
y_train[1]


# The location of prediction is exactly the predicted label!

# # Prediction of test set

# In[ ]:


pred_test = model.predict(X_test)


# In[ ]:


pred_test.shape


# In[ ]:


pred_test[0]


# In[ ]:


np.argmax(pred_test[0])


# # Submission

# In[ ]:


test_id = np.arange(1, X_test.shape[0]+1,1)
test_id


# In[ ]:


predictions = np.argmax(pred_test, axis = 1)


# In[ ]:


print(test_id.shape)
print(predictions.shape)


# In[ ]:


sub = pd.DataFrame(data = {'ImageId':test_id,
                           'Label':predictions})


# In[ ]:


sub.head()

