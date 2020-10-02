#!/usr/bin/env python
# coding: utf-8

# # Deep Learning in Keras

# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### Feedforward Neural Network

# In[36]:


mnist = pd.read_csv("../input/train.csv")
train, cv = train_test_split(mnist, test_size=0.1)
X_train, y_train = train.drop("label", axis=1).values / 255, train.label.values
X_cv, y_cv = cv.drop("label", axis=1).values / 255, cv.label.values
y_train_ohe = to_categorical(y_train)


# In[37]:


inputs = Input(shape=(28 * 28,))
X = Dense(300, activation="relu")(inputs)
X = Dropout(0.85)(X)
X = Dense(200, activation="relu")(X)
X = Dense(100, activation="relu")(X)
outputs = Dense(10, activation="softmax")(X)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_train,  y_train_ohe, epochs=50, batch_size=1000)


# In[38]:


y_hat_cv = np.argmax(model.predict(X_cv), axis=1)
accuracy_score(y_cv, y_hat_cv)


# In[41]:


X_test = pd.read_csv("../input/test.csv").values / 255
y_test = np.argmax(model.predict(X_test), axis=1)


# In[42]:


y_test_out = pd.DataFrame(y_test)
y_test_out.index = range(1, len(y_test) + 1)
y_test_out.index.name = "ImageId"
y_test_out.columns = ["Label"]
y_test_out.head()


# In[7]:


y_test_out.to_csv("MNIST_test_pred.csv", header=True)

