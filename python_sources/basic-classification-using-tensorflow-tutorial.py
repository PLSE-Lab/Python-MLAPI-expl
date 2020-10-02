#!/usr/bin/env python
# coding: utf-8

# # Basic Classification Using Tensorflow Tutorial

# ## Overview
# * Today, We will train our first neural network using Tutorial
# https://www.tensorflow.org/tutorials/keras/basic_classification

# ## Load Dataset

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y = train['label']
X = train.drop(labels=["label"], axis=1)


# ### Preprocess the data

# In[ ]:


X = X / 255.0
test = test / 255.0


# In[ ]:


from sklearn.model_selection import train_test_split
random_seed=0
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)


# ## Visualization

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[ ]:


plt.figure()
Xshow = X_train.values.reshape(-1,28,28,1)
plt.imshow(Xshow[0][:,:,0])
plt.colorbar()
plt.grid(False)


# In[ ]:


X_train.shape


# In[ ]:


plt.figure(figsize=(10,10))
class_names = ['0','1','2','3','4','5','6','7','8','9']
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Xshow[i][:,:,0], cmap=plt.cm.binary)


# ## Bulid the model

# ### Setup the layers

# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# ### Compile the model

# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


X_val = X_val.values.reshape(-1,28,28)
X_train = X_train.values.reshape(-1,28,28)
test = test.values.reshape(-1,28,28)


# ### Train the model

# In[ ]:


model.fit(X_train,Y_train, epochs=5)


# ### Evaluate accuracy

# In[ ]:


test_loss, test_acc = model.evaluate(X_val, Y_val)
print("Test Accuracy: {}".format(test_acc))


# ### Train all of Train dataset

# In[ ]:


X = X.values.reshape(-1,28,28)


# In[ ]:


model.fit(X,Y, epochs=5)


# ## Submission

# In[ ]:


predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission["Label"] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('./simpleMNIST.csv', index=False)

