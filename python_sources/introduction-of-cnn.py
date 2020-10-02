#!/usr/bin/env python
# coding: utf-8

# # Introduction of CNN 

# ## Load Dataset

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y = train['label']
X = train.drop(labels=["label"], axis=1)


# ## Preprocess the data

# In[ ]:


X = X / 255.0
test = test / 255.0


# In[ ]:


from sklearn.model_selection import train_test_split
random_seed=0
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models


# In[ ]:


plt.figure()
Xshow = X_train.values.reshape(-1,28,28,1)
plt.imshow(Xshow[0][:,:,0])
plt.colorbar()
plt.grid(False)


# ## Build the model

# ### Simple Convnet

# In[ ]:


model = models.Sequential()
model.add(layers.SeparableConv2D(32,(3,3), activation="elu", input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(64,(3,3), activation="elu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.SeparableConv2D(64,(3,3), activation="elu"))


# In[ ]:


model.summary()


# ### Convnet + Classifier

# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="elu"))
model.add(layers.Dense(10, activation="softmax"))


# In[ ]:


model.summary()


# ### Compile the model

# In[ ]:


model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ### Train the model

# In[ ]:


X_val = X_val.values.reshape(-1,28,28,1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
X = X.values.reshape(-1,28,28,1)


# In[ ]:


from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)
Y = to_categorical(Y, num_classes = 10)


# In[ ]:


model.fit(X, Y, epochs=20)


# In[ ]:


Y_val = to_categorical(Y_val, num_classes = 10)


# In[ ]:


test_loss, test_acc = model.evaluate(X_val, Y_val)
print("Test Accuracy: {}".format(test_acc))


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


# In[ ]:




