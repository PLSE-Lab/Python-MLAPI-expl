#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[ ]:


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")

print(f"test size {test.shape}")
print(f"train size {train.shape}")


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


sample_submission.head(3)


# In[ ]:


X = train.iloc[:,1:].values
Y = train.iloc[:,0].values
val = test.values


# In[ ]:


Y = tf.keras.utils.to_categorical(Y)
Y.shape


# In[ ]:


X_flattened = X.reshape(X.shape[0],28,28,1)
val_flattened = val.reshape(val.shape[0],28,28,1)


# In[ ]:


x_rescaled = X_flattened/255
val_rescaled = val_flattened/255


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x_rescaled,Y,train_size=0.75,random_state=10)


# In[ ]:


plt.imshow(x_train[50][:,:,0])


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding="same",activation="relu",input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32,(3,3),padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),padding="same",activation="relu"),
    tf.keras.layers.Conv2D(64,(3,3),padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


# In[ ]:


history = model.fit(x_train,y_train,epochs=25,validation_data=[x_test,y_test])


# In[ ]:


predictions = model.predict_classes(val_rescaled)
predictions[:10]


# In[ ]:


sample_submission['Label'] = predictions
sample_submission.head(3)


# In[ ]:


sample_submission.to_csv('submission.csv',index=False)


# In[ ]:




