#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf


# # Read Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Separate Training Data and Scaling

# In[ ]:


y = train['label'].values
y.shape


# In[ ]:


X = train.iloc[:, 1:].values
X.shape


# In[ ]:


X = X / 255


# # Build Model

# In[ ]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# # Training

# In[ ]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10)


# # Predict

# In[ ]:


prediction = model.predict(test.values)


# In[ ]:


prediction.shape


# In[ ]:


single = [np.argmax(_) for _ in prediction]


# # Make Submission

# In[ ]:


result = pd.DataFrame({
    "ImageId": list(range(1,len(prediction)+1)), 
    "Label": single
})


# In[ ]:


result.to_csv("submission.csv", index=False, header=True)


# In[ ]:




