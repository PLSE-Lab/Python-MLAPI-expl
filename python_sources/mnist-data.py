#!/usr/bin/env python
# coding: utf-8

# ## Loading Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical   


# In[ ]:


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


Y = np.array(train["label"])
print(Y.shape)
Y = to_categorical(Y)
print(Y.shape)


# In[ ]:


del train["label"]


# In[ ]:


X = train.values


# In[ ]:


img = X[14]
img = np.array(img, dtype = 'float')
img = img.reshape(28, 28)

plt.imshow(img)
plt.show()


# # --------------------------------------------------------------------------------------

# ## Data Reshaping for CNN

# In[ ]:


X.shape


# In[ ]:


X = X.reshape(X.shape[0], 28, 28, 1)


# In[ ]:


X.shape


# ## Setting up Model

# In[ ]:


model1 = Sequential()
model2 = Sequential()


# In[ ]:


cnn1 = Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = (28, 28, 1))
pool1 = MaxPool2D(pool_size = (2,2))
cnn2 = Conv2D(64, (3,3), padding = "same", activation = "relu")
pool2 = MaxPool2D(pool_size = (2,2))
drop1 = Dropout(0.2)


# In[ ]:


flatten = Flatten()
dense1 = Dense(units = 128, activation = "relu")
drop2 = Dropout(0.1)
dense2 = Dense(units = 10, activation = "softmax")


# In[ ]:


model1.add(cnn1)
model1.add(pool1)
model1.add(cnn2)
model1.add(pool2)
model1.add(drop1)
model1.add(flatten)
model1.add(dense1)
model1.add(drop2)
model1.add(dense2)


# In[ ]:


model2.add(cnn1)
model2.add(pool1)
model2.add(cnn2)
model2.add(pool2)
model2.add(drop1)
model2.add(flatten)
model2.add(dense1)
model2.add(dense2)


# In[ ]:


model1.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[ ]:


model2.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# ## Training data and observing result

# In[ ]:


# model1.fit(X, Y, epochs = 20, batch_size = 128)


# In[ ]:


model2.fit(X, Y, epochs = 5, batch_size = 128)


# ## Preparing Test Data and Getting Preds

# In[ ]:


X_test = np.array(test.values)


# In[ ]:


X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[ ]:


X_test.shape


# In[ ]:


# predictions1 = model1.predict(X_test)
predictions2 = model2.predict(X_test)


# In[ ]:


# pred1 = np.argmax(predictions1, axis = 1)
pred2 = np.argmax(predictions2, axis = 1)


# In[ ]:


# s1 = pd.DataFrame(pred1)
s2 = pd.DataFrame(pred2)


# In[ ]:


labels = np.array([i for i in range(1, 28001)])


# In[ ]:


# del s1["ImageI"]
# s1.insert(0, "ImageId", labels, True)


# In[ ]:


s2.insert(0, "ImageId", labels, True)


# In[ ]:


# s1.rename(columns = {0:'Label'}, inplace = True)


# In[ ]:


s2.rename(columns = {0:'Label'}, inplace = True)


# In[ ]:


# s1.to_csv("submit1.csv", index=False)


# In[ ]:


s2.to_csv("submit2.csv", index=False)

