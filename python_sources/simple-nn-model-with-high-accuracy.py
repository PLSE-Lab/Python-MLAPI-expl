#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


# load datasets
X_train = pd.read_csv("../input/train.csv")
X_train, Y_train = X_train.iloc[:, 2:], X_train.iloc[:, 1]

# convert text labels into one-hot vector
lb = preprocessing.LabelEncoder()
lb.fit(Y_train)
Y_oh = lb.transform(Y_train)
Y_oh = np.eye(len(np.unique(Y_oh)))[Y_oh]

# split datasets into random train and dev subset
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_oh, test_size = 0.3)


# In[ ]:


from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.models import Model


# In[ ]:


input_shape = X_train.shape[-1]
output_shape = Y_oh.shape[-1]

X_input = Input((input_shape, ))
X = Dense(128, activation = "relu")(X_input)
X = BatchNormalization()(X)
X = Dropout(0.5)(X)
X_output = Dense(output_shape, activation = "softmax")(X)

model = Model(inputs = X_input, outputs = X_output)


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])


# In[ ]:


train_history = model.fit(X_train, Y_train, epochs = 100)


# In[ ]:


plt.subplot(1, 2, 1)
plt.plot(train_history.history["acc"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.subplot(1, 2, 2)
plt.plot(train_history.history["loss"])
plt.title("Loss")
plt.xlabel("epoch")


# In[ ]:


model.evaluate(X_dev, Y_dev)


# In[ ]:


# load test set
X_test = pd.read_csv("../input/test.csv")
X_test, X_id = X_test.iloc[:, 1:], X_test.iloc[:, 0]


# In[ ]:


# prediction
pred = model.predict(X_test)


# In[ ]:


# convert prediction into required format
df = pd.DataFrame(pred)
df.columns = lb.classes_
df = pd.concat([X_id, df], axis = 1)


# In[ ]:


# save as csv
df.to_csv("submission.csv", index = False)


# In[ ]:




