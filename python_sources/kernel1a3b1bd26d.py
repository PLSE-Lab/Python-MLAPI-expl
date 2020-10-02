#!/usr/bin/env python
# coding: utf-8

# # Initialization

# ## Load libraries

# In[117]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print('Setup completed.')


# ## Import data

# In[89]:


train_df_path = '../input/train.csv'
test_df_path = '../input/test.csv'
train_df = pd.read_csv(train_df_path, index_col='label')
test_df  = pd.read_csv(test_df_path)
train_df = train_df / 255
test_X  = test_df / 255
print('Import completed.')


# !['MNIST Digits Dataset'](https://miro.medium.com/fit/c/1838/551/1*Riqqoa7vKHXnFHvaGfDFjA.png)

# # Exploration

# In[79]:


print(train_df.shape)
train_df.describe()


# In[80]:


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x=np.arange(10), y=train_df.index.value_counts(sort=False), palette='Blues')
ax.set(xlabel='Digit', ylabel='Counts')


# In[81]:


plt.figure(figsize=(12,6))
for i in range(10):  
    plt.subplot(2, 5, i+1)
    plt.imshow(train_df.iloc[i].values.reshape(28,28))
    plt.title(train_df.index[i])
plt.show()


# Are there any null values in the data set?

# In[82]:


train_df.isnull().sum(axis = 0).sum()


# # Modeling

# In[83]:


train_X, valid_X, train_y, valid_y = train_test_split(train_df.values, train_df.index, test_size=0.75)


# In[92]:


train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
valid_X = valid_X.reshape(valid_X.shape[0], 28, 28, 1)
test_X = test_X.values.reshape(test_X.shape[0], 28, 28, 1)


# In[96]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[97]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[99]:


model.fit(train_X, train_y,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(valid_X, valid_y))


# # Predictions

# In[106]:


predictions_valid = model.predict(valid_X)


# In[115]:


num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(4*num_rows,4*num_cols))
for i in range(num_images):  
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(valid_X[i][:,:,0])
    plt.title('Predicted: '+str(predictions_valid[i].argmax()))
plt.show()


# In[130]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(predictions_valid.argmax(axis=1), valid_y), square=True, annot=True, cmap='Blues')


# # Submission

# In[133]:


predictions = model.predict(test_X)
predictions = predictions.argmax(axis=1)
submission=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submission.to_csv("submission.csv", index=False, header=True)


# In[ ]:




