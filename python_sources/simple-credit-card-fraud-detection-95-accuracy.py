#!/usr/bin/env python
# coding: utf-8

# # Importing Neccessary Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization,Dropout,Dense,Flatten,Conv1D
from tensorflow.keras.optimizers import Adam


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# # Gathering the data and assessing the data

# In[ ]:


df = pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.Class.unique()


# # Uneven class distribution

# In[ ]:


df.Class.value_counts()


# In[ ]:


nf = df[df.Class==0]
f = df[df.Class==1]


# # Extracting random entries of class-0
# # Total entries are 1.5* NO. of class-1 entries

# In[ ]:


nf = nf.sample(738)


# # Creating new dataframe

# In[ ]:


data = f.append(nf,ignore_index=True)


# In[ ]:


data.shape


# In[ ]:


X = data.drop(['Class'],axis=1)
y=data['Class']


# # Train-Test Split

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)


# In[ ]:


X_train.shape,X_test.shape


# # Applying StandardScaler to obtain all the features in similar range

# In[ ]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:


y_train=y_train.to_numpy()
y_test=y_test.to_numpy()


# # Reshaping the input to 3D.

# In[ ]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# # CNN model

# In[ ]:


model=Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64,2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.summary()


# # Compiling and Fiting

# In[ ]:


model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))


# In[ ]:


def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()


# In[ ]:


plotLearningCurve(history,20)


# In[ ]:




