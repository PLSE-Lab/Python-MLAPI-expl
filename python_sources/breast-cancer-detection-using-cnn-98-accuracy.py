#!/usr/bin/env python
# coding: utf-8

# # Importing Neccessary Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# # Loading the Built-in Sklearn Breast Cancer Dataset

# In[ ]:


cancerData = datasets.load_breast_cancer()


# In[ ]:


X = pd.DataFrame(data = cancerData.data, columns=cancerData.feature_names )
X.head()


# In[ ]:


y = cancerData.target


# In[ ]:


X.shape


# # Splitting into Train and Test datasets

# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1,stratify=y)


# In[ ]:


X_train.shape


# In[ ]:


y_test.shape


# # Applying StandardScaler()

# In[ ]:


scaler = StandardScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Reshaping the dataset to 3-D to pass it through CNN

# In[ ]:


X_train = X_train.reshape(512,30,1)
X_test = X_test.reshape(57,30,1)


# # Preparing the Model

# In[ ]:


model = Sequential()
model.add(Conv1D(filters=16,kernel_size=2,activation='relu',input_shape=(30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(32,2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train,epochs=35,verbose=1,validation_data=(X_test,y_test))


# # Plots of Accuracy and Loss

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


plotLearningCurve(history,35)


# In[ ]:




