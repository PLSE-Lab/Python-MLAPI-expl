#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten , Dense


# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import io
df = pd.read_csv(io.BytesIO(uploaded['advertising.csv']))


# In[ ]:


df.head()


# In[ ]:


df.isnull().any()


# In[ ]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[ ]:


#outliers
from sklearn.model_selection import train_test_split
x = df.drop(labels = ['Ad Topic Line','City','Timestamp'],axis = 1)
y = ['Clicked on Ad']


# In[ ]:



x = df.drop('Country',axis = 1)


# In[ ]:


len(df['Ad Topic Line'].unique())


# In[ ]:


x.head(5)


# In[ ]:


#feature standardization , however sucky the features are ewwwwww shity
#the scales are varying much and neural network cant work easily in this so we have to preprocess the feaures


# In[ ]:


x = df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = df['Clicked on Ad']
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[ ]:


x_train


# In[ ]:


y_train .to_numpy


# In[ ]:


model = Sequential()
model.add(Dense(x.shape[1],activation='relu',input_dim = x.shape[1]))
model.add(Dense(128,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train,y_train,batch_size=10,epochs=20,verbose=1,validation_split=0.2)


# In[ ]:


y_pred = model.predict_classes(x_test)


# In[ ]:


model.evaluate(x_test,y_test.to_numpy())


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
boost = GradientBoostingClassifier()
boost.fit(x_train,y_train.to_numpy())
print(boost.score(x_test,y_test.to_numpy()))


# PLOTTING LEARNING CURVE

# In[ ]:


history.history


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','val'])


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','val'])


# confusion matrix

# In[ ]:


get_ipython().system('pip install mlxtend')


# In[ ]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


mat = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=mat)


# In[ ]:



plot_confusion_matrix(conf_mat=mat,show_normed=True)


# DEALING WITH TIMESTAMPS

# In[ ]:


df.head()


# In[ ]:


df['Timestamp']


# In[ ]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# In[ ]:


df['Timestamp']


# In[ ]:


df['year'] = df['Timestamp'].dt.year
df['month'] = df['Timestamp'].dt.month
df['day'] = df['Timestamp'].dt.day
df['week'] = df['Timestamp'].dt.week
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['hour'] = df['Timestamp'].dt.hour
df['minute'] = df['Timestamp'].dt.minute


# In[ ]:


df.head(2)


# In[ ]:





# In[ ]:


#feature important
feature = df[['year','month','day','week','day_of_week','hour','minute']]
target = df['Clicked on Ad']
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(feature,target)
print(model.feature_importances_)


# In[ ]:


# some eda
import seaborn as sns
sns.jointplot(x = df['minute'], y =df['Clicked on Ad'])


# In[ ]:


sns.jointplot(x = df['day_of_week'],y = df['Clicked on Ad'],kind = 'hex')


# In[ ]:


df.columns


# In[ ]:


x_new = df[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage',   'Male', 
         'month', 'day', 'week',
       'day_of_week', 'hour', 'minute']]
y_new = df['Clicked on Ad']


# In[ ]:


x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(x_new,y_new,test_size = 0.2,stratify = y_new)
x_train_new = scaler.fit_transform(x_train_new)
x_test_new = scaler.fit_transform(x_test_new)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
boost = GradientBoostingClassifier()
boost.fit(x_train_new,y_train_new)
print(boost.score(x_test_new,y_test_new))


# In[ ]:


model = Sequential()
model.add(Dense(x.shape[1],activation='relu',input_dim = x.shape[1]))
model.add(Dense(128,activation='relu',kernel_initializer='glorot_uniform'))
model.add(Dense(256,activation='tanh',kernel_initializer='glorot_uniform'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train,y_train,batch_size=10,epochs=30,verbose=1,validation_split=0.2)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','val'])


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','val'])


# In[ ]:


#handling string data


# In[ ]:




