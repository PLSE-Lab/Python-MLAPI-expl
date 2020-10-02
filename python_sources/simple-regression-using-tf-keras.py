#!/usr/bin/env python
# coding: utf-8

# ## **Simple Regression using Tensorflow **

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler


# In[ ]:


data_train = pd.read_csv('../input/random-linear-regression/train.csv')
data_train.head()


# In[ ]:


data_test = pd.read_csv('../input/random-linear-regression/test.csv')
data_test.head()


# **Clear Dataset**

# In[ ]:


data_train = data_train.dropna()
data_test = data_test.dropna()


# **Analyze features**

# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot(x='x',y='y',data=data_train);
plt.plot(data_train,data_train,'r');


#  **Standardization**

# In[ ]:


X = np.array(data_train.drop('y',axis=1).values)
y = np.array(data_train['y'].values)

Tx = np.array(data_test.drop('y',axis=1).values)
Ty = np.array(data_test['y'].values)

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
Tx = scaler.fit_transform(Tx)


# ** Create Keras Model **

# In[ ]:


model = Sequential()
model.add(Dense(1,input_shape=(1,), activation='relu'))


# **Compile Our Model**

# In[ ]:


model.compile(tf.keras.optimizers.SGD(
    learning_rate=0.1,
    name="SGD",     
    nesterov=True
),loss=MSE)


# **Create Early Stopping (To Prevent OverFit)**

# In[ ]:


early_stop = EarlyStopping(mode='min',monitor='val_loss',patience=50,verbose=1)
model.summary()


# In[ ]:





# In[ ]:





# **Train Model using Fit function**

# In[ ]:


model.fit(X,y,validation_data=(Tx,Ty),epochs=100,verbose=1,callbacks=[early_stop]);


# In[ ]:


loss = pd.DataFrame(model.history.history)
loss.plot(figsize=(12,5))


# Predict Values

# In[ ]:


y_predict = model.predict(Tx);


# # **Prediction**

# In[ ]:


prediction_frame =pd.DataFrame(Ty,columns=["True Y"])
prediction_frame['Prediction']  = y_predict
plt.figure(figsize=(12,8))
sns.scatterplot(data=prediction_frame);


# **DATAFRAME**

# In[ ]:


prediction_frame


# > Dont forget to upvote
