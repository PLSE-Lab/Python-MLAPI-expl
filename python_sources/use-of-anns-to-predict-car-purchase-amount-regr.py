#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# We would develop a regression model using Artificial Neural Network to predict the total dollar amount that customers are willing to pay to purchase a car given the following attributes:
# 
# * Customer Name
# * Customer e-mail
# * Country
# * Gender
# * Age
# * Annual Salary 
# * Credit Card Debt 
# * Net Worth 
# 
# **The model should predict: Car Purchase Amount **
# 

# **Importing the data**

# In[ ]:


dataset = pd.read_csv('../input/car-purchase-data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')


# In[ ]:


dataset.head()


# In[ ]:


import seaborn as sns
sns.pairplot(dataset)


# **Dropping unnecessary columns**

# In[ ]:


X=dataset.drop(["Customer Name", "Customer e-mail","Country","Car Purchase Amount"],axis=1)


# In[ ]:


print(X)


# In[ ]:


y = dataset["Car Purchase Amount"]
print(y)
y.shape


# In[ ]:


y = y.values.reshape(-1,1)
y.shape


# **Normalizing the values to improve the accuracy**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)
print(X_scaled)


# > **Training the Model**

# In[ ]:


from sklearn.model_selection import train_test_split
X_trian,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size=0.3)


# In[ ]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the model
model=Sequential()
# adding layers
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
# printing the summary
model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='mean_squared_error')


# In[ ]:


epochs_hist = model.fit(X_trian,y_train,batch_size=25,epochs=50,validation_split=.2,verbose=1)


# **Validating the model**

# In[ ]:


print(epochs_hist.history.keys())


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


np.set_printoptions(precision=2)
y_test= scaler.inverse_transform(y_test)
y_pred= scaler.inverse_transform(y_pred)
df=np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)),1)


# In[ ]:


final = pd.DataFrame(data=df, columns=["Predicted", "Actual"])
print(final)

