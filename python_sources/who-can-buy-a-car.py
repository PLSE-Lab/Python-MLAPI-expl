#!/usr/bin/env python
# coding: utf-8

# This dataset has information of customer like age,annual salary,annual salary,credit score.Based on this parameters we will try to predict the car buying capacity of customers.This informartion can be used for targeted marketing.This kernel is a work in process, if you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing the Python Modules **

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# **Importing the data set**

# In[ ]:


df=pd.read_csv('../input/car-purchase-data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')


# In[ ]:


df.head()


# **Vizualising the dataset**

# In[ ]:


sns.pairplot(df)
plt.ioff()


# From the pair plot we can see that Age,Annual salary have direct correlation with Car Purchase AmountBut Credit Card Debt and Net Worth Doent have much correlation between the car purchase amount.Many rich people actually dont buy expensive cars.

# **Cleaning and Creating the training and test data**

# In[ ]:


X=df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)


# We are dropping the columns Customer Name,Cutomer email and Country as these parameters donnt have an affect on Car buying capacity of the customers 

# In[ ]:


#X


# In[ ]:


y=df['Car Purchase Amount']


# In[ ]:


#y


# In[ ]:


X.shape


# In[ ]:


y.shape


# While using artificial neural network we need to scaled the values in X to get accurate results.We can do this using sklearn as shown below

# **NOrmalising Input **

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)


# In[ ]:


X_scaled


# We can see now that all the columns of X have the values in the range 0 to 1

# In[ ]:


X_scaled.shape


# In[ ]:


scaler.data_max_


# In[ ]:


scaler.data_min_


# **Normalising the output **

# In[ ]:


y=y.values.reshape(-1,1)


# In[ ]:


y.shape


# In[ ]:


y_scaled=scaler.fit_transform(y)


# In[ ]:


#y_scaled


# **Training the model **

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.15)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


import tensorflow.keras 
from keras.models import Sequential 
from keras.layers import Dense 

model=Sequential()
model.add(Dense(40,input_dim=5,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='linear'))


# In[ ]:


model.summary()


# Param 150 =25 * 25 Inputs +25 bials values

# In[ ]:


model.compile(optimizer='adam',loss='mean_squared_error')


# In[ ]:


epochs_hist=model.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)


# **Model Evaluation**

# In[ ]:


epochs_hist.history.keys()


# In[ ]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Training and Validation Loss')
plt.title('Model Loss Progress during training')
plt.legend(['Training Loss','Validation Loss'])
plt.ioff()


# We can see that by using 20 epochs we can acheive good accuracy

# **Predicting the Car buying capacity **

# In[ ]:


# Gender,Age,Annual Salary,Credit card debt,Net Worth 
X_test=np.array([[1,50,50000,10000,600000]])
y_predict=model.predict(X_test)


# In[ ]:


print('Expected Purchase Amount',y_predict)

