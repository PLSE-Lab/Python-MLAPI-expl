#!/usr/bin/env python
# coding: utf-8

# In this Kernel we will try to predict the price of house using machine learning.First we will study the affect of scaling the parameters on the accuracy of the model.This kernel is work in process.I will be updating the kernel in coming days.If you like my work please do vote.

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


# **Import Modules**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('../input/melbournehousingprices/melb_data.csv')
df.head()


# **Summary of Dataset**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# In[ ]:


#df.isnull().sum()


# **Addressing Missing Value**

# In[ ]:


df['BuildingArea'].fillna((df['BuildingArea'].mean()), inplace=True)


# Missing value in the Building area is updated by the mean area 

# **Dropping the Columns**

# In[ ]:


df1=df.drop(["YearBuilt","CouncilArea"],axis=1)


# We have dropped the Year built and Council Area column as many rows of data is missing.

# In[ ]:


df1[['Rooms','Bedroom2']].hist(figsize=(10,8),bins=10,color='b',linewidth='3',edgecolor='k')
df1[['BuildingArea','Landsize']].hist(figsize=(10,8),bins=20,color='b',linewidth='3',edgecolor='k',range=(1,1000))
#df1[['Landsize']].hist(figsize=(10,8),bins=10,color='b',linewidth='3',edgecolor='k',range=(1,100))
plt.tight_layout()
plt.show()


# **Matrix of Features**

# In[ ]:


X=df1[['Bedroom2','BuildingArea']].values
y=df1['Price'].values


# **Building Linear Regression Model in Keras**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[ ]:


model=Sequential()
model.add(Dense(1,input_dim=2))
model.compile(Adam(lr=0.8),'mean_squared_error')


# **Test Train Split**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


model.fit(X_train,y_train)


# The loss value is high as the the lose is difference of Square of Price.

# **Scaling the input to Improve Accuracy**

# In[ ]:


df1['BuildingArea1000']=df['BuildingArea']/1000
df1['Price100k']=df['Price']/1e5


# In[ ]:


X=df1[['Bedroom2','BuildingArea1000']].values
y=df1['Price100k'].values


# In[ ]:


#df1.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


model=Sequential()
model.add(Dense(1,input_dim=2))
model.compile(Adam(lr=0.8),'mean_squared_error')


# In[ ]:


model.fit(X_train,y_train)


# We can see that the loss has considerably reduced by Scaling the values of Building area and Size.

# **Accuracy of Model**

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train,y_train_pred)))
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_test,y_test_pred)))


# In[ ]:


model.fit(X_train,y_train,epochs=40,verbose=0)


# In[ ]:


y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train,y_train_pred)))
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_test,y_test_pred)))


# Increasing the number of Epochs Reduced Accuracy.

# In[ ]:




