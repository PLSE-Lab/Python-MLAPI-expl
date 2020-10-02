#!/usr/bin/env python
# coding: utf-8

# In this dataset we have real estate data.We have features like number of bedrooms,size in sqaure foot and age of the house.Our task is to build a model which can predict the price of the house using the data.If you like the work please do vote.

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


# ### Importing Python modules

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# ### Importing the dataset

# In[ ]:


df = pd.read_csv('../input/housing-data.csv')
df.head()


# We have 47 rows of data and have values of area,rooms,age and the price of house.

# ### Plotting Histogram

# In[ ]:


plt.figure(figsize = (15,5))
for i, feature in enumerate(df.columns):
    plt.subplot(1,4,i+1)
    df[feature].plot(kind='hist',title =feature)
    plt.xlabel(feature)


# We have the distribution of all the four features in our dataset.Now we will be using the values of Area,Rooms and Age to predict he price of the house.

# ### Creating the Matrix of Features

# In[ ]:


X = df[['sqft','bdrms','age']].values
y = df[['price']].values


# ### Building up the Model 

# In[ ]:


X = df[['sqft','bdrms','age']].values
y = df['price'].values


# ### Building Linear Regression Model

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam


# In[ ]:


model = Sequential()
model.add(Dense(1,input_shape=(3,)))
model.compile(Adam(lr=0.8),'mean_squared_error')


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
len(X_train)


# In[ ]:


model.fit(X_train,y_train,epochs=10)


# In[ ]:


df['price'].min()


# In[ ]:


df['price'].max()


# We can see that the loss function is going down but it has very big value.Because the loss is calculated interm of difference of actual and predicted prices.If we look at the house price in the dataset the minimum and maximum values are 169900 and 699900 respectively.So when we calculate the loss function the value of the loss function is also bigger.One more problem here is we have only 47 data point.This is very less data for our neural network.

# In[ ]:


df.describe()


# From the describe option we can seee that there is a big difference in the value range of Area,Price and the Age of the House.There is a possibility of improving our results with feature scaling/normalization.Which we will try out and see.

# ### Scaling the Features

# In[ ]:


df['sqft1000'] = df['sqft']/1000
df['age10'] = df['age']/10
df['price100k'] = df['price']/1e5


# In[ ]:


X = df[['sqft1000','bdrms','age10']].values
y = df['price100k'].values


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


model = Sequential()
model.add(Dense(1,input_dim =3))
model.compile(Adam(lr=0.1),'mean_squared_error')
model.fit(X_train,y_train,epochs = 50)


# Now we can see that the the loss function has considerably reduced by scaling the features.This shows the affect of scaling on improving the accuaracy of a Machine Learning Model.

# ### R Square Accuracy of the model

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train,y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test,y_test_pred)))


# So we can see that our Acuracy score test set is lower than for train set. We can get better results if we have more data and if we can add more layers to our neural network.

# In[ ]:




