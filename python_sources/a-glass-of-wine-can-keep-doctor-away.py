#!/usr/bin/env python
# coding: utf-8

# We will try to explore the dataset.We will be using deep learing to classify wine into different categories.We will be using Keras for out work.This kernel is a work in process.If you like my work please do vote.

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


# ### Importing the Python Modules

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# ### Importing the data

# In[ ]:


df = pd.read_csv('../input/principal-component-analysis/Wine.csv')
df.head()


# ### Creating Matrix of Target Values 

# In[ ]:


y = df['Customer_Segment']


# Customer Segement represents the class of the wine.This is out target variable which we will be predicting using Deep learning algorithm.

# In[ ]:


y.value_counts()


# We can see that the wine is categorised into three class based on the customer choice.

# In[ ]:


y_cat = pd.get_dummies(y)


# In[ ]:


y_cat.head()


# So we have converted our target y into three columns of data which will be input to out Deep Learning Model.

# ### Creating Matrix Of Features

# In[ ]:


X = df.drop('Customer_Segment',axis=1)


# In[ ]:


X.shape


# ### Pair Plot

# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(df,hue = 'Customer_Segment');


# In most of the plots we can see a clear separation between the three classes.Our Deep learning model would in in poistion to give good level of accuracy whicle predicting the class of the wine.

# ### Scaling the Features

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xsc = sc.fit_transform(X)


# ### Building the Neural Network

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K


# In[ ]:


K.clear_session()
model = Sequential()
model.add(Dense(5,input_shape=(13,),kernel_initializer = 'he_normal',activation = 'relu'))
model.add(Dense(3,activation = 'softmax'))

model.compile(RMSprop(lr=0.1),'categorical_crossentropy',metrics = ['accuracy'])

model.fit(Xsc,y_cat.values,batch_size=8,epochs=10,verbose=1,validation_split=0.2)


# ### Defining feature fuction to check data Seperation

# In[ ]:


K.clear_session()
model = Sequential()
model.add(Dense(8,input_shape=(13,),kernel_initializer = 'he_normal',activation = 'tanh'))
model.add(Dense(5,kernel_initializer = 'he_normal',activation = 'tanh'))
model.add(Dense(2,kernel_initializer = 'he_normal',activation = 'tanh'))
model.add(Dense(3,activation = 'softmax'))


model.compile(RMSprop(lr=0.05),'categorical_crossentropy',metrics = ['accuracy'])

model.fit(Xsc,y_cat.values,batch_size=16,epochs=20,verbose=1)


# In[ ]:


model.summary()


# We are going to extract the values at dens_3 and plot it on a 2D graph to show the seperation.

# In[ ]:


inp = model.layers[0].input
out = model.layers[2].output


# In[ ]:


features_function = K.function([inp],[out])


# In[ ]:


features = features_function([Xsc])[0]


# In[ ]:


plt.scatter(features[:,0],features[:, 1])


# We can see there there is a clear seperation of the three categories.Neural Networks are very good feature Learners.
