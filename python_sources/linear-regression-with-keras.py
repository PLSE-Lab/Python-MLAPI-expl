#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam, SGD

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


df=pd.read_csv('../input/weight-height.csv')
X=df[['Height']].values
y_true=df[['Weight']].values


# In[4]:


X


# In[5]:


model = Sequential()
model.add(Dense(1, input_shape=(1,)))


# In[6]:


model.summary()


# In[7]:


model.compile(Adam(lr=0.8), 'mean_squared_error')


# In[10]:


model.fit(X,y_true, epochs=35, batch_size=110)


# In[11]:


y_pred= model.predict(X)


# In[12]:


df.plot(kind='scatter',
       x='Height',
       y='Weight', title='Weight and Height in adults')
plt.plot(X, y_pred, color='red', linewidth=3)


# In[13]:


w,b=model.get_weights()


# In[14]:


w


# In[15]:


b


# In[ ]:




