#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam, SGD

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[2]:


file=pd.read_csv('../input/weight-height.csv')
X=file[['Height']].values
Y=file[['Weight']].values


# In[3]:


X


# In[4]:


Y


# In[5]:


model = Sequential()
model.add(Dense(1, input_shape=(1,)))


# In[6]:


model.summary()


# In[7]:


model.compile(Adam(lr=0.8), 'mean_squared_error')


# In[9]:


model.fit(X,Y, epochs=40, batch_size=120)


# In[18]:


y_pred= model.predict(X)


# In[19]:


file.plot(kind='scatter',
       x='Height',
       y='Weight', title='Weight and Height in adults')
plt.plot(X, y_pred, color='red', linewidth=3)


# In[23]:


a,b=model.get_weights()


# In[24]:


a


# In[25]:


b


# In[ ]:




