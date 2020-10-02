#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import pandas as pd
data = pd.read_csv("../input/leaf-classification/train.csv")


# In[ ]:


data.head()


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[ ]:



from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical


# In[ ]:



from pylab import rcParams
rcParams['figure.figsize'] = 10,10


# In[ ]:



parent_data = data.copy()    
ID = data.pop('id')


# In[ ]:


data.shape


# In[ ]:


y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
print(y.shape)


# In[ ]:


X = StandardScaler().fit(data).transform(data)
print(X.shape)


# In[ ]:



y_cat = to_categorical(y)
print(y_cat.shape)


# In[ ]:



model = Sequential()
model.add(Dense(1024,input_dim=192,  init='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(99, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ["accuracy"])


# In[ ]:


history = model.fit(X,y_cat,batch_size=32,
                    nb_epoch=400,verbose=0)


# In[ ]:


test = pd.read_csv("../input/leaf-classification/test.csv")


# In[ ]:


test.head()


# In[ ]:


index = test.pop('id')


# In[ ]:


test = StandardScaler().fit(test).transform(test)


# In[ ]:


yPred = model.predict_proba(test)


# In[ ]:


# Get the learned weights
dense_layers = [l for l in model.layers if l.name.startswith('dense')]
kernels, biases = zip(*[l.get_weights() for l in dense_layers])


# In[ ]:


print([k.shape for k in kernels])
print([b.shape for b in biases])


# In[ ]:


# Visualize the digits
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,5))
x, y = 5, 2
for digit in range(10):
    triggers = kernels[0].dot(kernels[1])[:, digit]
    triggers = triggers.reshape(16, 12) / np.absolute(triggers).max() * 255    # Make the base image black
    pixels = np.full((16, 12, 3), 0, dtype=np.uint8)
    # Color positive values green
    green = np.clip(triggers, a_min=0, a_max=None)
    pixels[:, :, 1] += green.astype(np.uint8)
    # Color negative values red
    red = -np.clip(triggers, a_min=None, a_max=0)
    pixels[:, :, 0] += red.astype(np.uint8)

    plt.subplot(y, x, digit+1)
    plt.imshow(pixels)
plt.show()


# In[ ]:


# Visualize the first 20 neurons in tsecond hidden layer
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,10))
x, y = 5, 4
for neuron in range(20):
    triggers = kernels[0].dot(kernels[1])[:, neuron]
    triggers = triggers.reshape(16, 12) / np.absolute(triggers).max() * 255
    # Make the base image black
    pixels = np.full((16, 12, 3), 0, dtype=np.uint8)
    # Color positive values green
    green = np.clip(triggers, a_min=0, a_max=None)
    pixels[:, :, 1] += green.astype(np.uint8)
    # Color negative values red
    red = -np.clip(triggers, a_min=None, a_max=0)
    pixels[:, :, 0] += red.astype(np.uint8)

    plt.subplot(y, x, neuron+1)
    plt.imshow(pixels)
plt.show()


# In[ ]:




