#!/usr/bin/env python
# coding: utf-8

# ### My first Kaggle work
# ## Exploring the files

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Importing the data

# Importing:

# In[ ]:


import pandas as pd
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", header=0, delimiter=',')
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", header=0, delimiter=',')


# Watch the data:

# In[ ]:


df.head()


# In[ ]:


test.head()


# Transform object into categorical:

# In[ ]:


df = df.fillna(0)
test = test.fillna(0)
df = pd.get_dummies(df)
test = pd.get_dummies(test)
df.head()


# Create training and validation:

# In[ ]:


trainingSize = int(df.shape[0]*3/4)

x_train = df.loc[0:trainingSize, df.columns != 'SalePrice']
y_train = df.loc[0:trainingSize, df.columns == 'SalePrice']

x_val = df.loc[trainingSize:, df.columns != 'SalePrice']
y_val = df.loc[trainingSize:, df.columns == 'SalePrice']


# ### Creating a model:
# 
# Here we just need to find a coeeficient to each column, and the result should be somewhat satisfactory.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from numpy.random import seed
seed(1)

model = Sequential(name='my_model')
model.add(Dense(x_train.shape[1], activation='relu',input_dim=x_train.shape[1],name="dense_input"))
model.add(Dense(1, activation='relu',name="dense_output",use_bias=True))

model.compile(loss='mean_squared_error', optimizer="adam")

model.summary()


# Training:

# In[ ]:


import matplotlib.pyplot as plt

history = model.fit(x_train,y_train,epochs=25)

plt.plot(history.history['loss'], label='loss')
plt.suptitle('Model training', fontsize=20)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=16)
plt.show()


# See the result with our validation datas:

# In[ ]:


import matplotlib.pyplot as plt

y_pred = model.predict(x_val.values)

plt.suptitle('Scatter of predicitons on validation data', fontsize=20)
plt.xlabel('True value', fontsize=18)
plt.ylabel('Predicted value', fontsize=16)
plt.scatter(y_val, y_pred)
plt.show()


# In[ ]:


import numpy as np

difference = y_val-y_pred

plt.suptitle('Difference between predicitons and validation data', fontsize=20)
plt.ylabel('Number', fontsize=16)
plt.xlabel('Difference', fontsize=18)
counts, bins = np.histogram(difference, bins = 100)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

print("Mean: " + str(difference["SalePrice"].mean()) + "; Standard deviation: " + str(difference["SalePrice"].std()))

