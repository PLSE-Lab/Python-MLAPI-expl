#!/usr/bin/env python
# coding: utf-8

# # Keras quality predict

# # Import

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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# # Data preparation

# In[ ]:


X_data = pd.read_csv('/kaggle/input/production-quality/data_X.csv', sep=',')
X_data.head()


# In[ ]:


x_cols = list(X_data.columns)
x_cols[0] = 'date_time'
X_data.columns = x_cols


# In[ ]:


X_data.shape


# In[ ]:


Y_train = pd.read_csv('/kaggle/input/production-quality/data_Y.csv', sep=',')
Y_train.head()


# In[ ]:


Y_train.shape


# In[ ]:


Y_submit = pd.read_csv('/kaggle/input/production-quality/sample_submission.csv', sep=',')
Y_submit.head()


# In[ ]:


Y_submit.shape


# In[ ]:


train_df = X_data.merge(Y_train, left_on='date_time', right_on='date_time')
test_df = X_data.merge(Y_submit, left_on='date_time', right_on='date_time').drop('quality', axis=1)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


assert train_df.shape[0] == Y_train.shape[0]
assert test_df.shape[0] == Y_submit.shape[0]


# # EDA

# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


train_df.hist(figsize=(35, 30));


# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(train_df['quality'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['quality'])

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Quality distribution');


# In[ ]:


sns.pairplot(data=train_df);


# # Get target

# In[ ]:


y = train_df['quality']
train_df.drop(['quality'], axis=1, inplace=True)


# In[ ]:


train_df.drop(['date_time'], axis=1, inplace=True)
test_df.drop(['date_time'], axis=1, inplace=True)


# # Train test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.33)


# # Scale it

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
test_df = scaler.transform(test_df)


# In[ ]:


model = Sequential()
model.add(Dense(17, activation='tanh', input_shape=(17,)))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(34, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

model.summary()


# In[ ]:


model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['mse', 'mae'])


# In[ ]:


batch_size = 64
epochs = 15


# In[ ]:


history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))


# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test mse:', score[0])
print('Test mae:', score[2])


# In[ ]:


Y_submit['quality'] = model.predict(test_df)


# In[ ]:


Y_submit.head()


# In[ ]:


Y_submit.to_csv('submission.csv',index=False)

