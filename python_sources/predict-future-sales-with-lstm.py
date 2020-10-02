#!/usr/bin/env python
# coding: utf-8

# ### Data Overview

# reference: https://www.kaggle.com/kcbighuge/predicting-sales-with-a-nested-lstm

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

default_path = '../input/'


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


train_df = pd.read_csv(default_path+'sales_train.csv')
items_df = pd.read_csv(default_path+'items.csv')
test_df = pd.read_csv(default_path+'test.csv')


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


train_df.head()


# ### Outliers

# In[ ]:


sns.boxplot(y = 'item_cnt_day', data = train_df)


# ### Duplicates

# In[ ]:


train_df = train_df.drop_duplicates()
train_df[train_df.duplicated()]


# In[ ]:


train_df = train_df[train_df['item_cnt_day'] < 1100]
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')


# In[ ]:


train_df.info()


# In[ ]:


train_df.tail()


# ### Reshape the training set as time series

# In[ ]:


# Impute all the new item_id in test set as 0
dataset = train_df.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0,aggfunc=np.sum)
dataset = dataset.reset_index()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.tail()


# In[ ]:


test_df.head()


# ### Prepare the test set

# In[ ]:


dataset = pd.merge(test_df, dataset, on=['item_id', 'shop_id'], how='left')
dataset = dataset.fillna(0)
dataset.head()


# In[ ]:


dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)
dataset.head()


# ### Define the LSTM input

# In[ ]:


X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]

X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)


# In[ ]:


dataset.values.shape


# In[ ]:


dataset.values[:,:-1].shape


# ### LSTM Model

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[ ]:


model = Sequential()
model.add(LSTM(units=64, input_shape=(33,1)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])

model.summary()


# In[ ]:


from keras.callbacks import EarlyStopping

callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]

history = model.fit(X_train, y_train, batch_size=4096, epochs=25,callbacks=callbacks_list)


# In[ ]:


#plt.plot(history.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(history.history['mean_squared_error']), label= 'training RMSE')
plt.legend(loc=1)


# In[ ]:


pred = model.predict(X_train)

pred = pred.clip(0, 20)


# In[ ]:


print(y_train.shape)
print(pred.shape)


# In[ ]:


from sklearn.metrics import mean_squared_error
import math

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train, pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))


# In[ ]:


LSTM_prediction = model.predict(X_test)


# In[ ]:


LSTM_prediction = np.round(LSTM_prediction,2)


# In[ ]:


LSTM_prediction = LSTM_prediction.clip(0, 20)


# In[ ]:


submission = pd.DataFrame({'ID': test_df['ID'], 'item_cnt_month': LSTM_prediction.ravel()})
submission.to_csv('submission.csv',index=False)


# In[ ]:




