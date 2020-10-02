#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import keras


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm


# In[ ]:


train_df = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv",parse_dates=['timestamp'],index_col='timestamp')


# In[ ]:


train_df.head(10)


# In[ ]:


plt.figure(figsize=(9,9))
plt.plot(train_df.index.year,train_df['cnt'])


# In[ ]:


plt.figure(figsize=(9,9))
sns.lineplot(x=train_df.index,y='cnt',data=train_df)


# In[ ]:


df_by_month = train_df.resample("M").sum()

plt.figure(figsize=(9,9))
sns.lineplot(x=df_by_month.index,y='cnt',data=df_by_month)


# In[ ]:


plt.figure(figsize=(14,8))
sns.pointplot(x=train_df.index.hour,y='cnt',data=train_df,hue='is_holiday')


# In[ ]:


plt.figure(figsize=(11,5))
sns.pointplot(x=train_df.index.dayofweek,y='cnt',data=train_df)


# In[ ]:


train_len =  int(0.9*len(train_df))
test_len = len(train_df) - train_len

train,test = train_df.iloc[:train_len],train_df.iloc[train_len:len(train_df)]
print(train_df.shape,train.shape,test.shape)


# In[ ]:


#train_trans = train[['t1','t2','hum','wind_speed']].to_numpy()
#test_trans = test[['t1','t2','hum','wind_speed']].to_numpy()


# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


rs = RobustScaler()
rs_cnt = RobustScaler()

t_c = ['t1','t2','hum','wind_speed']

train.loc[:,t_c] = rs.fit_transform(train[t_c].to_numpy())
test.loc[:,t_c] = rs.transform(test[t_c].to_numpy())


# In[ ]:


train['cnt'] = rs_cnt.fit_transform(train[['cnt']])
test['cnt'] = rs_cnt.transform(test[['cnt']])


# In[ ]:


train.to_numpy()
test.to_numpy()


# In[ ]:


def create_dataset(x,y,time_steps=1):
    x_train,y_train = [],[]
    
    for i in range(len(x)-time_steps):
        v = x.iloc[i:(i+time_steps)].values
        x_train.append(v)
        y_train.append(y.iloc[i+time_steps])
        
    return np.array(x_train),np.array(y_train)


# In[ ]:


time_steps = 24

x_train,y_train = create_dataset(train,train.cnt,time_steps)
x_test,y_test = create_dataset(test,test.cnt,time_steps)

print(x_train.shape,y_train.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,LSTM ,Bidirectional,Dropout


# In[ ]:


model = Sequential()

model.add(Bidirectional(LSTM(128,
                            input_shape=(x_train.shape[1],x_train.shape[2]))))
model.add(Dropout(0.25))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')


# In[ ]:


history = model.fit(x_train,y_train,
                   epochs=30,
                   batch_size=32,
                   validation_split=0.1,
                   shuffle=False 
                   )


# In[ ]:


plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_test_inv = rs_cnt.inverse_transform(y_test.reshape(1,-1))
y_pred_inv = rs_cnt.inverse_transform(y_pred)


# In[ ]:


plt.figure(figsize=(12,12))
plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Bike Count')
plt.xlabel('Time Step')
plt.legend()
plt.show();


# In[ ]:




