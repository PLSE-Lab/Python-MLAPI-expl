#!/usr/bin/env python
# coding: utf-8

# * FAT: Forget about time
# * Delta: Predict (x\* - last MidPrice) rather than x\*

# In[ ]:


import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler


# In[ ]:


INPUT_DIR = '../input/'


# In[ ]:


nRows = 430039
nGiven = 10
nPredict = 15
nFeature = 7


# In[ ]:


filename = os.path.join(INPUT_DIR, 'train_data.csv')
train_csv = pd.read_csv(filename)
timestamp = []
for i in range(nRows):
    dt = time.strptime(train_csv['Date'][i] + ' ' + train_csv['Time'][i], "%Y-%m-%d %H:%M:%S")
    dt_new = time.mktime(dt)
    timestamp.append(dt_new)


# In[ ]:


index = ['MidPrice','LastPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1']
train_ds = train_csv[index]


# In[ ]:


mid_price = np.array(train_csv['MidPrice']).astype(np.float64)


# In[ ]:


x_raw = np.array(train_ds.values).astype(np.float64)


# In[ ]:


X = []
y = []

fake_point = 0
volume_sum = 0
volume_len = 0

noise_flag = True

for k in range(nRows - nGiven - nPredict):
    if k%10000==0:
        print(k,end = '\r')
    x_cur = x_raw[k:k+nGiven].copy()
    last_mp = x_cur[nGiven-1,0]
    for axis in [0,1,3,5]: # MidPrice, LastPrice, BidPrice1, AskPrice1
        x_cur[:,axis] -= last_mp
        x_cur[:,axis] /= last_mp

    for i in range(9,0,-1):
        x_cur[i,2]-=x_cur[i-1,2]
        volume_sum+=x_cur[i,2]
        volume_len+=1
    x_cur[0,2]=volume_sum/volume_len
    
    if noise_flag:
        x_cur*=(1 + 0.001*(np.random.rand(10,nFeature) - 0.5)*2)
    
    if timestamp[k+nGiven+nPredict] - timestamp[k]> (3*(nGiven+nPredict)): 
        fake_point+=1
    else:
        token = True
        for i in range(nGiven+nPredict-1):
            if timestamp[k+1+i] - timestamp[k+i] != 3:
                token = False
                break
        if token:
            X.append(x_cur)
            y.append((sum(mid_price[k+nGiven:k+nGiven+nPredict])/nPredict-
                         mid_price[k+nGiven-1])/mid_price[k+nGiven-1])
        else:
            fake_point+=1
print()
print(fake_point)


# In[ ]:


X = np.array(X).astype(np.float64)
y = np.array(y).astype(np.float64)
y = y.reshape((-1,1))


# In[ ]:


X_tmp = X.reshape(-1,nFeature)


# In[ ]:


x_scaler = StandardScaler().fit(X_tmp)
X_tmp_norm = x_scaler.transform(X_tmp)


# In[ ]:


X_norm = X_tmp_norm.reshape(-1,nGiven,nFeature)


# In[ ]:


if not noise_flag:
    assert(X_norm[0][1][1]==X_norm[0][2][1])


# In[ ]:


y_std = np.std(y)
y_mean = np.mean(y)


# In[ ]:


y_norm = (y-y_mean)/y_std


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X_norm, y_norm, test_size=0.05, random_state=42)
print(X_train.shape)
print(y_train.shape)


# In[ ]:


model = Sequential()
model.add(LSTM(input_shape=(None, nFeature),activation='softsign',dropout=0.5, units=256, return_sequences=True))
model.add(LSTM(units=256,activation='softsign',dropout=0.5, return_sequences=False))
model.add(Dense(64,kernel_initializer="glorot_normal",activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
model.compile(loss='mean_squared_error', optimizer='Adam')


# In[ ]:


batch_size = 128
epochs = 12


# In[ ]:


hists = []
hist = model.fit(X_train, y_train, 
                 epochs = epochs,
                 batch_size = batch_size,
                 validation_data=(X_dev,y_dev))
hists.append(hist)


# In[ ]:


hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
hist_df.index = np.arange(1, len(hist_df)+1)
fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(16, 10))
axs.plot(hist_df['val_loss'], lw=5, label='Validation MSELoss')
axs.plot(hist_df['loss'], lw=5, label='Training MSELoss')
axs.set_ylabel('MSELoss')
axs.set_xlabel('Epoch')
axs.grid()
axs.legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show();


# In[ ]:


filename = os.path.join(INPUT_DIR, 'test_data.csv')
test_csv = pd.read_csv(filename)


# In[ ]:


nRows_test = 10000


# In[ ]:


index = ['MidPrice','LastPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1']
test_ds = test_csv[index]
mid_price_test = np.array(test_csv['MidPrice']).astype(np.float64)
x_test_raw = np.array(test_ds.values).astype(np.float64)


# In[ ]:


X_test = []
mid_price_batch_test = []
volume_sum_test = 0
volume_len_test = 0
for k in range(int(nRows_test/nGiven)):
    x_cur = x_test_raw[k*nGiven:k*nGiven+nGiven].copy()
    last_mp = x_cur[nGiven-1,0]
    for axis in [0,1,3,5]: # MidPrice, LastPrice, BidPrice1, AskPrice1
        x_cur[:,axis] -= last_mp
        x_cur[:,axis] /= last_mp

    for i in range(9,0,-1):
        x_cur[i,2]-=x_cur[i-1,2]
        volume_sum_test+=x_cur[i,2]
        volume_len_test+=1
    x_cur[0,2]=volume_sum_test/volume_len_test
    
    X_test.append(x_cur)
    mid_price_batch_test.append(mid_price_test[k*10+nGiven-1])


# In[ ]:


X_test = np.array(X_test).astype(np.float64)
X_test = X_test.reshape(-1,nGiven,nFeature)
mid_price_batch_test = np.array(mid_price_batch_test).astype(np.float64).reshape(-1,1)


# In[ ]:


X_tmp_test = X_test.reshape(-1,nFeature)
X_tmp_norm_test = x_scaler.transform(X_tmp_test)
X_norm_test = X_tmp_norm_test.reshape(-1,nGiven,nFeature)


# In[ ]:


y_test_pred = model.predict(X_norm_test)


# In[ ]:


y_test_pred = y_test_pred.reshape(-1,1)


# In[ ]:


result = y_test_pred * mid_price_batch_test * y_std + y_mean + mid_price_batch_test


# In[ ]:


offset = 142


# In[ ]:


result_offset = result[offset:int(nRows_test/10)]


# In[ ]:


submission = pd.DataFrame({'caseid':list(range(offset+1,1001)),'midprice':result_offset.reshape(-1,)})


# In[ ]:


submission.to_csv('submission', index=False)
submission.head()

