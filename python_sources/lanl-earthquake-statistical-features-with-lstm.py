#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from statistics import mean

import os

import random
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm_pandas

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


step = 100000000
stop = 600000000
X = pd.DataFrame(dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat','time_to_failure'])
j = 0
for i in tqdm(range(0, stop, step)):
    train_df = pd.read_csv("../input/train.csv",
                           skiprows = i,
                           nrows = step,
                           dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}
                          )
    train_df.columns = ['acoustic_data','time_to_failure']
    seg_len = 5000
    segments = int(np.floor(train_df.shape[0] / seg_len))
    for segment in range(segments):
        x = train_df.acoustic_data[segment*seg_len:segment*seg_len+seg_len]
        X.loc[j,'mean'] = np.mean(x)
        X.loc[j,'std']  = np.std(x)
        X.loc[j,'99quat'] = np.quantile(x,0.99)
        X.loc[j,'50quat'] = np.quantile(x,0.5)
        X.loc[j,'25quat'] = np.quantile(x,0.25)
        X.loc[j,'1quat'] =  np.quantile(x,0.01)
        X.loc[j,'time_to_failure'] = train_df.time_to_failure.values[segment*seg_len+seg_len-1]
        j +=1
    del train_df
    gc.collect()
    


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
X.iloc[:,:-1] = scaler.fit_transform(X.iloc[:,:-1])


# In[ ]:


def getTrainBatch(dfl,seg_len,batch_size):
    x = np.empty([batch_size,seg_len,6])
    y = np.empty([batch_size,1])
    for i,rn in enumerate(np.random.randint(dfl.shape[0]-seg_len, size=batch_size)):
        df = dfl.loc[rn:rn+seg_len-1,:]
        x[i,:,:] = df.iloc[:,:-1]
        y[i] = df.iloc[-1,-1]
    return x,y


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.callbacks import History


# In[ ]:


model = Sequential()

model.add(CuDNNLSTM(64 ,return_sequences=True ,input_shape=(30, 6)))
model.add(CuDNNLSTM(64,return_sequences=True))
model.add(CuDNNLSTM(64))
model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='linear'))

print(model.summary())


# In[ ]:


model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mae'])


# In[ ]:


loss = []
val_loss = []
for j in tqdm(range(501)):
        #print('Generating training batch :',j)
        x_train,y_train = getTrainBatch(X,30,batch_size=1024)
        history = model.fit(x_train,
                            y_train,
                            batch_size=16,
                            epochs=10,
                            validation_split=0.1,
                            verbose=0)
        loss = loss + history.history['loss']
        val_loss = val_loss + history.history['val_loss']

        if (j%10==0):
            print('loss :',mean(loss[-10:]),' val_loss :',mean(val_loss[-10:]))
        del x_train, y_train
        gc.collect()


# In[ ]:


# predicting the submission
def predictSubmission(seg_id):
    X_test = pd.DataFrame(dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat'])
    test_df = pd.read_csv('../input/test/' + seg_id + '.csv')
    seg_len = 5000
    segments = int(np.floor(test_df.shape[0] / seg_len))
    for i,segment in enumerate(range(segments)):
        x = test_df.acoustic_data[segment*seg_len:segment*seg_len+seg_len]
        X_test.loc[i,'mean'] = np.mean(x)
        X_test.loc[i,'std']  = np.std(x)
        X_test.loc[i,'99quat'] = np.quantile(x,0.99)
        X_test.loc[i,'50quat'] = np.quantile(x,0.5)
        X_test.loc[i,'25quat'] = np.quantile(x,0.25)
        X_test.loc[i,'1quat'] =  np.quantile(x,0.01)
    y = model.predict(scaler.transform(X_test).reshape(1,30,6))
    return y[0][0]


# In[ ]:


tqdm_pandas(tqdm())
submission = pd.read_csv('../input/sample_submission.csv')
submission.loc[:,'time_to_failure']=submission.loc[:,'seg_id'].progress_apply(predictSubmission)
submission.to_csv('submission_11.csv',index=False)


# In[ ]:


submission.head()


# In[ ]:




