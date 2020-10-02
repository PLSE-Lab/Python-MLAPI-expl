#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


os.listdir('../input/')


# In[ ]:


train_data = pd.read_csv('../input/train.csv',dtype = {'acoustic_data':np.int16,'time_to_failure':np.float32})


# In[ ]:



rows = 150000
segments = int(np.floor(train_data.shape[0] / rows))

X_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat'])
y_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['time_to_failure'])


# In[ ]:


for segment in tqdm(range(segments)):
    x = train_data.iloc[segment*rows:segment*rows+rows]
    y = x['time_to_failure'].values[-1]
    x = x['acoustic_data'].values
    X_train.loc[segment,'mean'] = np.mean(x)
    X_train.loc[segment,'std']  = np.std(x)
    X_train.loc[segment,'99quat'] = np.quantile(x,0.99)
    X_train.loc[segment,'50quat'] = np.quantile(x,0.5)
    X_train.loc[segment,'25quat'] = np.quantile(x,0.25)
    X_train.loc[segment,'1quat'] =  np.quantile(x,0.01)
    y_train.loc[segment,'time_to_failure'] = y
    


# In[ ]:


scaler = StandardScaler()
X_scaler = scaler.fit_transform(X_train)


# In[ ]:


gc.collect()


# In[ ]:


model = Sequential()
model.add(Dense(32,input_shape = (6,),activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(1))
model.compile(loss = 'mae',optimizer = 'adam')


# In[ ]:


model.fit(X_scaler,y_train.values.flatten(),epochs = 50)


# In[ ]:


sub_data = pd.read_csv('../input/sample_submission.csv',index_col = 'seg_id')


# In[ ]:


X_test = pd.DataFrame(columns = X_train.columns,dtype = np.float32,index = sub_data.index)


# In[ ]:


for seq in tqdm(X_test.index):
    test_data = pd.read_csv('../input/test/'+seq+'.csv')
    x = test_data['acoustic_data'].values
    X_test.loc[seq,'mean'] = np.mean(x)
    X_test.loc[seq,'std']  = np.std(x)
    X_test.loc[seq,'99quat'] = np.quantile(x,0.99)
    X_test.loc[seq,'50quat'] = np.quantile(x,0.5)
    X_test.loc[seq,'25quat'] = np.quantile(x,0.25)
    X_test.loc[seq,'1quat'] =  np.quantile(x,0.01)
    


# In[ ]:


X_test_scaler = scaler.transform(X_test)


# In[ ]:



pred = model.predict(X_test_scaler)


# In[ ]:


sub_data.head()


# In[ ]:


sub_data['time_to_failure'] = pred
sub_data['seg_id'] = sub_data.index


# In[ ]:


sub_data.head()


# In[ ]:


sub_data.to_csv('sub_earthquake.csv',index = False)


# In[ ]:




