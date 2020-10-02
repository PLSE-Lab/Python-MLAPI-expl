#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True)


# In[ ]:


df.columns=['sales']


# In[ ]:


df.head(10)


# In[ ]:


df.index.freq='MS'


# In[ ]:


df.plot(figsize=(12,7))


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(df)


# In[ ]:


decompose.plot();


# In[ ]:


len(df)


# In[ ]:


325-12


# In[ ]:


train=df.iloc[:313]
test=df.iloc[313:]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(train)


# In[ ]:


s_train=scaler.transform(train)


# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator
generator=TimeseriesGenerator(s_train,s_train,length=12,batch_size=1)


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[ ]:


model=Sequential()
model.add(LSTM(200,activation='relu',input_shape=(12,1)))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')


# In[ ]:


model.fit_generator(generator,epochs=100)


# In[ ]:


loss=model.history.history['loss']
plt.plot(range(len(loss)),loss,color='green')


# In[ ]:


first_eval_batch=s_train[-12:]
first_eval_batch=first_eval_batch.reshape((1,12,1))
c_batch=first_eval_batch
test_pred=[]
future_pred=[]


# In[ ]:


for i in range(len(test)+12):    
    c_pred=model.predict(c_batch)[0]
    test_pred.append(c_pred)
    c_batch=np.append(c_batch[:,1:,:],[[c_pred]],axis=1)


# In[ ]:


len(test_pred)


# In[ ]:


for i in test_pred[12:]:
    future_pred.append(i)


# In[ ]:


test_pred=test_pred[:12]


# In[ ]:


test_pred


# In[ ]:


len(future_pred)


# In[ ]:


test_fpred=scaler.inverse_transform(test_pred)
future_fpred=scaler.inverse_transform(future_pred)


# In[ ]:


test['test_prediction']=test_fpred


# In[ ]:


test


# In[ ]:


test.plot(figsize=(12,7),label='test',legend=True)


# In[ ]:


date=pd.date_range('2019-02-01',periods=12,freq='MS')


# In[ ]:


prediction=pd.DataFrame(data=list(zip(date,future_fpred)),columns=['date','prediction sale'])


# In[ ]:


prediction.head()


# In[ ]:


prediction=prediction.set_index('date')
prediction['pred_sale']=future_fpred


# In[ ]:


prediction


# In[ ]:


prediction.index.freq='MS'


# In[ ]:


prediction.drop('prediction sale',axis=1)


# In[ ]:


df.plot(figsize=(15,10),label='test',legend=True)
prediction['pred_sale'].plot(legend=True,label='predictions')


# In[ ]:




