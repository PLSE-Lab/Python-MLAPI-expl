#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Reading dataframe and changing one date colum to index and datetime dtype****

# In[ ]:


df=pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv',parse_dates=['Time Serie'],index_col=1)


# # Removing The unamed Column

# In[ ]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# # Checking some attributes and other function on dataframe

# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum()


# # Replacing The ND value in the Data Frame to Zero Tell me What Should I have Done With It?

# In[ ]:


df=df.replace('ND',0)


# In[ ]:


df


# # Changing Dtype to Floats so that working can be easy

# In[ ]:


for dtype in df.dtypes:
    df=df.astype('float')


# In[ ]:


df.dtypes


# # Litttle Bit Of Visualization 

# In[ ]:


fig1,(ax1,ax2)=plt.subplots(2,1,sharex=True)
df['AUSTRALIA - AUSTRALIAN DOLLAR/US$'].plot(ax=ax1)
df['EURO AREA - EURO/US$'].plot(ax=ax2)


ax1.set_title('AUSTRALIA - AUSTRALIAN DOLLAR/US$',loc='right')
ax2.set_title('EURO AREA - EURO/US$',loc='right')



fig2,(ax3,ax4)=plt.subplots(2,sharex=True,)
df['NEW ZEALAND - NEW ZELAND DOLLAR/US$'].plot(ax=ax3,)
df['UNITED KINGDOM - UNITED KINGDOM POUND/US$'].plot(ax=ax4)


ax3.set_title('NEW ZEALAND - NEW ZELAND DOLLAR/US$',loc='right')
ax4.set_title('UNITED KINGDOM - UNITED KINGDOM POUND/US$',loc='right')

plt.show()


# In[ ]:


len(df)


# In[ ]:


df.tail()


# # Subset the the dataframe to two year so training will take less time and rounding the values to 2 decimel points to reduce noise 

# In[ ]:


ml_df=df.loc['2018-01-01':]


# In[ ]:


ml_df=ml_df.round(2)


# In[ ]:


ml_df


# # As one observation represent one day so I split the data. Test data will get the 5 observation(row) you can set that test_ind to any integer you like for this type of data set as it represent one day with a row

# In[ ]:


test_ind=5


# # negative sign represent start from begining but dont include the test_ind values for train and for test it represent start from that index to the end

# In[ ]:


train=ml_df.iloc[:-test_ind]
test=ml_df.iloc[-test_ind:]


# In[ ]:


train


# In[ ]:


test


# # Scaling the dataset using MinMaxScaler and fitting only on train data so that test data cant get memorize 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


scaler.fit(train)


# In[ ]:


scaled_train=scaler.transform(train)
scaled_test=scaler.transform(test)


# In[ ]:


scaled_train.shape


# # Creating two generator validation and training generator the length value is the timestamp and you should set lenght value less then the test_perc/test_ind variable used above

# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


length=2
generator=TimeseriesGenerator(scaled_train,scaled_train,batch_size=1,length=length)


# In[ ]:


val_generator=TimeseriesGenerator(scaled_test,scaled_test,batch_size=1,length=length)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,LSTM


# # A small Recurrent Neural Network using LSTM Layer

# In[ ]:


n_features=scaled_train.shape[1]
model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(length,n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam',loss='mse')


# In[ ]:


model.summary()


# # Early Stopping for reducing overfitting as you cant remove overfitting completely on any dataset 

# In[ ]:


from keras.callbacks import EarlyStopping


# In[ ]:


early=EarlyStopping(monitor='val_loss',patience=3)


# In[ ]:


model.fit_generator(generator,epochs=10,callbacks=[early],validation_data=val_generator)


# In[ ]:


losses=pd.DataFrame(model.history.history)


# In[ ]:


losses.plot()


# In[ ]:


prediction=model.predict_generator(val_generator)


# In[ ]:


true_prediction=scaler.inverse_transform(prediction)


# In[ ]:


test_df=test[2:]


# In[ ]:


test_df


# In[ ]:


prediction_df=pd.DataFrame(true_prediction,columns=test.columns,index=test_df.index)


# In[ ]:


prediction_df


# In[ ]:


fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6),sharex=True)
ax1.plot(test_df)
ax2.plot(prediction_df)

ax1.set_title('True Value')
ax2.set_title('Predicted Value')


# # Checking the RMSE on Single Column you can check on the whole dataset

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


np.sqrt(mean_squared_error(test_df['NEW ZEALAND - NEW ZELAND DOLLAR/US$'],prediction_df['NEW ZEALAND - NEW ZELAND DOLLAR/US$']))


# # Thanks i hope you like it there is always room for improvements

# In[ ]:




