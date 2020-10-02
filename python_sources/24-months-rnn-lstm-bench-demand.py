#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sys 
import numpy as np # linear algebra
#from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
#from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
#from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
#from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
import keras.backend as K
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model


# In[ ]:


def adj_r2_score(r2,n,k):
	return 1 - ((1-r2)*(n-1)/(n-k-1))


# In[ ]:


df = pd.read_csv('../input/demand-data-update/Kaggle_Dataset_Update.csv', sep=',',index_col='Month')
df.index = pd.to_datetime(df.index, format = '%m')
df.head()


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


split_ratio = 0.6
split_point = int(round(len(df)* split_ratio))
train,test = df.iloc[:split_point,:],df.iloc[split_point:,:]
split_ratio = 0.6
split_point = int(round(len(df)* split_ratio))
train,test = df.iloc[:split_point,:],df.iloc[split_point:,:]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# In[ ]:


test_scaled.shape


# In[ ]:


X_tr_sc


# In[ ]:


X_tr_sc = train_scaled[:-1]
y_tr_sc = train_scaled[1:]
X_tst_sc = test_scaled[:-1]
y_tst_sc = test_scaled[1:]


# In[ ]:


X_tr_sc = X_tr_sc.reshape(X_tr_sc.shape[0],1,X_tr_sc.shape[1])
X_tst_sc = X_tst_sc.reshape(X_tst_sc.shape[0],1,X_tst_sc.shape[1])


# In[ ]:


X_tst_sc


# In[ ]:


K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(7, input_shape=(1, X_tr_sc.shape[1]), activation = 'relu', kernel_initializer = 'lecun_uniform', return_sequences = False))
model_lstm.add(Dense(1))
model_lstm.compile(loss = 'mean_squared_error', optimizer = 'adamax')
early_stop = EarlyStopping(monitor = 'loss', patience =5, verbose = 1)
history_model_lstm = model_lstm.fit(X_tr_sc,y_tr_sc, epochs = 1000, batch_size = 1, verbose = 1, shuffle = False, callbacks = [early_stop])


# In[ ]:


test_pred_sc = model_lstm.predict(X_tst_sc)
train_pred_sc = model_lstm.predict(X_tr_sc)


# In[ ]:


print("R2 score on the train set:\t{}".format(r2_score(y_tr_sc,train_pred_sc)))
r2_train = r2_score(y_tr_sc,train_pred_sc)


# In[ ]:


print("Adjusted R2 score on the Train set:\t {}".format(adj_r2_score(r2_train,X_tr_sc.shape[0],X_tr_sc.shape[1])))


# In[ ]:


print("R2 score on the Test set:\t{:0.3f}". format(r2_score(y_tst_sc,test_pred_sc)))
r2_test = r2_score(y_tst_sc, test_pred_sc)


# In[ ]:


print("Adjusted R2 score on the Test set:\t {}".format(adj_r2_score(r2_test,X_tst_sc.shape[0],X_tst_sc.shape[1])))


# In[ ]:


model_lstm.save('LSTM_Nonshift.h5')
score_lstm = model_lstm.evaluate(X_tst_sc, y_tst_sc, batch_size = 1)


# In[ ]:


print("LSTM: %F"%score_lstm)


# In[ ]:


test_pred_sc = model_lstm.predict(X_tst_sc)


# In[ ]:


#plt.plot(y_tst_sc,label = "Actual")
#plt.plot(test_pred_sc, label = "Predicted")
#plt.title("LSTM's Prediction (scaled)")
#plt.xlabel("Observation")
#plt.ylabel("Scaled values")
#plt.legend()
#plt.show()

test_pred = scaler.inverse_transform(test_pred_sc)
y_test = test[1:]

test_pred = test_pred.ravel()

test_pred = pd.Series(test_pred)

test_pred.index = test.index[1:]

plt.figure(figsize =(12,4))
plt.plot(df,'b')
plt.plot(test_pred,'r')
plt.show()

#plt.title("RMSE:{}",np.sqrt(sum((test_pred.values - y_test['count'].values)**2)/len(y_test)))


# In[ ]:


print(y_test)
print(test_pred)


# In[ ]:


moving_avg = df.rolling(2).mean()
plt.plot(df)
plt.plot(moving_avg, color='red')




# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = df.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


df_moving_avg_diff = df - moving_avg
df_moving_avg_diff.head(12)

df_moving_avg_diff.dropna(inplace=True)
#test_stationarity(df_moving_avg_diff)


# In[ ]:


#expwighted_avg = pd.ewm(df, span=2,adjusted = False)
#plt.plot(df)
#plt.plot(expwighted_avg, color='red')


# In[ ]:


df_diff = df - df.shift()
plt.plot(df_diff,color = 'green')
plt.plot(df,color = 'blue')
plt.plot(moving_avg, color='red')


# In[ ]:


df['Total_Demand'] = df['Total_Demand'].astype(float)


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(df_diff)
plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: {}'% sum((results_AR.fittedvalues-df_diff)**2))


# In[ ]:


model = ARIMA(df, order = (1,1,1))  
results_MA = model.fit(disp=-1)  
plt.plot(df_diff)
plt.plot(results_MA.fittedvalues, color='red')
#plt.title('RSS: {}'% sum((results_MA.fittedvalues-df_diff)**2))


# In[ ]:


model = ARIMA(df, order=(1, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(df_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


# In[ ]:





# In[ ]:





# In[ ]:




