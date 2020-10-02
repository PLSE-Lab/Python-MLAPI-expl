#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting with LSTM-generator
# 
# On this notebook, we will try to predict the alcohol sales for unknown future for 1 month using LSTM-generator. 
# You can find more informations about data in that site:
# * https://fred.stlouisfed.org/series/S4248SM144NCEN
# 
# 
# Let's import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/S4248SM144NCEN.csv', index_col= 'DATE', parse_dates=True)
df.index.freq = 'MS'


# In[ ]:


df.info()


# In[ ]:


df.columns= ['Sales']


# In[ ]:


df.head()


# In[ ]:


df.plot(figsize=(16,8))


# As we can see, we have 17 years sale data. And this data is acting like time series data.

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


result= seasonal_decompose(df['Sales'])


# In[ ]:


result.plot();


# In[ ]:


len(df)


# In[ ]:


train = df.iloc[:316]
test= df.iloc[316:]


# In[ ]:


test= test[0:12]


# In[ ]:


test.info()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


scaler.fit(train)


# In[ ]:


scaled_train = scaler.transform(train)


# In[ ]:


scaled_test = scaler.transform(test) 


# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


len(scaled_train)


# Now, we will start to create LSTM model for forecasting.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[ ]:


n_input = 12
n_feature = 1

train_generator = TimeseriesGenerator(scaled_train,scaled_train,length=n_input, batch_size=1)


# In[ ]:


model = Sequential()

model.add(LSTM(128,activation = 'relu', input_shape= (n_input, n_feature), return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.summary()


# In[ ]:


model.fit_generator(train_generator,epochs= 50)


# In[ ]:


my_loss= model.history.history['loss']
plt.plot(range(len(my_loss)),my_loss)


# In[ ]:


first_eval_batch = scaled_train[-12:]


# In[ ]:


first_eval_batch


# In[ ]:


first_eval_batch = first_eval_batch.reshape((1,n_input,n_feature))


# In[ ]:


model.predict(first_eval_batch)


# # Forecast Using RNN Model

# In[ ]:


#holding my predictions
test_predictions = []


# last n_input points from the training set
first_eval_batch = scaled_train[-n_input:]
# reshape this to the format RNN wants (same format as TimeseriesGeneration)
current_batch = first_eval_batch.reshape((1,n_input,n_feature))

#how far into the future will I forecast?

for i in range(len(test)):
    
    # One timestep ahead of historical 12 points
    current_pred = model.predict(current_batch)[0]
    
    #store that prediction
    test_predictions.append(current_pred)
    
    # UPDATE current batch o include prediction
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis= 1)


# In[ ]:


test_predictions


# In[ ]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[ ]:


true_predictions


# In[ ]:


test['Predictions'] =true_predictions


# In[ ]:


test.head()


# In[ ]:


test.plot(figsize=(12,8))


# In[ ]:


model.save('mycoolmodel.h5')


# Our model is not bad!
# 
# Thanks:) If you like it please vote. 
