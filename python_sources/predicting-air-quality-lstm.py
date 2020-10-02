#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# #### Database loading
# #### Each record represents the level of pollution in each hour (hourly measure)
# #### The idea is to predict the level of pollution at a certain time based on the previous hours

# In[ ]:


base = pd.read_csv('../input/PRSA_data_2010.1.1-2014.12.31.csv')
base.head()


# In[ ]:


# Deletion of records with unfilled values (missing values)
base = base.dropna()


# #### Exclusion of attributes that will not be used for analysis
# #### "No" is the registration number
# #### "Year", "month", "day" and "hour" are the time information that is not used for analysis
# #### "Cbwd" is a text attribute that is not related to predictions

# In[ ]:


base = base.drop('No', axis = 1)
base = base.drop('year', axis = 1)
base = base.drop('month', axis = 1)
base = base.drop('day', axis = 1)
base = base.drop('hour', axis = 1)
base = base.drop('cbwd', axis = 1)


# In[ ]:


# Predictive attributes are all but not index 0
base_treinamento = base.iloc[:, 1:7].values


# In[ ]:


base.head()


# ### Search of the values to be made the forecast (target), ie the first attribute pm2.5

# In[ ]:


poluicao = base.iloc[:, 0].values


# In[ ]:


# Application of normalization
normalizador = MinMaxScaler(feature_range = (0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)


# In[ ]:


# Need to change the format of the variable to apply normalization
poluicao = poluicao.reshape(-1, 1)
poluicao_normalizado = normalizador.fit_transform(poluicao)


# In[ ]:


#### Creation of the data structure that represents the time series, considering
#### 10 hours (window) earlier to predict the current time


# In[ ]:


previsores = []
poluicao_real = []
for i in range(10, 41757):
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6])
    poluicao_real.append(poluicao_normalizado[i, 0])
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)


# #### Creation of neural network structure. The last parameter with the value 6 represents, the number of predictive attributes

# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', 
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, poluicao_real, epochs = 100, batch_size = 64)


# #### In this example we will not use a specific database for testing...
# #### we will make the forecasts directly in the training database
# 

# In[ ]:


previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)


# #### Verification of the average in the results of the forecasts and in the actual results

# In[ ]:


print('Previsoes', previsoes.mean())
print('Poluicao', poluicao.mean())


# ### Lets Plot. 
# #### A bar chart will be generated because we have many records

# In[ ]:


plt.figure(figsize=(16,12))
plt.plot(poluicao, color = 'red', label = 'Real pollution')
plt.plot(previsoes, color = 'blue', label = 'Predictions')
plt.title('Pollution forecast')
plt.xlabel('Hours')
plt.ylabel('Pollution value')
plt.legend()


# 
# ### Hello!
# #### Leave your feedback on this notebook ... Thanks

# In[ ]:




