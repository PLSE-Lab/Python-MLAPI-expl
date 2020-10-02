#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Predictig price for Avocado's using ARIMA model


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
apdf = pd.read_csv("../input/avocado.csv")
#printinf dataset
apdf.head()


# In[ ]:


#finding types of avocados
apdf.groupby('type').groups
#There are two types of of avocada
    #Conventional 
    #Organic 


# In[ ]:


df = apdf[apdf.type=='organic']
df.head()


# In[ ]:


#finding regions
regions = df.groupby('region').groups
for reg in regions:
    print(reg)


# In[ ]:


#Lets predictthe price for Albany
state = 'Albany'
data = df.groupby(df.region)
date_price = data.get_group(state)[['Date', 'AveragePrice']].reset_index(drop=True)


# In[ ]:


date_price['Date'] = pd.to_datetime(date_price['Date'])
date_price.plot(x ='Date',y = 'AveragePrice',kind = 'line')


# In[ ]:


from statsmodels.tsa.stattools import adfuller
##Dicky fuller test
date_price.head()


# In[ ]:


train_data = date_price.set_index(['Date'])
train_data.head()


# In[ ]:


avg_price = train_data['AveragePrice']
var = avg_price.rolling(12).std()
mean = avg_price.rolling(12).mean()


# In[ ]:


#plotting var
var.plot()
mean.plot()


# In[ ]:


result = adfuller(avg_price, autolag='AIC')
print('Test statistic: ' , result[0])
print('p-value: '  ,result[1])
print('Critical Values:' ,result[4])


# In[ ]:


import statsmodels.api as sm

mod = sm.tsa.SARIMAX(train_data,order = (3,1,0),seasonal_order=(0,0,0,12))

results = mod.fit()
print(results.aic)
date_price.tail()


# In[ ]:


from datetime import date 
str_date = date(year = 2017,day = 30, month = 7)
pred = results.get_prediction(start = str_date,dynamic = False)
pred_ci = pred.conf_int()
pred_ci['Date'] = pred_ci.index

pred_ci.head()


# In[ ]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[ ]:


cool = date_price.plot(x ='Date',y = 'AveragePrice',kind = 'line')
cool1 = pred_ci.plot(x ='Date',y = 'lower AveragePrice',kind = 'line')
cool2 = pred_ci.plot(x ='Date',y = 'upper AveragePrice',kind = 'line')

