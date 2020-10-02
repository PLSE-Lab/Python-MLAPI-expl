#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score


# In[ ]:


df=pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True)


# In[ ]:


df.head()


# In[ ]:


df.index


# In[ ]:


df.index.freq='MS'


# In[ ]:


df.info()


# In[ ]:


df.plot(color='green',figsize=(10,7))


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


decompose=seasonal_decompose(df['S4248SM144NCEN'])
decompose.plot();


# In[ ]:


decompose.seasonal.plot(figsize=(10,7))


# In[ ]:


s_test=adfuller(df['S4248SM144NCEN'])
print("p-value :",s_test[1])


# In[ ]:


span=12
alpha=2/13


# In[ ]:


ses=SimpleExpSmoothing(df['S4248SM144NCEN']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1).rename('simpleexpsmoothing')
desa=ExponentialSmoothing(df['S4248SM144NCEN'],trend='add').fit().fittedvalues.shift(-1).rename('double-expo add')
desm=ExponentialSmoothing(df['S4248SM144NCEN'],trend='add').fit().fittedvalues.shift(-1).rename('double-expo mul')


# In[ ]:


df['S4248SM144NCEN'].plot(figsize=(10,7),legend=True)
ses.plot(legend=True)
desa.plot(legend=True)
desm.plot(legend=True)


# In[ ]:


tesa=ExponentialSmoothing(df['S4248SM144NCEN'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues.rename('tripple-expo add')
tesm=ExponentialSmoothing(df['S4248SM144NCEN'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues.rename('tripple-expo mul')


# In[ ]:


df['S4248SM144NCEN'].plot(figsize=(10,7),legend=True)
tesa.plot(legend=True)
tesm.plot(legend=True)


# In[ ]:


df['S4248SM144NCEN'].iloc[:24].plot(figsize=(10,7),legend=True,)
tesa[:24].plot(legend=True)
tesm[:24].plot(legend=True)


# In[ ]:


print('rmse tesa:',r2_score(df['S4248SM144NCEN'],tesa))
print('rmse tesm:',r2_score(df['S4248SM144NCEN'],tesm))


# In[ ]:


## since rmse of tesm is more near to 1


# In[ ]:


len(df)


# In[ ]:


325-36


# In[ ]:


train=df.iloc[:289]
test=df.iloc[289:]


# In[ ]:


model=ExponentialSmoothing(train['S4248SM144NCEN'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
model_predict=model.forecast(36)


# In[ ]:


train['S4248SM144NCEN'].plot(figsize=(12,7),label='Train',legend=True)
test['S4248SM144NCEN'].plot(legend=True,label='test')
model_predict.plot(legend=True,label='test-predictions')


# In[ ]:


test['S4248SM144NCEN'].plot(figsize=(10,7),legend=True,label='test')
model_predict.plot(legend=True,label='test-predictions')


# In[ ]:


print('rmse:',r2_score(test,model_predict))


# In[ ]:


final_model=ExponentialSmoothing(df['S4248SM144NCEN'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
final_prediction=final_model.forecast(36)


# In[ ]:


df['S4248SM144NCEN'].plot(figsize=(15,10),legend=True)
final_prediction.plot(label='prediction',legend=True)


# In[ ]:


final_prediction


# In[ ]:


date=pd.date_range('2019-02-01',periods=36,freq='MS')


# In[ ]:


predicted_df=pd.DataFrame(data=list(zip(date,final_prediction)),columns=['date','prediction sale'])


# In[ ]:


predicted_df=predicted_df.set_index('date')


# In[ ]:


predicted_df.head()


# In[ ]:




