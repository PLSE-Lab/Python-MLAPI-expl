#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot
from fbprophet import Prophet

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


column_types = {
    'store':'int8',
    'item':'int8',
    'sales':'float64',
}
train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv',dtype=column_types,parse_dates=['date'])
test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv',dtype=column_types,parse_dates=['date'])
submission = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


print('Train from {%s} to {%s}' % (train.date.min(),train.date.max()))
print('Test from {%s} to {%s}' % (test.date.min(),test.date.max()))


# ## PREPROCESS
# 
# Log1p Transform for sales.
# 
# 
# - all_data['SalePrice'] = np.log1p(all_data['SalePrice'])
# - predictions[name] = np.expm1(model.predict(train_x))

# In[ ]:


pre_skew = train['sales'].skew()
pre_kurt = train['sales'].kurt()

train['sales'] = np.log1p(train['sales'])

print('Training Set Sales Skew from {%f} downto {%f}' % (pre_skew,train['sales'].skew()))
print('Training Set Sales Kurtosis from {%f} downto {%f}' % (pre_kurt,train['sales'].kurt()))


# ### Prophet
# 
# only item==1&store==1.
# 
# https://blog.csdn.net/anshuai_aw1/article/details/83412058

# In[ ]:


rcParams['figure.figsize'] = 10, 5
train_ph = train[train.store==1][train.item==1][['date','sales']].copy().rename(index=str, columns={"date": "ds", "sales": "y"})
ph = Prophet()
ph.fit(train_ph)
forecast = ph.predict(train_ph[['ds']])
figure = ph.plot(forecast)
figure.show()


# In[ ]:


y = []
y_hat_ph = []
models = {}
for k,si in train.groupby(['store','item']):
    print(k,si.sales.min(),si.sales.max())
    _si = si[['date','sales']]
    _si = _si.rename(index=str, columns={'date':'ds','sales':'y'})
#     y+=(_si.y.tolist())
    model = Prophet()
    model.fit(_si)
#     y_hat_ph+=(ph.predict(_si[['ds']]).yhat.tolist())
    models[str(si.store.iloc[0])+'_'+str(si.item.iloc[0])] = model


# In[ ]:


# rcParams['figure.figsize'] = 20, 5
# plt.plot(y[:200])
# plt.plot(y_hat_ph[:200])
# plt.show()


# In[ ]:


# rmse = np.sqrt(mean_squared_error(y, y_hat_ph))
# print("The root mean squared error with prophet between y and y_hat_ph is {%f}." % rmse)


# ### Build Submission File

# In[ ]:


# 31+28+31=90
# future = m.make_future_dataframe(periods=90, freq='D')

future_data = models[str(1)+'_'+str(1)].make_future_dataframe(periods=90, freq='D')

# for....
forecast_data = models[str(1)+'_'+str(1)].predict(future_data)
forecast_data.iloc[-(365*2+90):].yhat.plot()


# In[ ]:


forecast_data.iloc[-10:]


# In[ ]:


forecast_datas = pd.DataFrame({'ds':[],'store':[],'item':[],'yhat':[]})
for k,si in train.groupby(['store','item']):
    print(k)
    forecast_data = models[str(si.store.iloc[0])+'_'+str(si.item.iloc[0])].predict(future_data)
    forecast_data['store'] = si.store.iloc[0]
    forecast_data['item'] = si.item.iloc[0]
    forecast_datas = forecast_datas.append(forecast_data.iloc[-90:][['ds','store','item','yhat']],ignore_index=True)
forecast_datas.info()


# In[ ]:


forecast_datas.rename(columns={'ds':'date'}, inplace = True)
test = test.merge(forecast_datas, on=['store','item','date'], how='left')
submission = submission.merge(test[['id','yhat']], on=['id'], how='left')
submission = submission.drop(['sales'], axis=1)
submission.rename(columns={'yhat':'sales'}, inplace=True)
submission.sales =  submission.sales.apply(np.expm1)
submission = submission.sort_values(by='id')
submission.head(10)


# In[ ]:


submission.to_csv('submission.csv', index=None)


# ## Summary
# 
# The end.
