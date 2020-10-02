#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("../input/AAPL.csv")


# In[ ]:


data.shape


# In[ ]:


data.tail()


# In[ ]:


d1 = data.set_index('Date')


# In[ ]:


d1.head()


# In[ ]:


d2 = d1.loc[:,'Close']


# In[ ]:


d2.head()


# In[ ]:


d2.shape


# In[ ]:


d2.plot()


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf


# In[ ]:


plot_acf(d2)


# In[ ]:


diff_d2 = d2.diff(periods=1)


# In[ ]:


diff_d2.head()


# In[ ]:


diff_d2 = diff_d2[1:]


# In[ ]:


diff_d2.head()


# In[ ]:


plot_acf(diff_d2, lags=12)


# In[ ]:


diff_d2.plot()


# In[ ]:


x = d2.values
x.size


# In[ ]:


train = x[:230]
test = x[230:]
train.size, test.size


# In[ ]:


from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA


# In[ ]:


import time
t1 = time.time()
ar_model = AR(train)
ar_fit = ar_model.fit()
ar_pred = ar_fit.predict(start=230, end=250)
print(time.time()-t1)


# In[ ]:


plt.plot(test)
plt.plot(ar_pred, color='red')


# In[ ]:


arima_model = ARIMA(train, order=(2,1,0))
arima_fit = arima_model.fit()
print(arima_fit.aic)
arima_pred = arima_fit.forecast(steps=21)[0]


# In[ ]:


plt.plot(test)
plt.plot(arima_pred, color='red')


# In[ ]:


import itertools
p=d=q = range(0,5)
pdq = list(itertools.product(p,d,q))


# In[ ]:


t1 = time.time()

smallest_aic = 100000
for param in pdq:
    try:
        arima_model = ARIMA(train, order=param)
        arima_fit = arima_model.fit()
        aic = arima_fit.aic
        if aic < smallest_aic:
            smallest_aic = aic
            best = param
    except:
        continue
        
print("smallest aic is ", smallest_aic)
print("best parameters are ", best)

print("total time taken: ", time.time()-t1)


# In[ ]:


arima_model = ARIMA(train, order=best)
arima_fit = arima_model.fit()
print(arima_fit.aic)
arima_pred = arima_fit.forecast(steps=21)[0]
plt.plot(test)
plt.plot(arima_pred, color='red')


# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
order = (1,1,1)
seasonal_order = (1,1,1,12)
sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit()
print(sarima_fit.aic)
sarima_pred = sarima_fit.forecast(steps=21)
plt.plot(test)
plt.plot(sarima_pred, color='red')


# In[ ]:


print(sarima_pred)


# In[ ]:


p=d=q = range(0,5)
P=D=Q = range(0,5)
m = range(0,12)
pdq = list(itertools.product(p,d,q))
PDQm = list(itertools.product(P,D,Q,m))


# In[ ]:


# t1 = time.time()

# min_aic = 10000000
# for t_param in pdq:
#     for s_param in PDQm:
#         try:
#             sarima_model = SARIMAX(train, order=t_param, seasonal_order=s_param)
#             sarima_fit = sarima_model.fit()
#             aic = sarima_fit.aic
#             if aic < min_aic:
#                 best_t = t_param
#                 best_s = s_param
#         except:
#             continue
        
# print("smallest aic is ", min_aic)
# print("best trend parameters are ", best_t)
# print("best seasonal parameters are ", best_s)

# print("total time taken : ", time.time()-t1)

# sarima_model = SARIMAX(train, order=best_t, seasonal_order=best_s)
# sarima_fit = sarima_model.fit()
# sarima_pred = sarima_fit.forecast(steps=21)

# plt.plot(test)
# plt.plot(sarima_pred)

