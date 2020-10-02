#!/usr/bin/env python
# coding: utf-8

# # libraries

# In[ ]:


import numpy as np
import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt


# In[ ]:


def parser(x):
    return datetime.strptime(x, "%Y-%m")


# In[ ]:


data = pd.read_csv("../input/sales-cars.csv", index_col = 0, parse_dates=[0], date_parser = parser)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.plot()


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data)


# In[ ]:


data.shift(1)


# In[ ]:


diff_data = data.diff(periods=1)


# In[ ]:


diff_data.head()


# In[ ]:


diff_data = diff_data[1:]


# In[ ]:


plot_acf(diff_data)


# In[ ]:


diff_data.head()


# In[ ]:


diff_data.plot()


# In[ ]:


x = data.values
train = x[:27]
test = x[27:]


# In[ ]:


train.size, test.size


# In[ ]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
ar_model = AR(train)
ar_fit = ar_model.fit()


# In[ ]:


predictions = ar_fit.predict(start=27, end=36)


# In[ ]:


plt.plot(test)
plt.plot(predictions, color='red')


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
arima_model = ARIMA(train, order=(2,1,0))
arima_fit = arima_model.fit()


# In[ ]:


print(arima_fit.aic)


# In[ ]:


predictions = arima_fit.forecast(steps=9)[0]


# In[ ]:


print(predictions)


# In[ ]:


plt.plot(test)
plt.plot(predictions, color='red')


# In[ ]:


import itertools
p=d=q = range(0,5)
pdq = list(itertools.product(p,d,q))


# In[ ]:


pdq


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


for param in pdq:
    try:
        arima_model = ARIMA(train, order=param)
        arima_fit = arima_model.fit()
        aic = arima_fit.aic
        print(param, aic)
    except:
        continue


# In[ ]:


arima_model = ARIMA(train, order=(3,2,4))
arima_fit = arima_model.fit()
predictions = arima_fit.forecast(steps=9)[0]
print(predictions)


# In[ ]:


plt.plot(test)
plt.plot(predictions, color='red')

