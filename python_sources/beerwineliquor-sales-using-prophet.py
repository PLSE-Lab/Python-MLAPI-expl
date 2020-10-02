#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from fbprophet import Prophet

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/BeerWineLiquor.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


# Column names for Prophet should be ds and y, so renaming the columns of above dataset
data.columns = ['ds', 'y']


# In[ ]:


data.head()


# In[ ]:


# Converting ds column of above dataset to datetime
data['ds'] = pd.to_datetime(data['ds'])


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


m = Prophet()
m.fit(data)


# In[ ]:


# Placeholder to hold future predictions
future = m.make_future_dataframe(periods=24, freq='MS')


# In[ ]:


len(data)


# In[ ]:


len(future)


# In[ ]:


forecast = m.predict(future)


# In[ ]:


forecast.shape


# In[ ]:


forecast.head()


# In[ ]:


forecast.columns


# In[ ]:


forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].tail(12)


# In[ ]:


m.plot(forecast);


# In[ ]:


# To remove "ConversionError: Failed to convert value(s) to axis units: '2014-01-01'" error after running the below code
pd.plotting.register_matplotlib_converters()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
m.plot(forecast)
plt.xlim('2014-01-01', '2020-01-01')


# In[ ]:


forecast.plot(x='ds', y='yhat', figsize=(8,10))


# In[ ]:


m.plot_components(forecast);


# In[ ]:




