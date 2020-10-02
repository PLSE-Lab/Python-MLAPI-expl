#!/usr/bin/env python
# coding: utf-8

# # HERE I HAVE USED AUTO ARIMA MODEL TO PREDICT COMING NUMBER OF POSITIVE CASE IN INDIA UPTO 16TH MAY  WITH ERROR PERCENTAGE = 0.36% AS TESTED AGAINST ACTUAL TOTAL CONFIRMED CASES ON 6TH MAY i.e 53,007

# In[ ]:


# import os
# print(os.listdir("../input/new_case.xlsx"))


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_excel('../input/new_case.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index(['Date'])
data


# **# these nan vlaue are of future prediction, and these values we needed to predict****

# In[ ]:



# SPLITING DATA INOT TRAIN AND TEST . HERE TELL WILL CONTAIN NAN FIELDS
train = data[:97]
valid = data[97:]

# # #plotting the data
train['Total Confirmed'].plot()
valid['Total Confirmed'].plot()
train


# In[ ]:


valid


# **PIP INSTALL  PMDARIMA     FOR AUTO ARIMA MODEL**

# In[ ]:


get_ipython().system('pip install pmdarima')


# In[ ]:


import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np


# In[ ]:


# Fit your model
model = pm.auto_arima(train, seasonal=True, m=6)


# **IGNORE THESE WARNINGS**

# In[ ]:


# FORECASTING
forecasts = model.predict(valid.shape[0])  # predict N steps into the future


# In[ ]:


# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(data.shape[0])
plt.figure(figsize=(12,10))
plt.plot(x[:97], train, c='blue', label='ACTUAL')
plt.plot(x[97:], forecasts, c='green', label='PREDICTED')

plt.legend(loc='best')
plt.show()


# In[ ]:


for i in forecasts:
    print(i)


# # **PREDICTED VALUES**

# * **6th MAY = 52,812**
# * **7th MAY = 56,158** 
# * **8th MAY = 59,308** 
# * **9th MAY = 62,781** 
# * **10th MAY = 66,204** 
# * **11th MAY = 69,510** 
# * 
