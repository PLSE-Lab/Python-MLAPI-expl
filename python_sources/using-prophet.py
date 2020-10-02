#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import random
import seaborn as sns


# In[ ]:


import pandas as pd
avocado = pd.read_csv("../input/avocado-prices/avocado.csv")


# In[ ]:


avocado=avocado.sort_values(['Date'])


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(avocado['Date'],avocado['AveragePrice'],color='red')


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(x='region',data=avocado)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(x='year',data=avocado)


# In[ ]:


avocado_prophet=avocado[['Date','AveragePrice']]
avocado_prophet=avocado_prophet.rename(columns={'Date':'ds','AveragePrice':'y'})


# In[ ]:


avocado_prophet


# In[ ]:


m=Prophet()
m.fit(avocado_prophet)


# In[ ]:


future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)


# In[ ]:


forecast


# In[ ]:


figure=m.plot(forecast,xlabel='Date',ylabel='Price')


# In[ ]:


fig=m.plot_components(forecast)
