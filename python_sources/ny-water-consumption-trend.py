#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df_w_consumption = pd.read_csv('../input/water-consumption-in-the-new-york-city.csv', index_col='Year')
print(df_w_consumption.info())
print(df_w_consumption.head())


# In[ ]:


#https://github.com/santosjorge/cufflinks/issues/185
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')


# In[ ]:


import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()


# In[ ]:


#https://stackoverflow.com/questions/46016712/plotly-change-figure-size-by-calling-cufflinks-in-pandas
layout1 = cf.Layout(
    height=800,
    width=1000
)
_=df_w_consumption.iplot(logy=True)


# In[ ]:


df_w_consumption.iplot(kind='box', subplots=True)


# https://github.com/santosjorge/cufflinks/blob/master/Cufflinks%20Tutorial%20-%20Pandas%20Like.ipynb

# In[ ]:


df_w_consumption[['NYC Consumption(Million gallons per day)', 'Per Capita(Gallons per person per day)']].iplot(kind='hist', subplots=True, bins=7)


# In[ ]:


df_w_consumption[['NYC Consumption(Million gallons per day)', 'Per Capita(Gallons per person per day)']].iplot(kind='hist', subplots=True, bins=7, histnorm='probability')


# https://github.com/santosjorge/cufflinks/blob/master/Cufflinks%20Tutorial%20-%20Plotly.ipynb

# In[ ]:


df_w_consumption.iplot(subplots=True)

