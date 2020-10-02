#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


from matplotlib.finance import candlestick2_ohlc
import matplotlib.ticker as ticker
import datetime as datetime
from datetime import date

pd.options.display.max_rows = 100


# In[ ]:


# Load only the necessary fields
fields = ['DATPRE', 'CODNEG', 'NOMRES', 'TOTNEG', 'PREMIN', 'QUATOT', 'CODISI', 'VOLTOT', 'PREMED', 'PREABE', 'PREULT', 'PREMAX']
datacsv = pd.read_csv(
    '../input/b3-stock-quotes/COTAHIST_A2009_to_A2018P.csv', 
     usecols=fields
)


# In[ ]:


data = datacsv[datacsv.CODNEG == 'PETR4']
data = data[(data['DATPRE'] >= '2017-12-01') & (data['DATPRE'] < '2019-01-01')]
data.tail()


# In[ ]:


fig, ax = plt.subplots()
candlestick2_ohlc(
    ax, 
    data.PREABE, 
    data.PREMAX, 
    data.PREMIN, 
    data.PREULT, 
    width=0.8 
)

xdate = [i for i in data.DATPRE]
ax.xaxis.set_major_locator(ticker.MaxNLocator(11))

def mydate(pre,pos):
    try:
        return xdate[int(pre)]
    except IndexError:
        return ''
    
#http://rica.ele.puc-rio.br/media/Revista_rica_n7_a10.pdf (esse link fala matematicamente algoritimo genetico)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

fig.autofmt_xdate()
fig.tight_layout()

plt.show()


# In[ ]:




