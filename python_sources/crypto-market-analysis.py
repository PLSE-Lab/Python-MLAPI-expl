#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data =pd.read_csv("../input/crypto-markets.csv")
#Exploration of Initial Data
print(data.head())


# In[ ]:


##Explore the columns and the availalbe Data ##
data.columns
df=data[['symbol','date','close']]
df.head()


# In[ ]:


df['date'] = pd.to_datetime(df.date)
#dg=df.groupby(['symbol',pd.Grouper(key='date',freq='1BM')],sort=False).mean()
#dg=df.groupby(['symbol',pd.Grouper(key='date',freq='A')],sort=False)
dg=df.groupby(['symbol',pd.Grouper(key='date',freq='A')],sort=False).agg(['min','max'])
dg.head()


# In[ ]:


#new_df= pd.pivot_table(df,index=['date'], columns = 'symbol', values = "close",
#                       aggfunc=len, fill_value=0)

#new_df= pd.pivot_table(df,index=['date','symbol'], values = "close",
#                       aggfunc=np.mean, fill_value=0)

#new_df= pd.pivot_table(df,index=['symbol'], columns=["date"],
#                       values = ["close",],aggfunc=[np.sum])
df['date'] = pd.to_datetime(df.date)
df['date']=df['date'].dt.strftime('%Y-%m')
#df['date']=df['date'].dt.strftime('%Y')
new_df= pd.pivot_table(df,index=['symbol'],columns=["date"],
                       values = ["close"],aggfunc=[np.mean])
new_df.head(100)

