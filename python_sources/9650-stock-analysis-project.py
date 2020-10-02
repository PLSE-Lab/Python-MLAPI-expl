#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style 
import pandas as pd
import numpy as np


# In[2]:


style.use('ggplot')


# In[3]:


##Get close price data from Yahoo Finance (except crude oil)


# In[4]:


start=dt.datetime(2018,1,1)
end=dt.datetime(2018,12,31)


# In[5]:


stock_list=['AAPL','AMZN','FB','XOM','WMT','TSCO','T','MRK','BA','PEP','MMM','NEE','DWDP','CCI','WBA','FDX','APC','PPL','AAL','NDAQ','COO','TIF','LNT','TTWO','JWN','CL','EURUSD=X','^GSPC','^TNX']


# In[6]:


file=open('../input/Stock Picks (trimmed).txt','r')
for line in file:
    print(line)


# In[11]:


import fix_yahoo_finance as fyf
from pandas_datareader import data as pdr


# In[ ]:


fyf.pdr_override()


# In[ ]:


data=pdr.get_data_yahoo(stock_list,start,end)
data['Close'].head()


# In[ ]:


##Get crude oil close price from Quandl 


# In[ ]:


pip install quandl


# In[ ]:


import quandl
symbol='CHRIS/CME_CL1'
crude_oil=quandl.get(symbol,start_date='2018-01-01',end_date='2018-12-31')
crude_oil['Last'].tail()


# In[ ]:


##Merge two tables 


# In[ ]:


merge_table=pd.merge(data['Close'],crude_oil['Last'],how='left',left_on='Date',left_index=True,right_index=True)
merge_table.head()


# In[ ]:


stock_dataset_ticker=merge_table.rename(columns={'Last':'CL1'})
stock_dataset_ticker


# In[ ]:


symbol_name=[]
company_name=[]
with open("C:\\Users\\is-mi\\Desktop\\Stock Picks (trimmed).txt",'r') as f:
    aa=f.readlines()[1:]
    for line in aa:
        symbol_name.append(line.strip().split('\t')[0])
        company_name.append(line.strip().split('\t')[1])
dictionary=dict(zip(symbol_name,company_name))


# In[ ]:


stock_dataset.rename(columns=dictionary)


# In[ ]:


##Calculate daily percent change


# In[ ]:


daily_change=stock_dataset.pct_change()
daily_change


# In[ ]:


##Correlation Table


# In[ ]:


corr_data=daily_change.corr()
corr_data


# In[ ]:


##Strongest Correlation Cofficient 


# In[ ]:


corr_rank=corr_data.unstack().sort_values(ascending=False).drop_duplicates()
corr_rank


# In[ ]:


strongest_positive_corr=corr_rank.iloc[[1]]
strongest_positive_corr


# In[ ]:


strongest_negative_corr=corr_rank.iloc[[-1]]
strongest_negative_corr


# In[ ]:


import seaborn as sns


# In[ ]:


f,ax=plt.subplots(figsize=(30,30))

mp1=sns.heatmap(corr_data,annot=True,linewidths=.5,fmt='.1f',cmap='YlGnBu',ax=ax)  
mp1.set_xticklabels(company_name,rotation=40)

mp1.set_yticklabels(company_name)

plt.show()

