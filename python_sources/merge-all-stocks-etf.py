#!/usr/bin/env python
# coding: utf-8

# 

# This kernel will show you how to combine all of the ETFs and stocks data into a single dataframe for each group.
# * Note: The data (last updated 11/10/2017) is presented in CSV format as follows: Date, Open, High, Low, Close, Volume, OpenInt. Note that prices have been adjusted for dividends and splits.
# * full historical daily price and volume data for all U.S.-based stocks and ETFs trading on the NYSE, NASDAQ, and NYSE MKT. It's one of the best datasets of its kind you can obtain.
# 
# * Source: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
# * Dan Ofer

# In[ ]:


import os
import pandas as pd


# In[ ]:


# kernels let us navigate through the zipfile as if it were a directory
# os.chdir('../input/Data/ETFs/')


# In[ ]:


os.chdir('../input/Data/Stocks/')


# In[ ]:


# os.chdir('../../../input')
# os.listdir()[:5]


# In[ ]:


# the data is initially stored in many small csv files
os.listdir()[:5]


# In[ ]:


data = []
csvs = [x for x in os.listdir() if x.endswith('.txt')]
# trying to read a file of size zero will throw an error, so skip them
csvs = [x for x in csvs if os.path.getsize(x) > 0]
for csv in csvs:
    df = pd.read_csv(csv)
    df['ticker'] = csv.replace('.txt', '')
    data.append(df)
data = pd.concat(data, ignore_index=True)
data.reset_index(inplace=True, drop=True)

print(data.shape)


# In[ ]:


data.head()


# In[ ]:


os.chdir('../../../input')
os.listdir()[:5]


# In[ ]:


os.listdir("../")[:5]


# In[ ]:


data.to_csv("../working/usa_stocks_priceVol_11102017.csv.gz",index=False,compression="gzip")


# ## Read in ETFs 
# * save to seperate file (but could also merge

# In[ ]:


os.chdir('Data/ETFs/')


# In[ ]:


os.listdir()[:4]


# In[ ]:


data = []
csvs = [x for x in os.listdir() if x.endswith('.txt')]
# trying to read a file of size zero will throw an error, so skip them
csvs = [x for x in csvs if os.path.getsize(x) > 0]
for csv in csvs:
    df = pd.read_csv(csv)
    df['ticker'] = csv.replace('.txt', '')
    data.append(df)
data = pd.concat(data, ignore_index=True)
data.reset_index(inplace=True, drop=True)

print(data.shape)


# In[ ]:


os.chdir('../../../input')


# In[ ]:


data.to_csv("../working/usa_ETF_priceVol_11102017.csv.gz",index=False,compression="gzip")


# In[ ]:




