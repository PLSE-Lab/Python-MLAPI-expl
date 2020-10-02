#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import datetime
import pandas_datareader.data as wb
from pandas.util.testing import assert_frame_equal


# In[ ]:


start = datetime.datetime(2020,1,1)
end = datetime.datetime(2020,7,16)


# In[ ]:


list_of_tickers = ['MRO',
'LDOS',
'LHX',
'RHP',
'HCAT',
'POL',
'TMHC',
'BLRX',
'SCOR',
'SNCR',
'BKD',
'UNFI',
'BGS',
'HEXO',
'NVTA',
'ENPH',
'NEP',
'SINT',
'CLUB',
'ASR',
'VNO',
'WLMS',
'CVGI',
'DRAD',
'ZUO',
'NVCR',
'YEXT',
'PFE',
'NTLA',
'SPG',
'HSBC',
'STWD',
'ACB',
'CGC',
'UNVR',
'CELH',
'BG',
'LVGO',
'AVID',
'GTES',
'TDG',
'FTI',
'JBLU',
'AIV',
'TROX',
'NTRA',
'APLS',
'NSTG',
'BBIO',
'DENN',
'BHC',
'RESI',
'RPD',
'DRNA',
'BZH',
'ONCY',
'NVUS',
'WEN',
'COG',
'EQT',
'MARK',
'JMIA',
'NET',
'PTON',
'MET',
'BYND',
'USFD',
'TSN',
'VZ',
'CMCSA',
'HOV',
'TOL',
'DVAX',
'SDC',
'ADT',
'SIRI',
'GRPN',
'STNG',
'INO',
'SPRT',
'PDCO',
'HTZ',
'CAR',
'ARPO',
'MPW',
'MOS',
'RF',
'WPM',
'URI',
'ENTA',
'NAT',
'KMI',
'CCL',
'ALKS',
'TDOC',
'DOCU',
'RCL',
'NCLH',
'LEN',
'SPCE',
'DAL',
'MRNA',
'NTNX',
'LUV',
'BA',
'MCK',
'GE',
'OGEN',
'ZS',
'NTAP',
'MSFT',
'CKPT',
'FB',
'TSLA',
'MCRB',
'GNMK',
'ZNGA',
'RESN',
'PLUG',
'GLUU',
'RIGL',
'TEVA',
'AMZN',
'AAPL',
'AMAT',
'NFLX',
'NKTR',
'CHK',
'AMD',
'ALGN',
'HPE',
'OXY',
'BBBY',
'RH',
'REGN',
'TWTR',
'FIT',
'BIOC',
'PDLI',
'MCHP',
'ENTA',
'MMP',
'GOLD',
'BZH',
'RESI',
'CVGI',
'QADA',
'MU',
'HRI',
'EVBG',
'GNMK',
'UNFI',
'BOX',
'SCOR',
'SLNO',
'CNNE',
'MET',
'DAL',
'BHC',
'DRNA',
'GTES',
'HBI',
'OSPN',
'LLNW',
'AAWW',
'KAR',
'RUN',
'GTLS',
'POL',
'HCAT',
'FSLY',
'WDC',
'FFIV',
'RHP',
'CHU',
'SPYD',
'SLP',
'ABMD',
'MITK',
'LHX',
'LMT']


# In[ ]:


p = wb.DataReader(list_of_tickers, 'yahoo',start,end)
res = p.stack().reset_index()
res = res.sort_values(by=['Symbols','Date'], ascending=True)


# In[ ]:


pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)


# In[ ]:


res.head()


# In[ ]:


subset = res[(res.Date.isin(['2020-02-19','2020-03-23','2020-04-29','2020-05-13','2020-06-08','2020-06-26','2020-07-15']))].reset_index()


# In[ ]:


subset = subset.pivot(index='Symbols', columns='Date', values='Adj Close')


# In[ ]:


subset.head(100)


# In[ ]:


subset.info()


# In[ ]:


subset['%_Change'] = (subset['2020-07-15 00:00:00'] - subset['2020-02-19 00:00:00'] )/ subset['2020-02-19 00:00:00'] * 100


# In[ ]:


subset.head(100)


# In[ ]:


subset.to_csv('stocks.csv')


# In[ ]:


import os
os.chdir(r'kaggle/working')


# In[ ]:


from IPython.display import FileLink
FileLink(r'stocks.csv')


# In[ ]:


pwd

