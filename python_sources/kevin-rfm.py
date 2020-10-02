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
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/datarfmyuk/rfmkevin.csv',  parse_dates=['TrxDate'])
df.head(10)


# In[ ]:


df.info()


# In[ ]:


print(df['TrxDate'].min(), df['TrxDate'].max())


# In[ ]:


sd = dt.datetime(2020,6,9)
df['hist']=sd - df['TrxDate']
df['hist'].astype('timedelta64[D]')
df['hist']=df['hist'] / np.timedelta64(1, 'D')
df.head(10)


# In[ ]:



df=df[df['hist'] < 730]
df.info()


# In[ ]:


rfmTable = df.groupby('CardID').agg({'hist': lambda x:x.min(), # Recency
                                        'CardID': lambda x: len(x), # Frequency
                                        'Amount': lambda x: x.sum()}) # Monetary Value

rfmTable.rename(columns={'hist': 'recency', 
                         'CardID': 'frequency', 
                         'Amount': 'monetary_value'}, inplace=True)
rfmTable.head(6)


# In[ ]:


quartiles = rfmTable.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))


# In[ ]:


#untuk recency
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
#untuk frequency dan monetary
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4 


# In[ ]:


rfmSeg = rfmTable
rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))


# In[ ]:


rfmSeg['RFMClass'] = rfmSeg.R_Quartile.map(str)                             + rfmSeg.F_Quartile.map(str)                             + rfmSeg.M_Quartile.map(str)
rfmSeg.head(10)


# In[ ]:


#Total Class

rfmSeg['Total_Class'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] + rfmSeg['M_Quartile']


rfmSeg.head()


# In[ ]:


print("Pelanggan setia: ",len(rfmSeg[rfmSeg['RFMClass']=='444']))
print('Pelanggan candu: ',len(rfmSeg[rfmSeg['F_Quartile']==4]))
print("Pelanggan on-time payment: ",len(rfmSeg[rfmSeg['M_Quartile']==4]))
print("Pelanggan perihatin: ",len(rfmSeg[rfmSeg['R_Quartile']==1]))
print('Pelanggan kapok: ', len(rfmSeg[rfmSeg['RFMClass']=='111']))
print('Pelanggan ideal: ', len(rfmSeg[rfmSeg['RFMClass']=='333']))
print('Pelanggan dibantu: ', len(rfmSeg[rfmSeg['RFMClass']=='222']))

