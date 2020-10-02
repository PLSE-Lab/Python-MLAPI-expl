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


df = pd.read_csv('../input/retailtransactiondata/Retail_Data_Transactions.csv', parse_dates=['trans_date'])


# In[ ]:


df.head(3)


# Data ini berada pada tingkat transaksi. yaitu satu baris untuk setiap transaksi yang dilakukan oleh pelanggan (perhatikan bahwa tidak pada level item).

# In[ ]:


df.info()


# 

# In[ ]:


print(df['trans_date'].min(), df['trans_date'].max())


# Jumlah hari dari tanggal studi dihitung sebagai berikut.

# In[ ]:


sd = dt.datetime(2015,4,1)
df['hist']=sd - df['trans_date']
df['hist'].astype('timedelta64[D]')
df['hist']=df['hist'] / np.timedelta64(1, 'D')
df.head()


# Hanya transaksi yang dilakukan dalam 2 tahun terakhir yang dipertimbangkan untuk dianalisis.

# In[ ]:


df=df[df['hist'] < 730]
df.info()


# The data will be summarized at customer level by taking *number of days to the latest transaction*, *sum of all transaction amount* and *total number of transaction*.

# In[ ]:


rfmTable = df.groupby('customer_id').agg({'hist': lambda x:x.min(), # Recency
                                        'customer_id': lambda x: len(x),               # Frequency
                                        'tran_amount': lambda x: x.sum()})          # Monetary Value

rfmTable.rename(columns={'hist': 'recency', 
                         'customer_id': 'frequency', 
                         'tran_amount': 'monetary_value'}, inplace=True)


# In[ ]:


rfmTable.head()


# Kami akan memeriksa detail pelanggan 'CS1112'. Tampaknya perhitungannya benar (transaksi terakhir adalah 77 hari yang lalu / jumlah total transaksi adalah 6 / jumlah totalnya adalah 358.

# In[ ]:


df[df['customer_id']=='CS1112']


# Analisis RFM melibatkan pengkategorian R, F dan M ke dalam 3 kategori atau lebih. Untuk kenyamanan, mari buat 4 kategori berdasarkan kuartil (kuartil membagi sampel menjadi 4 bagian dengan proporsi yang sama)

# In[ ]:


quartiles = rfmTable.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))


# mari kita konversi informasi kuartil ke dalam kamus sehingga cutoff dapat diambil.

# In[ ]:


quartiles=quartiles.to_dict()
quartiles


# In[ ]:


# for Recency 

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
## for Frequency and Monetary value 

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[ ]:


rfmSeg = rfmTable
rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))


# Untuk analisis, sangat penting untuk menggabungkan skor menjadi satu skor. Ada beberapa metode. Salah satu metode adalah dengan hanya menggabungkan pecahan untuk membentuk angka 3 digit antara 111 dan 444. Kerugiannya di sini adalah terlalu banyak kategori (4x4x4). Demikian pula, tidak mudah untuk memprioritaskan skor seperti 421 dan 412.

# In[ ]:


rfmSeg['RFMClass'] = rfmSeg.R_Quartile.map(str)                             + rfmSeg.F_Quartile.map(str)                             + rfmSeg.M_Quartile.map(str)


# In[ ]:


rfmSeg.head()


# In[ ]:


rfmSeg.sort_values(by=['RFMClass', 'monetary_value'], ascending=[True, False])


# In[ ]:


rfmSeg.groupby('RFMClass').agg('monetary_value').mean()


# Kemungkinan lain adalah menggabungkan skor menjadi satu skor (misalnya 4 + 1 + 1). Ini akan membawa skor antara 3 dan 12. Keuntungannya di sini adalah bahwa setiap skor memiliki kepentingan yang sama. Tetapi beberapa skor memiliki komponen sebanyak sgements (misalnya, -413 iklan 431)

# In[ ]:


rfmSeg['Total Score'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] +rfmSeg['M_Quartile']
print(rfmSeg.head(), rfmSeg.info())


# In[ ]:


rfmSeg.groupby('Total Score').agg('monetary_value').mean()


# In[ ]:


res = pd.read_csv('../input/retailtransactiondata/Retail_Data_Response.csv')
res.sort_values('customer_id', inplace=True)

print(res.head(), res.info())


# In[ ]:


rfmSeg.reset_index(inplace=True)
rfmSeg.head()


# In[ ]:


rfmSeg.sort_values('customer_id', inplace=True)
rfm2=pd.merge(rfmSeg, res, on='customer_id')


# In[ ]:


rfm2.info()

