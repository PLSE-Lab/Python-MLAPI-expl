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
from matplotlib import pyplot as plt


# In[ ]:


plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot


# In[ ]:


pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame
pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame


# In[ ]:


promosi = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')
offline = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')
online = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
produk = pd.read_csv('/kaggle/input/uisummerschool/Product.csv')


# In[ ]:


promosi


# In[ ]:


promosi.describe()


# In[ ]:


promosi['total']=promosi['Offline Spend']+promosi['Online Spend']


# In[ ]:


promosi


# In[ ]:


promosi.plot(kind='line',x='Date',y='total')


# In[ ]:


promosi.set_index('Date')[['Offline Spend','Online Spend']].plot()
# promosi.plot(kind='line',x='Date',y='Offline Spend')
# promosi.plot(kind='line',x='Date',y='Online Spend')
# plt.show()


# In[ ]:


# import seaborn as sns


# In[ ]:


promosi.plot(kind='box',x='Date',y='total')


# In[ ]:


promosi.set_index('Date')[['Offline Spend','Online Spend','total']].plot(kind='box')


# In[ ]:


promosi.hist('Offline Spend')
promosi.hist('Online Spend')
promosi.hist('total')


# In[ ]:


# df = pd.DataFrame('Date')
# pd.to_datetime('13000101', format='%Y%m%d', errors='ignore')
# pd.to_datetime('13000101', format='%Y%m%d', errors='coerce')
# pd.to_datetime(,)


# In[ ]:


promosi['tanggal'] = pd.to_datetime(promosi['Date']).dt.strftime('%d')
promosi['bulan'] = pd.to_datetime(promosi['Date']).dt.strftime('%m')
promosi['tahun'] = pd.to_datetime(promosi['Date']).dt.strftime('%y')


# In[ ]:


promosi


# In[ ]:


bof = promosi.groupby('bulan')['Offline Spend'].sum().reset_index(name='monthly offline')
bon = promosi.groupby('bulan')['Online Spend'].sum().reset_index(name='monthly online')


# In[ ]:


bof


# In[ ]:


bon


# In[ ]:


# total_bulanan = pd.DataFrame(np.array([bof['monthly offline'],bon['monthly online']]))
# total_bulanan
# total_bulanan['offline bulanan']=bof('monthly offline')
# total_bulanan['online bulanan']=bon('monthly online')


# In[ ]:


# total_bulanan


# Tahapan selanjutnya: Menguji ada tidaknya perbedaan rata-rata promosi bulanan antara offline dan online (menggunakan t-test)

# In[ ]:


promosi.groupby('bulan')['Offline Spend'].sum().reset_index(name='monthly offline').plot(kind='bar')


# In[ ]:


offline


# Tahapan Selanjutnya:
# * Merangkum data hasil penjualan tiap hari
# * Merangkum data hasil penjualan tiap bulan
# * Membuat plot time series dari dua data di atas
# * Merangkum data hasil penjualan dari 10 produk yang penjualannya paling tinggi
# * Merangkum data hasil penjualan dari 10 produk yang penjualannya paling rendah
# * Membuat bar chart dari dua data di atas
# 

# In[ ]:


penjualan_offline = offline.groupby(['InvoiceDate'])['Quantity'].sum().reset_index(name='sales_offline')
penjualan_offline


# In[ ]:


online


# In[ ]:


penjualan_online = online.groupby(['Date'])['Quantity'].sum().reset_index(name='sales_online')
penjualan_online


# Menguji rata-rata hasil penjualan offline dengan penjualan online sama atau tidak 
# Tahapan: 
# Uji homogenitas varians
# Uji rata-rata dua populasi independent

# In[ ]:


import pandas as pd 
import scipy.stats as stats


# In[ ]:


stats.levene(penjualan_offline.sales_offline.dropna(),penjualan_online.sales_online.dropna())


# In[ ]:


stats.ttest_ind(penjualan_offline.sales_offline.dropna(),penjualan_online.sales_online.dropna(),equal_var=False)


# Karena hasil pengujian perbedaan rata-rata penjualan offline dan online tidak sama (yaitu lebih besar penjualan offline), maka langkah selanjutnya adalah membuat model regresi linier, dengan variabel berikut:
# Y : hasil penjualan offline
# X1: promosi offline
# X2: promosi online

# Tahapan selanjutnya:
# * Merangkum data hasil penjualan tiap hari
# * Merangkum data hasil penjualan tiap bulan
# * Membuat plot time series dari dua data di atas
# * Merangkum data hasil penjualan dari 10 produk yang penjualannya paling tinggi
# * Merangkum data hasil penjualan dari 10 produk yang penjualannya paling rendah
# * Membuat bar chart dari dua data di atas
# 
# * Merangkum data dari Product Category (Enhanced E-commerce) berdasarkan quantity
# * Merangkum data dari Product Category (Enhanced E-commerce) berdasarkan total revenue
# * Mengambil 10 nilai terbesar dari keduanya
# * Mengambil 10 nilai terkecil dari keduanya
# * Membuat plot dari 10 nilai terbesar dari keduanya
# * Membuat plot dari 10 nilai terkecil dari keduanya
# 
# 

# In[ ]:


produk


# * Menggabungkan quantity dari offline dan online dari produk yang sama dan membuat bar plot dari 10 data terbesar quantity-nya
# * Merangkum data hasil penjualan tiap hari (dari offline dan online)
# * Membuat plot time series dari data di atas
# * Forecasting data hasil penjualan tiap hari menggunakan metode Naive, Exponential Smoothing, ARIMA, etc.

# **TIME SERIES FORECASTING**

# In[ ]:


grouped = online.groupby(['Date'])['Revenue'].sum().reset_index(name='total_revenue')
grouped['Date'] = grouped['Date'].astype(str)
grouped['Date'] = pd.to_datetime(grouped['Date'])
grouped.head()


# In[ ]:


grouped.plot(kind='line',x='Date',y='total_revenue')


# In[ ]:


grouped.plot(kind='box',x='Date',y='total_revenue')


# In[ ]:


grouped['d1'] = grouped['total_revenue'].shift()
grouped['d2'] = grouped['total_revenue'].shift(2)
grouped['d3'] = grouped['total_revenue'].shift(3)
grouped['d4'] = grouped['total_revenue'].shift(4)


# In[ ]:


grouped


# In[ ]:


data = grouped[4:]
data


# In[ ]:


y = pd.DataFrame(data, columns = ['total_revenue']) 
train_y = y[0:316]
train_y


# In[ ]:


test_y = y[316:]
test_y


# In[ ]:


x = pd.DataFrame(data, columns = ['Date','d1', 'd2','d3','d4']) 
train_x = x[0:316]
train_x


# In[ ]:


test_x = x[316:]
test_x


# In[ ]:


from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# In[ ]:


model = sm.OLS(train_y['total_revenue'],train_x[['d1','d2','d3','d4']])
hasil = model.fit()
print(hasil.summary())


# In[ ]:


data = grouped[3:]
data


# In[ ]:


x = pd.DataFrame(data, columns = ['Date','d1', 'd2','d3']) 
train_x = x[0:317]
train_x


# In[ ]:


test_x = x[317:]
test_x


# In[ ]:


y = pd.DataFrame(data, columns = ['total_revenue']) 
train_y = y[0:317]
train_y


# In[ ]:


test_y = y[317:]
test_y


# In[ ]:


model2 = sm.OLS(train_y['total_revenue'],train_x[['d1','d2','d3']])
hasil2 = model2.fit()
print(hasil2.summary())


# In[ ]:


# model2.predict(test_x[['d1','d2','d3']])


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


regresi = LinearRegression()  
regresi.fit(train_x[['d1','d2','d3']],train_y['total_revenue'])
print(model)


# In[ ]:


y_cap = regresi.predict(train_x[['d1','d2','d3']])
e2_train = train_y['total_revenue']-y_cap


# In[ ]:


prediksi_y = regresi.predict(test_x[['d1','d2','d3']])
prediksi_y


# In[ ]:


e2_test = test_y['total_revenue']-prediksi_y
e2_test


# In[ ]:


mse2_train = (e2_train**2).sum()/len(e2_train)
mse2_train


# In[ ]:


mse2_test = (e2_test**2).sum()/len(e2_test)
mse2_test


# In[ ]:


from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot


# In[ ]:


autocorrelation_plot(train_y['total_revenue'])


# In[ ]:


from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA


# In[ ]:


model3 = ARIMA(train_y['total_revenue'], order=(4,1,0))
model3_fit = model3.fit(disp=0)
print(model3_fit.summary())


# In[ ]:


e3_train = model3_fit.resid
mse3_train = (e3_train**2).sum()/len(e3_train)
mse3_train


# In[ ]:




