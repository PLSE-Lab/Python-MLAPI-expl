#!/usr/bin/env python
# coding: utf-8

# # Prediksi Covid19 di Indonesia menggunakan Model Polynomial Regression

# Pada kesempatan ini saya akan menggunakan model Polynomial untuk memperkirakan laju penyebaran Covid19 sampai dengan akhir April 2020. Polynomial Regression membuat sekenario terburuk terhadap laju pertumbuhan Covid19, sekenario terburuk dipilih sebagai langkah agar pemerintah dan kita semua lebih serius untuk bersama sama melawan Covid19 ini.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# Kasus pertama di Indonesia dilaporkan pada 01 Maret 2020, tanggal ini akan kita jadikan sebagai hari ke-0 untuk memulai prediksi

# In[ ]:


df = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/confirmed_acc.csv', parse_dates=['date'])
df = df.iloc[39:]
df.reset_index(inplace=True)
df['day'] = np.arange(df.shape[0])
df = df[['date','day','cases']] 
df


# In[ ]:


plt.plot(df['day'],df['cases'],marker='o')
plt.show()


# ## Build model polynomial

# Model polynomial ini menggunakan polynomial 5 degree

# In[ ]:


x = np.array(df['day']) 

poly = PolynomialFeatures(5, include_bias=False) ## polynomial with 5 degree
poly.fit(x.reshape(-1,1))
day_poly = poly.transform(x.reshape(-1,1))


# In[ ]:


model = LinearRegression()
model.fit(day_poly, df['cases'])
case_pred = model.predict(day_poly)
case_pred


# In[ ]:


r2 = r2_score(df['cases'],case_pred)
rmse = np.sqrt(mean_squared_error(df['cases'],case_pred))
print(r2)
print(rmse)


# In[ ]:


plt.scatter(df['day'],df['cases'],marker='o')
plt.plot(df['day'],case_pred, 'y-')
plt.show()


# ### Prediksi sampai akhir bulan April 2020

# In[ ]:


date_pred = pd.date_range('20200408', periods = 23)
day_pred = np.arange(38,38+23)
cases_pred = model.predict(poly.fit_transform(day_pred.reshape(-1,1))).round()


# In[ ]:


prediksi = pd.DataFrame({'date_pred': date_pred, 'day_pred': day_pred, 'cases_pred': cases_pred})
prediksi['real'] = np.NaN * 23
prediksi['real'][0:3] = [2956,3293,3512] ## evaluasi prediksi mulai 08 April
prediksi


# In[ ]:


sns.set()
plt.figure(figsize=(16,8))
# plt.subplot(121)
# plt.scatter(df['day'],df['cases'])
# plt.plot(df['day'],case_pred, 'r-')
# plt.title('Kasus COVID19 di Indonesia 01 Maret - 07 April 2020 ')
# plt.xlabel('Hari')
# plt.ylabel('Total Kasus')
# plt.subplot(122)
plt.scatter(df['day'],df['cases'])
plt.plot(prediksi['day_pred'],prediksi['cases_pred'], 'g-')
plt.title('Prediksi Kasus COVID19 di Indonesia s/d 30 April 2020 ')
plt.xlabel('Hari')
plt.ylabel('Total Kasus')
# plt.savefig('covid19.png') ### save image
plt.show()


# Diperkirakan sampai akhir April 2020, kasus COVID19 di Indonesia bisa mencapai 25000 lebih kasus, tentu laju pertumbuhan kasus ini sangat dipengaruhi oleh kebijakan pemerintah dan seberapa disiplin kita untuk bersama sama menekan laju pertumbuhan covid19 ini, semoga kebijakan pemerintah dapat efektif menahan laju kasus covid19 dan laju pertumbuhan kasus covid19 bisa dibawah angka prediksi.

# In[ ]:




