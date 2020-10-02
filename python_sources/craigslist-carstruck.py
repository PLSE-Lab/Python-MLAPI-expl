#!/usr/bin/env python
# coding: utf-8

# ### 1. IMPORT LIBRARY

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### 2. LOAD DATA

# In[ ]:


data = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv')
data


# > ###### cek dimensi dan info tabel

# In[ ]:


data.info()


# ###### summary statistic of dataset

# In[ ]:


# summary statistic
data.describe()


# In[ ]:


# df.describe()


# ### 3. Data Cleansing

# > ##### 1. Cek missing values

# In[ ]:


# count_missing = data.isnull().sum().sort_values(ascending = False)
percentage_missing=round(data.isnull().sum()/len(data)*100,2).sort_values(ascending = False)
print(percentage_missing)


# ##### 2. Drop kolom
# 1. Size and vin, karena jumlah missing value diatas 50%.
# 2. Url dan image_url, karena field tidak bisa dianalisis.

# In[ ]:


df=data.drop(columns=['size','vin','url','image_url'])
df


# In[ ]:


# data[['county_fips','county_name','state_fips','title_status','state_code','paint_color','drive','transmission', 'manufacturer','make','weather']]


# ##### 3. Fill missing value

# In[ ]:


percentage_missing=round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)
print(percentage_missing)


# In[ ]:


# fill the missing value
categorical = ['type','condition','paint_color','cylinders',
               'drive','manufacturer','make','state_fips',
               'state_code','county_name','county_fips','fuel',
               'transmission','year','title_status']
numerical = ['odometer','weather','price']
numericals = ['odometer','weather']

for num in numericals :
    mean=df[num].mean()
    df[num] = df[num].fillna(mean)

for num in categorical :
    modus=df[num].mode().values[0]
    df[num] = df[num].fillna(modus)

df


# ##### 4. Cek outlier

# In[ ]:


# boxplot weather

f=plt.figure(figsize=(13,12))
f.add_subplot(3,3,1)
sns.boxplot(df['weather'],orient = "h")
f.add_subplot(3,3,2)
sns.boxplot(df['odometer'],orient = "h")
f.add_subplot(3,3,3)
sns.boxplot(df['price'],orient = "h")
f.add_subplot(3,3,4)
sns.boxplot(df['year'],orient = "h")


# In[ ]:


# drop field odometer, karena banyak sekali outlier. Price tidak dihapus karena akan digunakan untuk analisis.
df1=df.drop(columns=['odometer'])
df1.info()


# ### 4. INSIGHT VISUALISASI

# In[ ]:


df1


# #### 1. Distribution of Year (last 50 years)[](http://)

# In[ ]:


print('Year max')
df1[['year']].sort_values('year',ascending=False).head(1)


# In[ ]:


# distribution of year (last 50 years)
year50=df1[df['year']>1969]
plt.figure(figsize=(24,10))
ax3 = sns.distplot(year50['year'])
ax3.set(title="Distribusi Mobil Dalam 50 Tahun Terakhir")
# Terlihat bahwa distribusi mobil pada tahun 1970 sampai dengan tahun 2007 mengalami trend yang meningkat,
# hal ini menunjukkan bahwa demand terhadap mobil pada rentang tahun tersebut meningkat.
# Berbeda dengan distribusi mobil pada tahun 2007 sampai sekarang, terlihat bahwa grafik mengalami perubahan yang
# fluktuatif dan cenderung menurun pada tahun 2016. Hal ini menunjukkan bahwa demand terhadap mobil tahun 2007 sampai 2019 menurun.


# In[ ]:


# mean price each year
year50=df1[df['year']>1969]
priceyear=year50[['year', 'price']]
priceyears=priceyear.groupby('year').mean().sort_values('year',ascending=True)
len(priceyears)


# In[ ]:


plt.figure(figsize=(20,3))

x=range(50)
plt.bar(x,priceyears['price'])
plt.xticks(x,priceyears.index)
plt.xlabel('Year')
plt.ylabel('Mean Price')
plt.title('Mean Price of Each Year')
plt.show()

# terlihat bahwa rata-rata harga di setiap tahunnya mengalami pertumbuhan yang fluktuatif.


# > #### 2. Comparison fuel type

# In[ ]:


print(df1.fuel.unique())


# In[ ]:


# count of fuel in last 50 years
count=year50['fuel'].value_counts()
count


# In[ ]:


plt.figure(figsize=(7,3))
sns.countplot(year50['fuel'], order = year50['fuel'].value_counts().index)
plt.title('Number of Fuel Type in Last 50 Year')
plt.show()

# Terlihat bahwa tipe fuel gas mendominasi dalam 50 tahun terakhir. Hal ini dapat dijadikan insight, untuk mengeluarkan
# mobil dengan tipe fuel gas.


# In[ ]:


# mean price for each fuel type
pricefuel=year50[['fuel', 'price']]
pricefuels=pricefuel.groupby('fuel').mean().sort_values('price',ascending=False).head()
pricefuels


# In[ ]:


plt.figure(figsize=(7,3))

x=range(5)
plt.bar(x,pricefuels['price'])
plt.xticks(x,pricefuels.index)
plt.xlabel('Fuel Type')
plt.ylabel('Mean Price')
plt.title('Mean Price of Each Fuel Type')
plt.show()

# rata-rata harga yang paling tinggi adalah tipe fuel diesel, sedangkan rata-rata harga untuk tipe fuel gas
# masuk ke dalam rentang kurang dari 100.000 dollar. Hal inilah yang menyebabkan tipe fuel gas banyak diproduksi.


# #### 3. Manufacturer Distribution[](http://)

# In[ ]:


# count of manufacturer distribution in last 50 years
car=year50['manufacturer'].value_counts().head(10)
car


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(year50['manufacturer'], order = year50['manufacturer'].value_counts().iloc[:10].index, hue=year50['fuel'])
plt.title('Top 10 Manufacturer Distribution in Last 50 Year')
plt.show()

# Terlihat bahwa ford mendominasi distribusi mobil dalam kurun waktu 50 tahun terakhir, dimana tipe fuel yang paling banyak
# untuk setiap manufakturnya adalah gas. Hal ini menunjukkan bahwa minat masyarakat terhadap mobil dengan tipe fuel gas
# masih tinggi. Salah satunya karena harganya yang cenderung murah jika dibandingkan dengan tipe diesel (grafik nomer 2).


# In[ ]:


# mean price for each manufacturer
manufacturer=year50[['manufacturer', 'price']]
manufacturers=manufacturer.groupby('manufacturer').mean().sort_values('price',ascending=False).head(10)
manufacturers


# In[ ]:


plt.figure(figsize=(12,3))

x=range(10)
plt.bar(x,manufacturers['price'])
plt.xticks(x,manufacturers.index)
plt.xlabel('Fuel Type')
plt.ylabel('Mean Price')
plt.title('Top 10 Mean Price of Each Manufacturer')
plt.show()

# terlihat bahwa manufaktur chev memiliki rata-rata harga yang paling tinggi, sehingga demand terhadap mobil yang
# dikeluarkan oleh manufaktur ini sedikit. Hal itulah yang menyebabkan distribusi chev dalam 50 tahun
# terakhir tidak masuk ke dalam top 10 manufaktur distribusi dalam 50 tahun terakhir.


# #### 4. Distribution of each state

# In[ ]:


# number of distribution each state
plt.figure(figsize=(15,5))
yearss=year50[year50['state_name']!='FAILED']
sns.countplot(yearss['state_name'], order = yearss['state_name'].value_counts().iloc[:10].index)
plt.title('Top 10 State Distribution in Last 50 Year')
plt.show()

# Terlihat bahwa California menjadi negara dengan distribusi mobil paling banyak dalam 50 tahun terakhir. 10 negara ini bisa
# dijadikan acuan untuk manufaktur dalam mendistribusikan mobilnya.


# In[ ]:




