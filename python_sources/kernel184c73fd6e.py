#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.preprocessing import Imputer 
from pylab import savefig
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Craigslist Vehicles**

# Craigslist merupakan koleksi kendaraan bekas untuk dijual paling besar di dunia. Data craigslist berikut merupakan entry data milik Amerika Serikat. Data yang tersedia meliputi spesifikasi dari kendaraan bekas yang akan dijual. Terdiri dari 26 variabel untuk menggambarkan kondisi kendaraan sebelum dijual.

# In[ ]:


vehicles2=pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv')


# Tipe Data

# In[ ]:


vehicles2.info()


# In[ ]:


vehicles2.head()


# Penghapusan Variabel URL dan Image URL

# In[ ]:


vehicles=vehicles2.drop(['url', 'image_url'], axis=1)


# **Identifikasi Missing Value**

# In[ ]:


count_missing=vehicles.isnull().sum().sort_values(ascending=False)
count_missing.head(22)


# In[ ]:


# Persentase Missing Value
print(' Missing Value Size :', round(count_missing[0]/1723065*100), '%')
print(' Missing Value Vin :', round(count_missing[1]/1723065*100), '%')
print(' Missing Value Type :', round(count_missing[2]/1723065*100), '%')


# Jumlah Missing Value yang melebihi 60% selanjutnya akan dilakukan penghapusan variabel

# In[ ]:


vehicles_new=vehicles.drop(['size', 'vin'], axis=1)
vehicles_new.head()


# **Identifikasi Karakteristik Data Kendaraan Bekas**

# In[ ]:


vehicles_new.describe()


# **Pembagian Data Kategorik dan Numerik**

# In[ ]:


numerik=['price','year','odometer','lat','long','weather','county_fips']
categorical=vehicles_new.drop(numerik,axis=1)
numerical=vehicles_new[numerik]
categorical.head()


# **Mengatasi Missing Value Data Dengan Modus**

# Imputasi missing value pada data kategorik dilakukan dengan menggunakan modus agar tidak merubah macam kelompok yang telah terbentuk.

# In[ ]:


vehicles_new['city'] = vehicles_new['city'].fillna((vehicles_new['city'].mode()[0]))
vehicles_new['manufacturer'] = vehicles_new.fillna((vehicles_new['manufacturer'].mode()[0]))
vehicles_new['make'] =vehicles_new['make'].fillna((vehicles_new['make'].mode()[0]))
vehicles_new['condition'] = vehicles_new['condition'].fillna((vehicles_new['condition'].mode()[0]))
vehicles_new['cylinders'] = vehicles_new['cylinders'].fillna((vehicles_new['cylinders'].mode()[0]))
vehicles_new['fuel'] =vehicles_new['fuel'].fillna((vehicles_new['fuel'].mode()[0]))
vehicles_new['title_status'] = vehicles_new['title_status'].fillna((vehicles_new['title_status'].mode()[0]))
vehicles_new['transmission'] = vehicles_new['transmission'].fillna((vehicles_new['transmission'].mode()[0]))
vehicles_new['drive'] = vehicles_new['drive'].fillna((vehicles_new['drive'].mode()[0]))
vehicles_new['type'] = vehicles_new['type'].fillna((vehicles_new['type'].mode()[0]))
vehicles_new['paint_color'] = vehicles_new['paint_color'].fillna((vehicles_new['paint_color'].mode()[0]))
vehicles_new['county_name'] = vehicles_new['county_name'].fillna((vehicles_new['county_name'].mode()[0]))
vehicles_new['state_fips'] = vehicles_new['state_fips'] .fillna((vehicles_new['state_fips'] .mode()[0]))
vehicles_new['state_code'] = vehicles_new['state_code'].fillna((vehicles_new['state_code'].mode()[0]))
vehicles_new['year'] = vehicles_new['year'].fillna((vehicles_new['year'].mode()[0]))
vehicles_new['county_fips'] = vehicles_new['weather'].fillna((vehicles_new['county_fips'].mode()[0]))


# **Mengatasi Missing Value dengan Median**

# Imputasi missing value dengan menggunakan median dikarenakan bentuk data skew.

# In[ ]:


vehicles_new['odometer'] = vehicles_new['odometer'].fillna((vehicles_new['odometer'].median()))
vehicles_new['weather'] = vehicles_new['weather'].fillna((vehicles_new['weather'].median()))
count_missing1=vehicles_new.isnull().sum().sort_values(ascending=False)
count_missing1.head(22)


# **Analisis Univariat**

# > ** Barplot Tipe Kendaraan Bekas di Amerika**

# In[ ]:


vehicles_new.type.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title('Tipe Kendaraan Bekas')
plt.ylabel('Jumlah Kendaraan')
plt.xlabel('Tipe Kendaraan');


# Dari total 1.723.065 data kendaraan bekas, 3 tipe kendaraan teratas adalah sedan, suv, dan truck.

# **Barplot Bahan Bakar Kendaraan**

# In[ ]:


vehicles_new['fuel'].value_counts().plot(kind='bar',color='mediumturquoise')
plt.title("Bahan Bakar Kendaraan")
plt.ylabel('Jumlah Kendaraan')
plt.xlabel('Tipe Bahan Bakar');


# Tipe bahan bakar dari kendaraan bekas paling banyak yakni gas 

# **Pie Chart Kondisi Kendaraan**

# In[ ]:


vehicles_new['condition'].value_counts().plot.pie(figsize=(6, 6), autopct='%.2f')
plt.title("Kondisi Kendaraan");


# 66,4% kendaraan bekas yang akan dijual memiliki kondisi yang sangat baik

# ** Histogram Tahun Mobil**

# In[ ]:


vehicles_new['year'].describe()


# In[ ]:


year=vehicles_new['year'][vehicles_new['year']>1920]
year


# In[ ]:


year.hist(bins=15,color='hotpink');
plt.title("Histogram Tahun Kendaraan")
plt.ylabel('Jumlah Kendaraan')
plt.xlabel('Tahun Kendaraan')


# Kendaraan bekas paling banyak berada pada interval tahun 2000-2020

# **Barplot Harga Kendaraan**

# In[ ]:


vehicles_new['price'].describe()


# In[ ]:


price=vehicles_new['price'][vehicles_new['price']>3.295000e+03]
price


# In[ ]:


vehicles_new.transmission.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5), color='crimson')
plt.title("Transmisi Kendaraan")
plt.ylabel('Jumlah Kendaraan')
plt.xlabel('Jenis Transmisi');


# Dari data kendaraan bekas yang akan dijual, kendaraan dengan transmisi outomatic memiliki jumlah lebih banyak dibanding lainnya.

# In[ ]:


f=plt.figure(figsize=(20,5))
f.add_subplot(1,3,1)
sns.distplot(numerical['lat'])
f.add_subplot(1,3,2)
sns.distplot(numerical['long'])


# **Analisis Bivariat **

# **Analisis Korelasi**
# 
# beberapa informasi yang dapat diambil dari heatmap:
# * Harga kendaraan bekas memiliki korelasi paling tinggi dengan tahun kendaraan bekas dibanding lainnya
# * Kode negara dengan cuaca memiliki korelasi yang sangat kuat dari seluruh variabel yang ada

# In[ ]:


corr = vehicles_new.corr(method = 'spearman')
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,5))
fig.set_size_inches(15,10)
cmap=sns.cm.rocket_r
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True, cmap=cmap)


# **Boxplot Harga Kendaraan Bekas dengan Transmisi, Bahan Bakar, dan Kondisi**

# In[ ]:


f=plt.figure(figsize=(10,10))
tp=vehicles_new[['price','year']]
tp=tp[tp['price']<150000]
f.add_subplot(3,1,1)
g=sns.boxplot(y=tp['price'], x=vehicles_new['transmission'])
f.add_subplot(3,1,2)
g=sns.boxplot(y=tp['price'], x=vehicles_new['fuel'])
f.add_subplot(3,1,3)
g=sns.boxplot(y=tp['price'], x=vehicles_new['condition'])
plt.xticks(rotation=60)


# Terdapat data outlier pada boxplot antara harga dengan transmisi, bahan bakar dan kondisi kendaraan bekas

# **Timeseries Plot**

# In[ ]:


plt.figure(figsize=(20,10))
tp=vehicles_new[['price','year']]
tp=vehicles_new[tp['year']>=1900]
tp=tp[tp['price']<150000]
p_y=tp.groupby('year').mean()
p_y.reset_index(level=0, inplace=True)
plt.plot(p_y['year'].astype('int'),p_y['price'],color='coral')
plt.xticks(rotation=90,fontsize=10)
plt.show()


# Semakin baru tahun kendaraan tidak menentukan tinggi dan rendahnya harga kendaraan tersebut. Dalam pertengahan interval tahun 1990 hingga 1920 terjadi kenaikan harga mobil yang signifikan bahkan menyentuh harga tertinggi sepanjang tahun 1900 hingga 2020. Dalam interval tahun 2000 hingga 2020 mulai terjadi kenaikan secara bertahap harga mobil hingga didapat harga mobil tertinggi pada tahun 2020. 

# In[ ]:


g = sns.lmplot('long','lat', vehicles_new, hue="type", fit_reg=False);


# Persebaran data kendaraan bekas tipe sedan lebih merata diseluruh latitude. Untuk persebaran data coupe berada di beberapa titik namun cenderung terpusat.

# In[ ]:


g = sns.lmplot('year',"price", tp);


# In[ ]:




