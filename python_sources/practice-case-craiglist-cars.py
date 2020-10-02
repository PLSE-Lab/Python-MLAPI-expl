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


# 

# # Practice Case For EDA Case 1 : Craiglist Carsstruck
# Nama : Topik Zulkarnain
# ***

# In[ ]:


data = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv')


# ## Pemahaman Data
# Pada proses ini akan dieksplor terlebih dahulu mengenai karakteristik data 

# In[ ]:


data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.shape


# Hasil dari info() ataupun data.shape menunjukkan bahwa terdapat 26 kolom dengan jumlah entri tiap kolom berjumlah 1,723,064 . Berikutnya akan dicek missing value pada tiap kolom, lalu dicari persentase missing value agar dapat diberi keputusan apakah kolom tersebut akan di drop atau tidak. Untuk batas missing value yang digunakan adalah 60% jika lebih maka kolom akan di drop, jika tidak maka akan dilakukan metode statistik deskriptif untuk melengkapi missing value.

# In[ ]:


((data.isnull().sum())/(1723064))*100


# berdasarkan output diatas maka data vin dan data size harus didrop karna mengandung missing value lebih dari 60% sehingga kedua data tersebut tidak berguna

# In[ ]:


datadrop = data.drop(columns =['vin', 'size'])
datadrop.isnull().sum()


# In[ ]:


# Data yang mengandung missing values namun masih dapat digunakan akan diisi dengan menggunakan nilai rata-rata, nilai median, atau modus
numerical = ['odometer','weather' ]
categorical = ['url', 'city', 'year', 'manufacturer', 'make', 'cylinders', 'fuel','title_status','drive', 'type','paint_color', 'image_url','county_fips','county_name',"state_fips","condition", "transmission", "state_code" ]


# In[ ]:


for x in numerical :
    datadrop[x] = datadrop[x].fillna(datadrop[x].mean())


# In[ ]:


for x in categorical :
    datadrop[x] = datadrop[x].fillna(datadrop[x].mode().values[0])


# In[ ]:


datadrop.isnull().sum()


# ## Missing value sudah diinput dengan menggunakan nilai rata-rata pada data numerik, dan nilai yang paling sering muncul pada data kategori

#  # Visualisasi Data
#  ***

# In[ ]:


dataclean = datadrop
dataclean.head()


# In[ ]:


import numpy as np
datanumerical = dataclean[['weather', 'lat', 'long','price','odometer']]
datacategorical = dataclean[['url', 'city', 'year', 'manufacturer', 'make', 'cylinders', 'fuel','title_status','drive', 'type','paint_color', 'image_url','county_fips','county_name',"state_fips","condition", "transmission", "state_code"]]
datanumerical_array = np.array(datanumerical)


# ### Visualisasi data dengan menggunakan boxplot

# In[ ]:


import matplotlib.pyplot as plt
colors = ['blue', 'red' , 'yellow', 'cyan', "green"]
fig = plt.figure(figsize = (10,10))
axes = 300
bp = plt.boxplot(datanumerical_array, 
                 patch_artist=True,
                 notch=True)

for i in range(len(bp['boxes'])):
    
    bp['boxes'][i].set(facecolor=colors[i])
    
    bp['caps'][2*i + 1].set(color=colors[i])
    
plt.xticks([1, 2, 3, 4, 5], ['weather','lat','long','odometer','price'])

plt.show()


# ### Visualisasi hubungan antar feature dengan menggunakan heatmap

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#create correlation with hitmap

#create correlation
corr = dataclean.corr(method = 'spearman')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(70,12)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)


# #### Visualisasi Sebaran Harga Mobil 30 Tahun Terakhir
# Dari scatter plot dan statistik korelasi dapat disimpulkan bahwa harga mobil tidak dipengaruhi oleh waktu

# In[ ]:


pricecar = dataclean[['year','price']]
pricechange = pricecar[pricecar['year']>=1990]
plt.scatter(pricechange['year'], pricechange['price'])
plt.title('year vs price ')
plt.xlabel('year')
plt.ylabel('Price Billion $')
plt.show()


# In[ ]:


pricechange['price'].corr(pricechange['year'])


# ### Rata-rata harga tiap brand

# In[ ]:


make_price =  dataclean[["make","price"]]
average_make_price = make_price.groupby("make").mean()
average_make_price.head()


# ### Rata-rata harga mobil berdasarkan manufakturnya

# In[ ]:


highprice = dataclean[["manufacturer","price"]]
highprice_manufacture = highprice.groupby("manufacturer").mean()
highprice_manufacture.head()


# ### Visualisasi Grafik Mobil Termahal Berdasarkan Manufaktur

# In[ ]:


price_car = dataclean[["manufacturer", "price"]]
price_car_clean = price_car[pricecar['price']>=5000000]
price_car_bar = price_car_clean.groupby("manufacturer").max().sort_values("price").plot(kind='bar').plot(figsize = (20,20))


# #### Visualisasi kondisi mobil 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
labels = pd.DataFrame(dataclean["condition"].value_counts())
plt.pie(dataclean["condition"].value_counts(), labels = labels.index, autopct='%.2f')
plt.show()
plt.figure(figsize=(10,10))
labels = pd.DataFrame(dataclean["fuel"].value_counts())
plt.pie(dataclean["fuel"].value_counts(), labels = labels.index, autopct='%.2f')
plt.show()
plt.figure(figsize=(10,10))
labels = pd.DataFrame(dataclean["transmission"].value_counts())
plt.pie(dataclean["transmission"].value_counts(), labels = labels.index, autopct='%.2f')
plt.show()


# In[ ]:





# ### Analisis Korelasi

# In[ ]:


dataclean.corr(method = 'pearson').style.background_gradient().set_precision(2)


# In[ ]:


dataclean.corr(method = 'spearman').style.background_gradient().set_precision(2)


# In[ ]:




