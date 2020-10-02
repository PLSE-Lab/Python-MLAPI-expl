#!/usr/bin/env python
# coding: utf-8

# **Analisa korelasi antara emisi gas buang terhadap pertumbuhan GDP Indonesia**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/Indicators.csv')
data.shape


# Data dari wold development indicator adalah dataset yang besar, memiliki sekitar 5 jutaan data dari berbagai negara.
# mari kita melihat **10 data teratas** dari dataset ini:

# In[ ]:


data.head(10)


# * **CountryName** adalah kolom untuk nama negara, 
# * **CountryCode** adalah kolom untuk kode negara, 
# * **IndicatorName** adalah nama dari inidikator 
# * **IndicatorCode** adalah kode dari indikator, 
# * **Year** adalah tahun data dan
# * **Value** adalah nilai dari indikator tersebut

# Mari kita melihat  negara2 yang terdaftar dalam dataset ini:

# In[ ]:


countries = data.CountryName.unique().tolist()
countries


# Jumlah total negara yang terdaftar:

# In[ ]:


len(countries)


# Jumlah tahun dalam data set kita:

# In[ ]:


years = data['Year'].unique().tolist()
len(years)


# Jumlah indikator dalam data set kita:

# In[ ]:


indicators = data['IndicatorName'].unique().tolist()
len(indicators)


# Rentang tahun dalam dataset kita:

# In[ ]:


print(min(years)," sampai ",max(years))


# ok, setelah melakukan analisa dari dataset ini, mari kita menseleksi data mengenai negara kita "Indonesia"

# In[ ]:


mask = data['CountryName'].str.contains("Indonesia")
data_indonesia = data[mask]
data_indonesia.head(10)


# jumlah dari data yang terkait dengan indonesia:

# In[ ]:


len(data_indonesia)


# dari sekian banyak data ini, mari kita menseleksi data mengenai emisi gas buang CO2 di Indonesia dari tahun ke tahun:
# 

# In[ ]:


mask2 = data['IndicatorName'].str.contains('CO2 emissions \(metric') 
data_co2_indonesia = data[mask & mask2]
data_co2_indonesia.head()


# dari perintah head kita dapat melihat bahwa tahun awal data ini adalah pada tahun 1960, untuk lebih jelas lagi, mari kita tampilkan data minimum dan maximum tahun data

# In[ ]:


print(min(data_co2_indonesia['Year'])," sampai ",max(data_co2_indonesia['Year']))


# oops, dan ternyata data emisi gas buang kita hanya sampai pada tahun 2011 saja.

# OK setelah menampilkan data emisi gas buang, sekarang mari kita tampilkan data pertumbuhan GDP indonesia:
# 

# In[ ]:


mask3 = data['IndicatorName'].str.contains('GDP per capita \(constant 2005') 
data_gdp_indonesia = data[mask & mask3]
data_gdp_indonesia.head()


# ok seperti yang kita lakukan terhadap data gas buang, mari kita analisa rentang tahun pada data pertumbuhan GDP:

# In[ ]:


print(min(data_gdp_indonesia['Year'])," sampai ",max(data_gdp_indonesia['Year']))


# ok nampaknya kita memiliki perbedaan jumlah data indikator antara data pertumbuhan GDP *(tahun 1960 sampai tahun 2014)* dan data emisi gas buang *(tahun 1960 sampai 2011)*

# sekarang mari kita proyeksikan pertumbuhan emisi gas buang di indonesia dengan menggunakan** bar plot**:

# In[ ]:


# get the years
years = data_co2_indonesia['Year'].values
# get the values 
co2 = data_co2_indonesia['Value'].values

# create
plt.bar(years,co2)
plt.show()


# kemudian, mari kita proyesikan pertumbuhan GDP kita dari tahun 1960 dengan menggunakan** line plot**

# In[ ]:


# switch to a line plot
plt.plot(data_gdp_indonesia['Year'].values, data_gdp_indonesia['Value'].values)

# Label the axes
plt.xlabel('Tahun')
plt.ylabel(data_gdp_indonesia['IndicatorName'].iloc[0])

#label the figure
plt.title('GDP Per Kapita Indonesia')

# to make more honest, start they y axis at 0
plt.axis([1959, 2011,0,47000])

plt.show()


# setelah membandingkan kedua grafik diatas, kita melihat adanya pertumbuhan dari 2 indikator tersebut, sekarang mari kita relasikan pertumbuhan indikator tersebut dengan menggunakan** scatter plot**

# jika kita ingin menggunakan scatter plot, kita harus memastikan terlebih dahulu bahwa kedua dimensi memiliki jumlah yang sama, kita memiliki data dari tahun 1960 sampai 2014 untuk GDP dan dari tahun 1960 sampai 2011 untuk data emisi gas buang, untuk itu kita perlu melakukan normalisasi data dengan membuang data GDP diatas tahun 2011
# 

# In[ ]:


data_gdp_indonesia_norm = data_gdp_indonesia[data_gdp_indonesia['Year'] < 2012]
print(len(data_co2_indonesia))
print(len(data_gdp_indonesia_norm))


# nah sekarang kita memiliki jumlah data yang sama, sehingga memungkinkan . untuk membuat scatter plot

# In[ ]:


fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_title('Emisi gas buang vs. GDP per kapita',fontsize=10)
axis.set_xlabel(data_gdp_indonesia_norm['IndicatorName'].iloc[0],fontsize=10)
axis.set_ylabel(data_co2_indonesia['IndicatorName'].iloc[50],fontsize=9)

X = data_gdp_indonesia_norm['Value']
Y = data_co2_indonesia['Value']

axis.scatter(X, Y)
plt.show()


# wow, kita dapat menyimpulkan satu hipotesis dari grafik diatas: 
# 
# **pertumbuhan jumlah gas buang berbanding lurus dengan pertumbuhan GDP per kapita di indonesia**

# sekarang, mari kita menghitung jumalh korelasi dari kedua indikator tersebut:

# In[ ]:


np.corrcoef(data_gdp_indonesia_norm['Value'],data_co2_indonesia['Value'])


# kita mendapatkan nilai korelasi sebesar** 0.987 **yang mana merupakan korelasi yang kuat andara kedua indikator tersebut

# In[ ]:




