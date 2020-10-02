#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------------------------
# Copy and edit from 
# https://www.kaggle.com/anggriyulio/cleaning-data
# 
# Metode extract data latitide dan longitude diubah menjadi
#     
#     str = row.strip('http://maps.google.co.id/?q=').strip('-')
# 
# Karena jika menggunakan
# 
#     str = re.search('(-?([0-9]{1}|[0-9]0|[1-8]{1,2}).[0-9]{1,6},(-?(1[0-8]{1,2}|9[1-9]{1}).[1-9]{1,6}))', row)
#     
# data latitude/longitude dengan contoh pola data 98.017183 (angka 0 dibelakang koma) menjadi hilang (NaN)
# 
# ------------------------------------------------------------------------------------------------

# ### Masalah
# - Format data tidak konsisten untuk setiap row, seperti penulisan strip (-) untuk data yang kosong.
# - Nama kota perlu di bersihkan kembali
# - Koordinat latitude longitude sebaiknya dipisah untuk mempermudah pengolahan data lebih lanjut

# #### Memuat library dan membuat fungsi yang dibutuhkan

# In[ ]:


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:




data = pd.read_csv('../input/list-faskes-bpjs-indonesia/Data Faskes BPJS 2019.csv', header=0)

def namaKota(row):
    str = row.replace('Kode Faskes dan Alamat Rumah Sakit BPJS di ','')
    return str


def remSpace(row):
    str = " ".join(row.split())
    return str

def searchLatLong(row):
    str = re.search('(-?([0-9]{1}|[0-9]0|[1-8]{1,2}).[0-9]{1,6},(-?(1[0-8]{1,2}|9[1-9]{1}).[1-9]{1,6}))', row)
    if str:
        return str.group()
    return np.NaN

def searchLatLongV2(row):
    str = row.strip('http://maps.google.co.id/?q=').strip('-')
    if str:
        return str
    return np.NaN
        
def valid_latitude(row):
    if float(row) in range(-90.0, 90.0):
        return row
    return np.NaN


def valid_longitude(row):
    return row


# ####  Menghapus karakter \t \n atau spasi yang berlebih

# In[ ]:


data['TelpFaskes'] = data['TelpFaskes'].apply(remSpace)
data['NamaFaskes'] = data['NamaFaskes'].apply(remSpace)
data['KotaKab'] = data['KotaKab'].apply(remSpace)


# #### Membersihkan nama Kota/Kab

# In[ ]:


data['KotaKab'] = data['KotaKab'].apply(namaKota)


# #### Memisahkan latitude longitude dari url dan menambah kolom khusus

# In[ ]:


data['LatLongFaskes'] = data['LatLongFaskes'].apply(searchLatLongV2)

lat = []
lon = []

for row in data['LatLongFaskes']:
    try:
        latitude = float(row.split(',')[0])
        longitude = float(row.split(',')[1])
        if (-90.0 <= latitude <= 90.0):
            lat.append(latitude)
        else:
            lat.append(np.NaN)
        
        if (-180.0 <= longitude <= 180.0):
            lon.append(longitude)
        else:
            lon.append(np.NaN)
    except:
        lat.append(np.NaN)
        lon.append(np.NaN)

data['Latitude'] = lat
data['Longitude'] = lon
data = data.drop("LatLongFaskes", axis=1)


# #### Mengatasi format data yang tidak konsisten

# In[ ]:


data = data.astype(str)
data = data.applymap(lambda x: re.sub(r'^-$', str(np.NaN), x))
data.to_csv('Data Faskes BPJS 2019-clean_data.csv')


# In[ ]:


dataBpjsProv = pd.DataFrame(data.groupby(['Provinsi'])['KodeFaskes'].count())


# Data Populasi didapat dari website [BPS](https://www.bps.go.id/statictable/2009/02/20/1267/penduduk-indonesia-menurut-provinsi-1971-1980-1990-1995-2000-dan-2010.html)

# In[ ]:


dataPupulasi = pd.read_csv('../input/pupulasi-indonesia-2010/indo_12_1.csv', header=0, index_col = 0)
dataPupulasi = dataPupulasi[['Pupulasi2010']]


# In[ ]:


dataPupulasi.head()


# In[ ]:


dataBpjsProv['JumlahFaskes'] = dataBpjsProv['KodeFaskes']
dataBpjsProv = dataBpjsProv.drop(columns=['KodeFaskes'])
dataBpjsProv.head()


# Cari Ratio Perbandingan Jumlah Penduduk dan Jumlah Faskes, 

# In[ ]:


DataFaskesPupulasi = pd.concat([dataBpjsProv,dataPupulasi], axis = 1,sort=True)
DataFaskesPupulasi['RatioPendudukFaskes'] = DataFaskesPupulasi['Pupulasi2010']/DataFaskesPupulasi['JumlahFaskes']


# In[ ]:


DataFaskesPupulasi.sort_values('RatioPendudukFaskes')

