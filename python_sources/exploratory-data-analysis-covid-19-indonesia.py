#!/usr/bin/env python
# coding: utf-8

# # Deskripsi Data

# Dataset ini berisi 893 informasi pasien dengan 11 variabel berbeda sampai dengan tanggal 31 maret 2020
# * patient_id = ID pasien
# * gender = jenis kelamin pasien
# * age = usia pasien
# * nationality = kewarganegaraan pasien
# * province = wilayah / provinsi pasien
# * current_state = kota pasien
# * contacted_with = dihubungi / terinfeksi oleh nomor id pasien
# * confirm_date = tanggal konfirmasi
# * release_date = tanggal keluarnya
# * deceased_date = tanggal kematian
# * hospital = lokasi rawat inap

# **Catatan**: Struktur kumpulan data, grafik, dan observasi dapat bervariasi di suatu tempat karena dataset ini terus diperbarui. Jadi, pengamatan tekstual statis mungkin sedikit berbeda dari representasi grafis. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


pasien = pd.read_csv('../input/indonesia-coronavirus-cases/patient.csv')
cases = pd.read_csv('../input/indonesia-coronavirus-cases/cases.csv')


# In[ ]:


pasien.head()


# In[ ]:


pasien.describe()


# In[ ]:


pasien.info()


# In[ ]:


pasien['current_state'].value_counts()


# Dari 893 pasien terinveksi covid-19, 8 pasien sudah sembuh, 15 pasien meninggal, sedangkan 143 pasien sudah diisolasi. 

# In[ ]:


male = pasien.loc[pasien['gender']=='male','age'].mean()
female = pasien.loc[pasien['gender']=='female','age'].mean()
print('Distribusi rata-rata umur pasien laki-laki: %i' %male, 'tahun')
print('Distribusi rata-rata umur pasien perempuan: %i' %female, 'tahun')


# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
plt.title("distribusi pasien berdasarkan umur")
sns.kdeplot(data=pasien['age'], shade=True)


# In[ ]:


sns.FacetGrid(pasien, hue="current_state", size=5)  .map(sns.distplot, "age")  .add_legend()
plt.title('PDF with age')
plt.show()


# **Interpretasi:**
# 1. Distribusi pasien yang meninggal berada diusia 50 - 70 tahun
# 2. Distribusi pasien yang sembuh berada diusia 20 - 60 tahun 

# ## Countplot

# In[ ]:


pasien.current_state.value_counts().plot.bar().grid()


# In[ ]:


sns.countplot(x='gender', hue='current_state', data=pasien)


# **Interpretasi:**
# 
# Sebagian besar yang terkena covid-19 berjenis kelamin laki-laki

# # Barplot

# In[ ]:


pasien.province.value_counts().plot.bar()


# **Interpretasi:**
# 
# Sebagian besar orang yang terinfeksi covid-19 berada di DKI Jakarta

# In[ ]:


plt.figure(figsize=(15,5))
pasien.contacted_with.value_counts().plot.bar()


# **Interpretasi:**
# 
# Sebagian besar pasien terinfeksi oleh pasien dengan ID 1.0

# In[ ]:


plt.figure(figsize=(15,5))
pasien.confirmed_date.value_counts().plot.bar()


# **Interpretasi:**
# 
# Jumlah kasus postif covid-19 per hari terbanyak terdapat di tanggal 24-Mar-2020 (106 kasus)

# # Scatter Plot

# In[ ]:


sns.FacetGrid(pasien, hue = 'current_state', size = 7).map(plt.scatter, 'age', 'province').add_legend()
plt.title('Persebaran pasien covid-19')
plt.show()


# In[ ]:


sns.FacetGrid(pasien, hue = 'current_state', size = 7).map(plt.scatter, 'age', 'hospital').add_legend()
plt.title('Persebaran pasien covid-19 di rumah sakit')
plt.show()

