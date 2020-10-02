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


df = pd.read_excel("/kaggle/input/used_car_data.xlsx")


# In[ ]:


df


# **Memisahkan merek dari nama**

# In[ ]:


merek = []
for name in df['Name']:
    merek.append(name.split(" ")[0].upper())
    
df['Merek'] = merek


# In[ ]:


df


# In[ ]:


# Fungsi untuk menyamakan satuan menjadi km/kg
# Menggunakan asumsi 1 liter minyak = 0,8 kg
def samakan_satuan(mileage):
    if pd.notna(mileage):
        satuan= mileage.split(" ")[1]
        jarak = float(mileage.split(" ")[0])
        if satuan == "kmpl":
            jarak = jarak / 0.8
            return jarak
        return jarak


# In[ ]:


# Membuat suatu kolom yang memuat jarak tempuh bahan bakar yang telah disamakan satuannya menjadi km/kg 
df['Mileage'] = df.apply(lambda row: samakan_satuan(row['Mileage']), axis=1)


# In[ ]:


# Fungsi untuk membuang satuan CC
def buang_satuan_CC(engine):
    if pd.notna(engine) and engine != 'null CC' :
        return float(engine.split(" ")[0])
    else:
        return np.nan


# In[ ]:


# Membuang satuan CC pada kolom Engine
df['Engine'] = df.apply(lambda row: buang_satuan_CC(row['Engine']), axis=1)


# In[ ]:


# Fungsi untuk membuang satuan bhp
def buang_satuan_bhp(power):
    if pd.notna(power) and power != 'null bhp' :
        return float(power.split(" ")[0])
    else:
        return np.nan


# In[ ]:


# Membuang satuan bhp pada kolom Power
df['Power'] = df.apply(lambda row: buang_satuan_bhp(row['Power']), axis=1)


# In[ ]:


df


# # **NOMOR 1**

# In[ ]:


df['Price'].groupby(df['Merek']).count()


# # **NOMOR 2**

# In[ ]:


df['Location'].groupby(df['Location']).count().sort_values(ascending=False)


# Dapat dilihat bahwa Kota yang memiliki mobil bekas paling banyak adalah Kota Mumbai dengan jumlah 790 mobil

# # **NOMOR 3**

# * Mencari penyebaran menggunakan Histogram dan Kernel Density Estimate

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.distplot(df['Year'])


# Berdasarkan grafik yang didapat, distribusi tahun edisi tersebut berupa negatively skewed (skewed left)

# # **NOMOR 4**

# In[ ]:


df[df['Kilometers_Driven'] < 100000].count()


# Total mobil yang memiliki pemakaian di bawah 100.000 km ada sebanyak 5470 unit. Pada Mileage, Engine, Power, dan Seats, nilainya berbeda karena pada kolom tersebut terdapat nilai NaN.

# # **NOMOR 5**

# In[ ]:


category = []
for km in df['Kilometers_Driven']:
    if km >= df['Kilometers_Driven'].median():
        category.append("Tinggi")
    else:
        category.append("Rendah")
        
df['Category'] = category


# In[ ]:


df


# Menurut saya batas kilometer total jarak pemakaian bisa dikategorikan sebagai rendah atau tinggi dapat diambil dari median datanya, karena median tidak dipengaruhi outliernya. Dalam dataset ini nilai median Kilometers_Driven nya adalah 53000. Sehingga nilai Km di atas atau sama dengan 53000 dikategorikan "Tinggi", sedangkan sisanya dikategorikan "Rendah".

# # **NOMOR 6**

# * Mencari outlier dengan menggunakan box plot

# In[ ]:


sns.set(rc={'figure.figsize':(16,8)})
sns.boxplot(x=df['Kilometers_Driven'])


# * Mencari outlier dengan menggunakan Scatter Plot

# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['Year'], df['Kilometers_Driven'])
ax.set_xlabel('Year')
ax.set_ylabel('kilometres Driven')
plt.show()


# In[ ]:


data = sorted(df['Kilometers_Driven'])
quantile1, quantile3 = np.percentile(data, [25, 75])

iqr = quantile3 - quantile1
lower_bound = quantile1 - (1.5 * iqr)
upper_bound = quantile3 + (1.5 * iqr)

outliers = sorted(df[(df['Kilometers_Driven'] > upper_bound) | (df['Kilometers_Driven'] < lower_bound)].Kilometers_Driven)

for i in outliers:
    print(i,end=", ")

print()
print()
print("Jumlah outlier :",len(outliers))


# Dapat terlihat dengan jelas pada visualisasi data menggunakan box plot dan scatter plot bahwa terdapat suatu data yang sangat berbeda / pencilan dalam penyebaran datanya. Berdasarkan data yang ditunjukkan nilai yang lebih besar sama dengan 131765 merupakan outlier dengan jumlah outlier adalah sebanyak 202 buah.

# # **NOMOR 7**

# In[ ]:


df['Year'].corr(df['Kilometers_Driven'])


# Dapat dilihat bahwa nilai dari korelasi antara tahun pembuatan dan juga jarak yang ditempuh adalah -0.16937, yang berarti relasi antara tahun pembuatan dan jarak ditempuh adalah weakly negative relationship atau dapat dibilang tahun pembuatan mobil hampir tidak berkaitan dengan total jarak yang ditempuh.

# # **NOMOR 8**

# In[ ]:


df[(df['Owner_Type'] == 'Third') | (df['Owner_Type'] == 'Fourth & Above')].count()


# Sehingga mobil yang merupakan kepemilikan ketiga atau lebih terdapat 122 mobil. Terdapat perbedaan nilai pada Engine, Power, Seats karena terdapatnya nilai NaN pada kolom tersebut.

# # **NOMOR 9**

# In[ ]:


# Membuat DataFrame yang menyortir rata-rata jarak yang ditempuh dari setiap bahan bakar yang ada 
df.groupby('Fuel_Type')['Mileage'].mean().sort_values(ascending=False)


# Untuk mengetahui bahan bakar yang paling hemat, diperlukannya persamaan dalah satuan jarak yang ditempuh (terdapat 2 jenis satuan yaitu km/liter dan km/kg). Setelah disamakan satuannya, dibuatlah DataFrame yang menampilkan yang menampilkan rata-rata jarak yang ditempuh dalam km/kg. Berdasarkan DaraFrame yang telah dibuat, dapat terlihat jelas bahwa tipe bahan bakar yang paling hemat adalah CNG.

# # **NOMOR 10**

# **Membuang nilai null**

# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop(columns = ['Category', 'Merek'])


# In[ ]:


df = df.dropna(subset = ['Mileage', 'Engine', 'Power', 'Seats'])


# In[ ]:


df.isnull().sum()


# **Untuk mengecek hubungan antar kolom terhadap Price, kita akan terlebih dahulu membagi proses nya ke dalam 2 tahap, yaitu kolom-kolom Numerical terhadap Price dan juga kolom-kolom Categorical terhadap Price**
# 
# **Nb : Untuk Numerical terhadap Price, asumsi -0.3 sampai 0.3 menyatakan no correlation**
# 
# 
# **Nb : Untuk Categorical, asumsi alpha = 5%**

# 1. **Numericals terhadap Price**

# * **Hubungan Year dan Price**

# In[ ]:


df['Price'].corr(df['Year'])


# Karena nilai dari correlation coefficient diantara -0.3 dan 0.3 maka dapat disimpulkan bahwa Year tidak memiliki hubungan terhadap Price

# * **Hubungan Kilometers_Driven dan Price**

# In[ ]:


df['Price'].corr(df['Kilometers_Driven'])


# Karena nilai dari correlation coefficient diantara -0.3 dan 0.3 maka dapat disimpulkan bahwa Kilometers_Driven tidak memiliki hubungan terhadap Price

# In[ ]:


df['Price'].corr(df['Mileage'])


# Karena nilai dari correlation coefficient lebih kecil dari -0.3 maka dapat disimpulkan bahwa Mileage memiliki hubungan terhadap Price yang berlawanan arah.

# In[ ]:


df['Price'].corr(df['Engine'])


# Karena nilai dari correlation coefficient lebih besar dari 0.3 maka dapat disimpulkan bahwa Engine memiliki hubungan terhadap Price.

# In[ ]:


df['Price'].corr(df['Power'])


# Karena nilai dari correlation coefficient lebih besar dari 0.3 maka dapat disimpulkan bahwa Power memiliki hubungan terhadap Price.

# In[ ]:


df['Price'].corr(df['Seats'])


# Karena nilai dari correlation coefficient diantara -0.3 dan 0.3 maka dapat disimpulkan bahwa Kilometers_Driven tidak memiliki hubungan terhadap Price

# 2. **Categorical terhadap Price**

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA


# In[ ]:


maov = MANOVA.from_formula('Name+Location+Fuel_Type+Transmission+Owner_Type~Price', data = df)
print(maov.mv_test())


# Untuk kasus ini, kita dapat menggunakan Wilks' Lambda untuk mengecek korelasi terhadap Price. Dapat dilihat bahwa pada tabel Price, p-value (Pr > F) dari Wilks' Lambda memiliki nilai yang lebih kecil dari 5% atau 0.05 sehingga dapat dibilang kolom-kolom tersebut memiliki korelasi terhadap Price.
# 
# Oleh karena itu, kita akan mengecek korelasi tiap kolom terhadap Price. Untuk pengecekan tiap kolom, alpha yang akan digunakan adalah 5/5 % = 1% karena terdapat 5 kolom yang akan dicek.

# * **Korelasi antara Name dan Price**

# In[ ]:


reg = ols('Price ~ Name', data = df).fit()
aov = sm.stats.anova_lm(reg, type = 2)
print(aov)


# Dapat dilihat bahwa p-value (PR(>F)) antara Name dan Price lebih kecil dari 1% atau 0.01 maka dapat disimpulkan bahwa Name memiliki korelasi terhadap Price

# * **Korelasi antara Location dan Price**

# In[ ]:


reg = ols('Price ~ Location', data = df).fit()
aov = sm.stats.anova_lm(reg, type = 2)
print(aov)


# Dapat dilihat bahwa p-value (PR(>F)) antara Location dan Price lebih kecil dari 1% atau 0.01 maka dapat disimpulkan bahwa Location memiliki korelasi terhadap Price

# * **Korelasi antara Fuel_Type dan Price**

# In[ ]:


reg = ols('Price ~ Fuel_Type', data = df).fit()
aov = sm.stats.anova_lm(reg, type = 2)
print(aov)


# Dapat dilihat bahwa p-value (PR(>F)) antara Fuel_Type dan Price lebih kecil dari 1% atau 0.01 maka dapat disimpulkan bahwa Fuel_Type memiliki korelasi terhadap Price

# * **Korelasi antara Transmission dan Price**

# In[ ]:


reg = ols('Price ~ Transmission', data = df).fit()
aov = sm.stats.anova_lm(reg, type = 2)
print(aov)


# Dapat dilihat bahwa p-value (PR(>F)) antara Transmission dan Price lebih kecil dari 1% atau 0.01 maka dapat disimpulkan bahwa Transmission memiliki korelasi terhadap Price

# * **Korelasi antara Owner_Type dan Price**

# In[ ]:


reg = ols('Price ~ Owner_Type', data = df).fit()
aov = sm.stats.anova_lm(reg, type = 2)
print(aov)


# Dapat dilihat bahwa p-value (PR(>F)) antara Owner_Type dan Price lebih kecil dari 1% atau 0.01 maka dapat disimpulkan bahwa Owner_Type memiliki korelasi terhadap Price

# **Kesimpulan : Faktor-faktor yang mempengaruhi Price adalah Mileage, Engine, Power, Name, Location, Fuel_Type, Transmission, dan Owner_Type**
