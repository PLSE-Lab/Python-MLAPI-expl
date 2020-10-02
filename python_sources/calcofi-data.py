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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats

#data'yi data_bottle degiskenine atayip, pd.read_csv ile okuduk.
data_bottle = pd.read_csv("/kaggle/input/calcofi/bottle.csv")
data_bottle.head()


# In[ ]:


len(data_bottle.columns)


# In[ ]:


#sutunlardaki bos/non degerlerini gormek icin. 
#buna gore hangi sutunlari alip anlamlandirabilecegimize karar vermek kolaylasir.
data_bottle.isna() 


# #### Depthm = Depth in meters
# #### T_degCWater = temperature in degree Celsius
# #### Salnty = Salinity in g of salt per kg of water (g/kg)

# In[ ]:


deep = data_bottle[['Depthm']]
temperature = data_bottle[['T_degC']]
salt = data_bottle[['Salnty']]


# In[ ]:


#sectigimiz 3 sutunu concat ile ardarda eklemek istiyoruz.
new_data = pd.concat([deep,temperature,salt], axis = 1, ignore_index=True)
new_data


# In[ ]:


#Kalici bir degisiklik ile 0,1,2 olarak cikan sutun numaralarini istedigimizle yeniden adlandiralim
new_data.rename(columns={0:"Derinlik", 1: "Sicaklik",2: "Tuzluluk"}, inplace=True)
new_data.head()


# In[ ]:


#sectigimiz sutunlardan olusan yeni data kumesi hakkinda:
#sayi, ortalama, standart sapma, ceyrekler,min, max degerler hakkinda bilgiye ulasmak icin:
new_data.describe().T


# In[ ]:


#kac tane non degeri oldugunu gormek icin
new_data.isna().sum()


# In[ ]:


#non degerlerini kaldirmak icin
new_data.dropna(inplace=True)
new_data


# In[ ]:


#non degeri kaldi mi diye kontrol
new_data.isna().sum()


# In[ ]:


#non degerleri kaldirilinca gercege daha yakin sonuclar icin tekrar bakalim
new_data.describe().T


# In[ ]:


#Mod ve Medyan degerlerini de ayri olarak gormek icin:
median_derinlik = np.median(new_data.Derinlik)
print('Median Derinlik:',median_derinlik)
mod_derinlik = stats.mode(new_data.Derinlik)
print('Mod Derinlik:',mod_derinlik)

median_sicaklik = np.median(new_data.Sicaklik)
print('Median Sicaklik:',median_sicaklik)
mod_sicaklik = stats.mode(new_data.Sicaklik)
print('Mod Sicaklik:',mod_sicaklik)

median_tuzluluk = np.median(new_data.Tuzluluk)
print('Median Tuzluluk:',median_tuzluluk)
mod_tuzluluk = stats.mode(new_data.Tuzluluk)
print('Mod Tuzluluk:',mod_tuzluluk)


# In[ ]:


#Range Hesabi:
range_derinlik = np.max(new_data.Derinlik)-np.min(new_data.Derinlik)
print("Derinlik Range: ",range_derinlik) 

range_tuzluluk = np.max(new_data.Tuzluluk)-np.min(new_data.Tuzluluk)
print("Tuzluluk Range: ",range_tuzluluk)

range_sicaklik = np.max(new_data.Sicaklik)-np.min(new_data.Sicaklik)
print("Sicaklik Range: ",range_sicaklik)


# In[ ]:


#Variance Hesabi:
variance_derinlik = np.var(new_data.Derinlik)
print("Derinlik Variance: ", variance_derinlik)

variance_sicaklik = np.var(new_data.Sicaklik)
print("Sicaklik Variance: ", variance_sicaklik)

variance_tuzluluk = np.var(new_data.Tuzluluk)
print("Tuzluluk Variance: ", variance_tuzluluk)


# In[ ]:


#Standart Sapma Hesabi:
ss_derinlik = np.std(new_data.Derinlik)
print("Derinlik Standart Sapma: ", ss_derinlik)

ss_tuzluluk = np.std(new_data.Tuzluluk)
print("Tuzluluk Standart Sapma: ", ss_tuzluluk)

ss_sicaklik = np.std(new_data.Sicaklik)
print("Sicaklik Standart Sapma: ", ss_sicaklik)


# In[ ]:


#Pearson Korelasyon:

import matplotlib.pyplot as plt
plt.style.use("ggplot")
f,ax=plt.subplots(figsize = (8,8))
# corr() is actually pearson correlation
sns.heatmap(new_data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()


# In[ ]:


#Derinlik ve Sicaklik arasinda iki farkli hesaplamayla Pearson Korelasyon iliski durumu:

p1 = new_data.loc[:,["Derinlik","Sicaklik","Tuzluluk"]].corr(method= "pearson")
p2 = new_data.Derinlik.cov(new_data.Sicaklik)/(new_data.Derinlik.std()*new_data.Sicaklik.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation: ')
print(p2)


# In[ ]:


#SPEARMAN icin once siralama rank etme var:

ranked_data = new_data.rank()
spearman_corr = ranked_data.loc[:,["Derinlik","Sicaklik","Tuzluluk"]].corr(method= "pearson")
print("Spearman's correlation: ")
print(spearman_corr)


# In[ ]:


#Derinlik ile Tuzluluk arasi ilisi:

sns.lmplot(x='Derinlik',y='Tuzluluk',data = new_data)


# In[ ]:


#Derinlik ile Tuzluluk arasi ilisi:

sns.lmplot(x='Derinlik',y='Sicaklik',data = new_data)


# In[ ]:




