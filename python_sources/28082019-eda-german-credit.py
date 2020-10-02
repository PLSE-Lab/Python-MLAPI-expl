#!/usr/bin/env python
# coding: utf-8

# ## Context
# The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes. The link to the original dataset can be found below.

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


#import visualization
import matplotlib.pyplot as plt
import seaborn as sns

#import preprocessing
from sklearn import preprocessing


# ### Import Data

# In[ ]:


dfa = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')
df = dfa.drop(columns=['Unnamed: 0'], axis=1)
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# ## Cleansing

# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


#detect percentage of missing value which above 60%
null = df.isnull().sum().sort_values(ascending=False)
print('Percentage of missing value is', round((null[0]/1000)* 100)) #size
print('Percentage of missing value is', round((null[1]/1000)* 100)) #vin
#condition
#percentage missing value of size and vin is more than 60%
#so that size and vin drop from the table


# ### Numerical and Categorical Type

# In[ ]:


category = ['Sex','Job','Housing','Saving accounts',
            'Checking account','Purpose']
numerical  = df.drop(category, axis=1)
categorical = df[category]
numerical.head()


# In[ ]:


categorical.head()


# In[ ]:


#fill value in categorical
for cat in categorical:
    mode = categorical[cat].mode().values[0]
    categorical[cat]=df[cat].fillna(mode)


# In[ ]:


#detect categorical missing value
categorical.isnull().sum().sort_values(ascending=False) 


# In[ ]:


#concat table categorical and numerical
dfinal = pd.concat([categorical,numerical],axis=1)
dfinal.head()


# ## Outlier Detection

# In[ ]:


fig=plt.figure(figsize=(13,12))
axes=330
#put data numerical
for num in numerical:
    axes += 1
    fig.add_subplot(axes)
    #set title of num
    sns.boxplot(data = numerical, x=num) 
plt.show()


# ## Visualization

# ### Bar Plot

# In[ ]:


#filter by full-type=gas
data_sex = dfinal[dfinal['Sex']=='female']
top=data_sex.sort_values('Age',ascending=False).head(5)

plt.figure(figsize=(12,6))

x=range(5)
plt.bar(x,top['Age']/6**9)
plt.xticks(x,top['Purpose'])
plt.xlabel('Purpose')
plt.ylabel('Age')
plt.title('10 Most Fuel Type In Cars')
plt.show()
top


# #### Insight
# Peminjam kredit dengan gender female banyak yang melakukan peminjaman dengan tujuan untuk melakukan pembelian mobil, urusan pendidikan, perbaikan perkakas, dan pembelian radio/TV. Namun untuk tujuan pembelian mobil terdapat perbedaan pada jenis checking account, yaitu little dan moderate

# ### Countplot

# In[ ]:


fig=plt.figure(figsize=(20,12))
axes=230
#put data categorical
for cat in categorical:
    axes += 1
    fig.add_subplot(axes)
    #set title of cat
    sns.countplot(data = categorical, x=cat) 
plt.show()


# #### Fungsi
# Countplot digunakan untuk **menjumlahkan** variabel dependen ke variabel independen
# #### Insight
# 1. Countplot sex menunjukkan bahwa sex=male yang paling banyak mengambil kredit di bank
# 2. Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled). Countplot job menunjukkan bahwa job 
# 3. Coutplot housing menunjukkan bahwa orang yang melakukan kredit rata-rata memiliki kepemilikan rumah pribadi
# 4. Sekitar -800 orang yang mengambil kredit di bank merupakan orang yang masuk kategori little untuk penggolongan dalam saving account yang tidak tiap hari dilakukan
# 5. Sedangkan sekitar -700 peminjam memiliki intensitas daily checking account untuk jenis peminjaman little
# 6. Tujuan paling banyak orang mengambil kredit adalah untuk keperluan vacation/others
# 

# ### Distribution Plot

# In[ ]:


fig=plt.figure(figsize=(13,12))
fig.add_subplot(2,2,1)
sns.distplot(numerical['Credit amount'])
fig.add_subplot(2,2,2)
sns.distplot(numerical['Age'])
fig.add_subplot(2,2,3)
sns.distplot(numerical['Duration'])


# #### Fungsi
# Distribution plot untuk mengetahui **distribusi** kumulatif dari variabel yang dipilih
# #### Insight
# Ketiga variabel tidak memiliki distribusi normal

# ### Violin Plot

# In[ ]:


fig=plt.figure(figsize=(20,12))
axes=230
#put data categorical
for cat in numerical:
    axes += 1
    fig.add_subplot(axes)
    #set title of cat
    sns.violinplot(data = numerical, x=cat) 
plt.show()


# ### Heatmap****

# In[ ]:


#create correlation with hitmap

#create correlation
corr = dfinal.corr(method = 'pearson')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(20,5)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)


# #### Fungsi
# Heatmap digunakan untuk mengetahui korelasi dari masing-masing variabel
# #### Insight
# 1. Pengambil kredit memiliki korelasi yang kuat antara age dan job
# 2. Pengambil kredit memiliki korelasi yang kuat antara credit amount dan age
# 3. Pengambil kredit memiliki korelasi yang kuat antara duration dan age

# ### Pair Plot

# In[ ]:


sns.pairplot(dfinal[["Age", "Job", "Credit amount", "Duration", "Purpose"]], hue='Purpose', diag_kind='hist')
plt.show()


# #### Fungsi
# Pair plot diatas menunjukkan distribusi dari masing-masing variabel dan hubungan antara variabel tersebut dengan variabel yang lain
# #### Insight
# Dari keseluruhan variabel dependen memiliki hubungan dengan variabel independen, tujuan pinjaman kredit mobil merupakan pinjaman yang paling sering dilakukan untuk keseluruhan variabel

# In[ ]:




