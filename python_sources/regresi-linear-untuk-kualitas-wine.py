#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Matplotlib Pyplot
import seaborn as sns # Plotting seaborn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Load dataset dan intip isi atasnya sebanyak 5 row

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


train.index = train['id']
train = train.drop(columns='id')
train.head()


# Mari kita lihat-lihat sekilas tentang data ini

# In[ ]:


train.isna().sum()


# Tidak ada data yang hilang dari dataset ini, mari kita inspeksi lebih lanjut

# In[ ]:


sns.pairplot(train)


# In[ ]:


train.describe()


# Seluruh data sudah dalam bentuk numerik, hanya saja perlu standarisasi dengan Z-Score pada fitur-fitur input, selanjutnya kita akan melakukan feature engineering

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true,y_pred)**0.5


# Kita akan menginspeksi interaksi antara peubah dengan melihat korelasinya

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(train.corr(), annot=True, fmt='.03f',linewidths=.5)


# Bisa dilihat bahwa semua peubah memiliki interaksi dengan peubah quality, kecuali citric acid, free sulfur dioxide, dan sulphates

# In[ ]:


x = train.drop(columns=['quality','citric acid','free sulfur dioxide','sulphates'])
y = train['quality']


# Skalakan x dengan Z-Scorenya

# In[ ]:


x_scaled = StandardScaler().fit_transform(x)


# split train dan test untuk validasi

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state=666)


# Lakukan regresi dengan Regresi Linear

# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
rmse(y_test, pred)


# Didapatkan rmse sebesar 0.8, mari kita buat submisi

# In[ ]:


#load test dan skalakan
test = pd.read_csv('../input/test.csv')
test.index = test['id']
test = test.drop(columns=['id','citric acid','free sulfur dioxide','sulphates'])
test.head()


# In[ ]:


test_scaled = StandardScaler().fit_transform(test)
ans = reg.predict(test_scaled)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub.index = sub['id']
sub = sub.drop(columns='id')
sub['quality'] = ans
sub.head()


# In[ ]:


sub.to_csv('submission.csv')

