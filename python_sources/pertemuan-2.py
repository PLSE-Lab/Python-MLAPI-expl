#!/usr/bin/env python
# coding: utf-8

# ## Import package

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
import numpy as np 


# ## ini mencoba menginput image
# kalian bisa ambil urlnya, lalu masukan ke code

# In[ ]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")


# ## Input Data Titanic
# terdiri dari 2 data, data Train dan Test

# In[ ]:


# ini codingan untuk membaca data 
train = pd.read_csv("/kaggle/input/titanic/train.csv") 
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


# memunculkan data train
train


# In[ ]:


# memunculkan data test
test


# In[ ]:


# ini untuk memunculkan data hanya 5 row data di atas
train.head()


# In[ ]:


# ini untuk memuncilkan data hanya 5 row di bawah 
train.tail()


# In[ ]:


# ini code untuk melihat, berapa ukuran data (row, coloumns)
train.shape


# In[ ]:


# menunjukan, jumlah data, dan type data 
train.info()


# In[ ]:


# ini untuk data angka (numberic)
# untuk melihat persebarannya secara statistik
train.describe()


# ## Visualisasi latihan

# In[ ]:


# ini mengimport package untuk visualisasi data 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# ini unutuk membuat bar plot, 
sns.barplot(x='Sex', y='Survived', data=train)


# ## Latihan Visualisasi,
# #### coba kalian buat bar plot, 
# #### antara "Embarked" dengan "Survived"
# #### antara "Pclass" dengan "Survived" 
# #### antara "Cabin" dengan "Survived"
# 

# ## Spliting Data 
# memisahkan data atau memotong2 data

# In[ ]:


# ini untuk ngambil salah satu colomn, 
train["Sex"]
# format data merupakan (series), 


# In[ ]:


# mengambil 1 Columns
train[['Sex']]
# format data merupakan "DataFrame"


# In[ ]:


# ini untuk mengambil data lebih dari 1 
train[['Sex', 'Cabin']]


# In[ ]:


# ini buat ngambil row nya dari data 
train.iloc[2:9]


# #### latihan Spliting data, 
# coba kalian split masing columns

# ## Convert data To numberik
# ctt: untuk memasukan ke model kita harus merubah data pada numberik,karena modeling pada python hanya menerima inputan numerik 
# 

# In[ ]:


# ini buat ganti data kategorik jadi data numberik 
# ini memisalkan male = 1, female = 2
sex_map = {"male" : 1,"female" :0 }
train["Sex"] = train['Sex'].map(sex_map)


# In[ ]:


# ini buat ganti data kategorik jadi data numberik 
sex_map = {"male" : 1,"female" :0 }
test["Sex"] = test['Sex'].map(sex_map)


# In[ ]:


# cek data, apakah male dan female sudah berubah?
test.head()


# ### Latihan Convert 
# 
# kita tau bahwa masih banyak data yang belum di convert, coba convert data2 tersebut

# In[ ]:


# ini untuk menghitung jumlah setiap nilai colomnnya
# dapat juga di gunakan untuk mengetahui, berapa jumlah type dari setiap columns
train['Embarked'].value_counts()


# In[ ]:


train.head()


# ## ini untuk mendrop data
# usahakan dalam data jangan banyak data yang di drop, 
# coba kalian convert terlebih dahulu agar dapat di input pada model 

# In[ ]:


# ini untuk mendrop data colomns lebih dari 1
train = train.drop(['Ticket', "Cabin", "Embarked"], axis = 1)
test = test.drop(['Ticket', 'Cabin', "Embarked"], axis = 1)


# In[ ]:


# ini merupakan data yang sudah numberik semua
train


# In[ ]:


# kita cek apakah data tersebut sudah bertype int atau float, 
# bila sudah float dan int maka sudah siap untuk di masukan ke dalam model

train.info()

# tetapi masih ada data2 yang null/kosong belum terisi


# In[ ]:


# ini cara mengisi colomns yang kosong dengan rata2 nya
train['Age'] = train['Age'].fillna(train["Age"].mean())


# In[ ]:


train.info()


# ## Modeling

# 

# In[ ]:


train_data = train.drop("Survived", axis =1 )
target = train['Survived']

train_data.shape, target.shape


# ### Import data untuk modeling

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from  sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# #### cross validation
# 

# In[ ]:


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### model, model yang kita gunakan KNN
# kalian bisa coba model2 yang lain 
# 
# 
# ## penjelasan mengenain KNN ( K-Nearst Neightbors)
# 
# https://medium.com/@muhajir_29/algoritma-k-nearest-neighbors-knn-785d328bf110

# In[ ]:


model = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(model, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# ini merupakan rata2 accuracy 
round(np.mean(score)*100, 2)


# ## Prediction

# In[ ]:


test


# In[ ]:





# In[ ]:




