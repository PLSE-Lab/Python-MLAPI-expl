#!/usr/bin/env python
# coding: utf-8

# > ## Melakukan import tools numpy dan pandas untuk mengolah data, serta mengambil dataset

# In[1]:


import numpy as np # aljabar linear
import pandas as pd # pemrosesan data, CSV file I/O (seperti pd.read_csv)

import os
print(os.listdir("../input"))

# mendefinisikan nama-nama kolom
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# meload data ke data-frame
df = pd.read_csv('../input/Iris.csv', header=None, names=names)

# mengganti nama-nama header sesuai names
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = names #set the header row as the df header

# tampilkan 6 data teratas, untuk memastikan kebenaran data
df.head()


# In[2]:


# Melihat gambaran besar data
df.describe()


# In[3]:


# Menghitung jumlah data berdasarkan kelas
df.groupby('class').size()


# #### ^ Dapat dilihat bahwa setiap class memiliki 50 data

# In[4]:


# memilih data mana saja yang kita anggap sebagai atribut
kolom_atribut = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Masukkan atribut ke axis x, dan kelas ke axis y
X = df[kolom_atribut].values
y = df['class'].values

# Cara alternatifnya:
# X = df.iloc[ :, 0:4 ]
# y = df.iloc[ :, 4 ]


# ### Label Encoding
# Hal ini penting untuk merubah tipe data object (seperti data kita pada kolom class) menjadi tipe data numbers, agar dapat diterima oleh KNeighborClassifier.
# 
# Iris-setosa diganti menjadi 0,
# Iris-versicolor diganti menjadi 1, dan
# Iris-virginica diganti menjadi 2.

# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# ### Memilah dataframe menjadi training set dan test set
# Setelah dibagi, kita dapat menguji apakah classifier kita bekerja dengan benar atau tidak

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0 )


# ### Visualisasi Data

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Pairplot
plt.figure()
sns.pairplot( df, hue = "class", height = 3, markers = ["o", "s", "D"])
plt.show()


# ## Menggunakan KNN

# In[8]:


# Loading library
from sklearn.neighbors import KNeighborsClassifier

# inisiasi Learning model ( k = 3 )
knn = KNeighborsClassifier( n_neighbors = 3 )

# fitting model tersebut
knn.fit( X_train, y_train )

# prediksi responnya
pred = knn.predict( X_test )

# evaluasi tingkat akurasi
# from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
print ( accuracy_score( y_test, pred) )


# ## Menggunakan Cross-validation dan mencari nilai K yang optimal
# 
# K-fold cross-validation akan membagi dataframe menjadi beberapa bagian (contohnya 10), menjalankan algoritme KNN dengan berbagai nilai K (contohnya dari 1 sampai 50), lalu membandingkan hasilnya satu sama lain.

# In[9]:


# Buat list untuk menginisiasi nilai K, dari 1 sampai 49
list_k = list( range(1,50) )
print(list_k)

# membuat subset yang hanya memiliki nilai ganjil
neighbors = []
for x in list_k:
    if x%2 != 0:
        neighbors.append(x)
print( neighbors)

# list kosong yang akan menyimpan nilai dari cross-validation
cv_scores = []

# impor dulu cross_val_score
from sklearn.model_selection import cross_val_score

# lakukan 10-fold cross validation
for k in list_k:
    knn = KNeighborsClassifier( n_neighbors = k )
    scores = cross_val_score( knn, X_train, y_train, cv = 10, scoring = 'accuracy' )
    #print(scores)
    cv_scores.append( scores.mean() )
    
for x in range(len(cv_scores)):
    print(cv_scores[x])


# Visualisasi error KNN dengan berbagai nilai K

# In[10]:


# merubah nilai akurasi menjadi misklasifikasi atau error
MSE = [ 1-x for x in cv_scores ]

# Menentukan nilai k yang terbaik
k_optimal = list_k[ MSE.index(min(MSE)) ]
print ("Nilai K yang paling optimal adalah %d" % k_optimal)
print ("K tersebut memiliki akurasi %lf" % cv_scores[ MSE.index(min(MSE)) ] )

# plot error misklasifikasi dengan nilai k-nya
plt.plot( list_k, MSE )
plt.xlabel("Jumlah tetangga -- K")
plt.ylabel("Galat Misklasifikasi")
plt.show()


# In[11]:


d = {'k': list_k, 'accuracy': cv_scores}
df = pd.DataFrame(data=d)
df

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
import matplotlib.pyplot as plt

results = ols('accuracy ~ k', data=df).fit()
results.summary()


# In[12]:


sm.stats.anova_lm(results, typ=2)

