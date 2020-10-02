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


# # Practice Case Clustering
# Topik Zulkarnain
# 
# #### Introduction
# Pada case ini, akan dilakukan teknik clustering untuk mensegmentasi kostumer pada data german_credit_data. Tujuan dari practice ini adalah untuk mengelompokkan kostumer berdasarkan pola perilakunya dalam 3 kategori resiko yaitu good, medium dan bad.

# ## Loading data Set

# In[ ]:


#LOAD DATASET AND ALL MODULES
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/german-credit/german_credit_data.csv")


# ## Exploratory Data Analysis
# 

# In[ ]:


data.head()


# #### Berikut adalah deskripsi dari data
# * Unnamed : 0 = data yang tidak diperlukan akan didrop
# * Age = Usia dari kostumer
# * Sex = Jenis kelamin dari kostumer
# * Job = Jenis pekerjaannya
# * Housing = Status tempat tinggal
# * Saving Account = 
# * Checking Account = 
# * Credit Amount = Jumlah kredit
# * Duration = Lama durasi pengkreditan
# * Purpose = Tujuan pengkreditan

# terdapat missing value dalam dataset dengan feature saving account sebanyak 18.3%, dan checking accounts sebesar 39.4%. Maka akan digunakan statistika deskriptif untuk mengisi missing valuenya

# In[ ]:


percentage_missing_value = (data.isnull().sum())/(len(data)) * 100
percentage_missing_value


# In[ ]:


categorical_missing = ['Saving accounts', 'Checking account']
for x in categorical_missing :
    data[x] = data[x].fillna(data[x].mode().values[0])


# In[ ]:


data = data.drop(columns = ['Unnamed: 0'])


# In[ ]:


data.info()


# data sudah siap untuk 

# ## Data Visualization
# Akan dilakukan visualisai data untuk mendapatkan insight dari pola data 

# In[ ]:


data.head()
data['Job'] = data['Job']+1 #menghindari infinity value ketika transformasi data


# In[ ]:


data_numerik = ['Age', 'Credit amount', 'Duration', 'Job']
data_kategorik = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
data.head()


# In[ ]:


data.describe()


# Dari summary data didapatkan bahwa rataan umur kustomer adalah 35 tahun, jenis pekerjaannya adalah tipe 3, kredit yang dipinjam sebesar 3271, dan durasi peminjaman 21 hari.

# In[ ]:


fig = plt.figure(figsize = (20,20))
axes = 520
for cat in data_kategorik:
    axes += 1
    fig.add_subplot(axes)
    sns.countplot(data = data, x = cat, hue ='Purpose')
plt.show()


# #### Business Insight : Rekomendasi Tawaran Kredit Kepada Kostumer
# Berdasakan grafik beberapa feature terhadap tujuan pengkreditan maka dapat dibentuk suatu startegi tawaran yang dapat berupa suatu iklan atau promo terhadap kostumer dengan kriteria tertentu sebagai berikut :
# * Rekomendasi Kredit Car untuk kostumer dengan kriteria :
#     - Bergender pria 
#     - Memiliki job tipe 2
#     - Memiliki rumah sendiri
#     - Memiliki kategori saving accounts little
#     - Memiliki kategori checking accounts little
# * Rekomendasi kredit radio/TV
#     - Bergender pria 
#     - Memiliki job tipe 2
#     - Memiliki rumah sendiri
#     - Memiliki kategori saving accounts little
#     - Memiliki kategori checking accounts little
# ***

# #### Visualisasi Lama Durasi Peminjaman berdasarkan kriteria Kostumer

# In[ ]:


fig = plt.figure(figsize = (20,20))
axes = 520
for cat in data_kategorik:
    axes += 1
    fig.add_subplot(axes)
    sns.barplot(data = data, x = cat, y ='Duration')
plt.show()


# #### Business Insight : Resiko lama durasi peminjaman kredit
# Berdasarkan visualisasi data berdasarkan durasi peminjaman tiap kriteria, dapat dilihat bahwa kriteria yang beresiko lama peminjamannya adalah :
# * Pria lebih lama dibanding perempuan
# * Jenis pekerjaan 3 lebih lama daripada jenis pekerjaan lain
# * orang yang status kepemilikan rumahnya free lebih lama dari yang lain
# * Saving dan Check account moderate cenderung lebih lama durasi peminjamannya
# * Tujuan pengkreditan yang paling lama durasinya adalah untuk Vacation lalu untuk bisnis

# ## Clustering Kustomer
# Pada kasus clustering pada permasalahan ini akan digunakan feature yang paling berpengaruh diantaranya adalah jenis pekerjaan, jumlah pengkreditan, usia, dan durasi peminjaman 

# In[ ]:


data_cluster = data[['Age', 'Credit amount', 'Duration']]
data.head()


# In[ ]:


import seaborn as sns
cor = data_cluster.corr() #Calculate the correlation of the above variables
plt.figure(figsize=(10,10))
sns.heatmap(cor, square = True) #Plot the correlation as heat map


# #### Melihat sebaran data
# Akan dilakukan visualisai penyebaran data sebelum dilakukan teknik clustering

# In[ ]:


plt.figure(figsize=(15,3))
sns.boxplot(x = data_cluster['Age'])
plt.show()
plt.figure(figsize=(15,3))
sns.boxplot(x = data_cluster['Credit amount'])
plt.show()
plt.figure(figsize=(15,3))
sns.boxplot(x = data_cluster['Duration'])
plt.show()

dari boxplot diatas dapat dilihat bahwa terdapat banyak outlier pada data yang harus ditanggulangi dengan penggunaan transformasi data (pada kasus ini akan dilakukan transformasi logaritmik). Hal ini perlu dilakukan karena jarak antar data akan menyebar sangat jauh sehinggu pengclusteran akan sulit untuk dilakukan
# ## Algoritma K-Means

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans


# In[ ]:


#Data diberlakukan tranformasi logaritmik agar variansi data lebih masuk kedalam range
cluster_credit_duration = np.log(data_cluster[['Age','Credit amount', 'Duration']])
cluster_credit_duration.head()


# In[ ]:


X = np.array(cluster_credit_duration)


# In[ ]:


#Scree Plot untuk menentukan nilai K
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.figure(figsize = (15,5))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# pada scree plot dapat dilihat bahwa K yang optimal berada pada 3 atau 4. Pada kasus ini akan dipilih K=3, yaitu cluster yang baik, sedang, dan buruk.

# In[ ]:


#mengaktifkan algoritma k-means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


# In[ ]:


print(kmeans.cluster_centers_)


# In[ ]:


print(np.exp(kmeans.cluster_centers_))


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(num = None, figsize = (10,5), dpi = 80, facecolor = 'w', edgecolor ='k')
ax = Axes3D(fig)
ax.scatter3D(X[:,0], X[:,1], X[:,2], c = kmeans.labels_, cmap = 'rainbow')
ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c = 'black')

xLabel = ax.set_xlabel('Age', linespacing = 3.2)
yLabel = ax.set_ylabel('Credit amount', linespacing = 1.2)
zLabel = ax.set_zlabel('Duration', linespacing = 1.5)


# Berdasarkan dari plot 3D terlihat data telah di kelompokkan menjadi 3 yaitu cluster 0, 1 dan 2. Ciri cluster 0 adalah yang rataan customernya berumur sekitar 33 tahun, jumlah kreditnya 2435 dengan durasi peminjaman selama 18 hari. Ciri cluster 1 adalah yang rataan customernya berumur sekitar 35 tahun, jumlah kreditnya 6491 dengan durasi peminjaman selama 32 hari. Ciri cluster 2 adalah yang rataan customernya berumur sektiar 34 tahun, jumlah kreditnya 1087 dengan durasi peminjaman selama 10 hari. Berikut adalah pendefinisian cluster:
# * Cluster 0 : Good Credit Risk
# * Cluster 1 : Medium Credit Risk
# * Cluster 2 : Bad Credit Risk    

# In[ ]:


print(kmeans.labels_)


# In[ ]:


data['Cluster (K-Means)'] = kmeans.labels_


# In[ ]:


data.head()


# In[ ]:


#arr = []
#for i in range(len(data['Cluster (K-Means)'])):
#    if data['Cluster (K-Means)'][i] == 0:
#        arr.append('Good Credit')
#    elif data['Cluster (K-Means)'][i] == 1:
#        arr.append('Medium Credit')
#    else :
#        arr.append('Bad Credit')
#data['Cluster (K-Means)']=arr


# In[ ]:


data.head()


# In[ ]:


#Membuat feature pembayaran kredit per waktu, dibuat untuk memvalidasi cluster. 
# Cluster 0 adalah cluster good credit yang berarti jumlah kredit banyak namun durasi pembayaran sebentar.
data['Pay/Time'] = data['Credit amount'] / data['Duration']


# In[ ]:


data.sort_values(by = ['Pay/Time'], ascending = False).head(10)


# In[ ]:


data.sort_values(by = ['Pay/Time'], ascending = True).head()


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(data=data, x='Cluster (K-Means)', hue='Purpose')


# Berdasarkan plot diatas dapat disimpulkan bahwa tujuan kredit yang bagus risikonya (cluster 0) urutannya adalah :
# 1. Car
# 2. Radio/TV
# 3. Furniture / Equipment
# 4. Education
# 5. Repairs
# 6. Vacation / others
# 
# Tujuan kredit yang sedang resikonya  (cluster 1) urutannya adalah :
# 
# 1. Car
# 2. Radio/TV
# 3. Business
# 4. Furniture/Equipment
# 5. Education
# 6. Vacation/others
# 7. Repairs
# 8. Domestic Appliances
# 
# Tuujuan kredit yang besar resikonya (cluster 2) urutanya adalah :
# 1. Radio/Tv
# 2. Car
# 3. Education
# 4. Business
# 5. Domestic Appliances
# 6. Repairs
# 7. Vacation

# In[ ]:


fig = plt.figure(figsize = (20,20))
axes = 520
for cat in data_kategorik:
    axes += 1
    fig.add_subplot(axes)
    sns.countplot(data = data, y = cat, hue ='Cluster (K-Means)')
plt.show()


# ## Hierarchical Clustering

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_


# In[ ]:


plt.figure(figsize = (20,25))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(num = None, figsize = (10,5), dpi = 80, facecolor = 'w', edgecolor ='k')
#ax = plt.axes(prohection='3d')
ax = Axes3D(fig)
ax.scatter3D(X[:,0], X[:,1], X[:,2], c = labels, cmap = 'rainbow')
#ax.scatter3D(labels.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c = 'black')

xLabel = ax.set_xlabel('Age', linespacing = 3.2)
yLabel = ax.set_ylabel('Credit amount', linespacing = 1.2)
zLabel = ax.set_zlabel('Duration', linespacing = 1.5)


# In[ ]:


print(labels)


# Hasil clustering dari Hierarchical Clustering dengan K=3 memiliki kelompok yang cukup mirip dengan K-Means hanya saja cluster kredit dengan medium risk lebih banyak sehingga cluster good dan bad menjadi lebih sedikit dibanding K-Means

# In[ ]:


data['Cluster (Hierarchical)'] = labels


# In[ ]:


data.head(10)


# # DBSCAN

# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


dbscan = DBSCAN(eps = 0.09, min_samples = 3)
dbscan.fit(X)


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(num = None, figsize = (15,10), dpi = 80, facecolor = 'w', edgecolor ='k')
#ax = plt.axes(prohection='3d')
ax = Axes3D(fig)
ax.scatter3D(X[:,0], X[:,1], X[:,2], c = dbscan.labels_, cmap = 'rainbow')
#ax.scatter3D(labels.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c = 'black')


xLabel = ax.set_xlabel('Age', linespacing = 3.2)
yLabel = ax.set_ylabel('Credit amount', linespacing = 1.2)
zLabel = ax.set_zlabel('Duration', linespacing = 1.5)


# In[ ]:


print(dbscan.labels_)


# Hasil dari DBSCAN dengan epsilon 0.09 dan min samples 3, terlihat bahwa DBSCAN tidak cocok untuk mengcluster data ini

# In[ ]:


####


# ### KESIMPULAN
# Metode clustering terbaik pada kasus ini adalah dengan menggunakan clustering K-Means, dari hasil cluster maupun visualisasi data dapat dilihat resiko credit nya sehingga dapat diambil manfaat peneglompokkan kostumer berdasarkan pola perilakunya

# In[ ]:


data


# In[ ]:




