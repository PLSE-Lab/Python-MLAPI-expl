#!/usr/bin/env python
# coding: utf-8

# # Import library

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import silhouette_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Load & Quick-look the dataset

# Pada tahap pertama ini, digunakan untuk menentukan feature yang akan digunakan untuk klasteri sasi agar mendapat klaster yang hampir seragam

# In[ ]:


df = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')
df.shape


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df = df.drop(columns = 'Unnamed: 0')


# Kolom 'Unnamed: 0' didrop karena memiliki data yang sama dengan index, sehingga dianggap tidak perlu

# In[ ]:


df.info()


# Dari info datas, terdapat beberapa variable numerik dan kategorikal yang terlihat dari tipedatanya. Namun ada feature yang memiliki tipe data numerik padahal merupakan feature kategorikal, contohnya Job

# In[ ]:


df.describe()


# Dari yang ditampilkan, rata rata pengguna credit memiliki umur 35 tahun, credit amount 3271 dan durasinya 20 bulan.

# # 2. Identifying the missing values

# Pada tahap ini, dilakukan identifikasi terhadap data yang memiliki NULL value, atau bisa dikatakan kosong pada baris tertentu

# In[ ]:


df.isnull().sum()


# Setelah dilakukan pengecekan, null values berada pada saving account dan checking account yang merupakan data kategorik, sehingga tidak dilakukan pengisian value dikarenakan proses klasterisasi menggunakan data numerik.

# # 3. Get insight with data visualization

# Pada tahap ini, kita akan melakukan pengamatan dari visualisasi data yang diharapkan memberi pandangan baru terhadap data yang akan kita teliti.

# In[ ]:


df['Cicil'] = df['Credit amount'] / df['Duration']


# Kolom Cicil merupakan pembagian dari Credit amount dan Duration, yang merupakan pembayaran rutin yang harus dibayarkan tiap bulan selama periode durasi tertentu.

# In[ ]:


#create correlation with heatmap
corr = df.corr(method = 'pearson')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(20,20)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)


# Dari tabel diatas, dapat disimpulkan bahwa korelasi yang relatif besar terjadi pada feature Cicil & Credit amount sebesar 0.52, Duration & Credit amount sebesar 0.62 dan lain hal.

# In[ ]:


sns.pairplot(df)


# Dari diatas kita dapat mengetahui persebaran data numerik yang terbentuk.

# In[ ]:


plt.figure(figsize=(16, 6))
sns.countplot(x='Purpose', data=df, order = df['Purpose'].value_counts().index).set_title('Jumlah dan tujuan mengambil kredit')


# Setelah mengetahui jumlah orang untuk setiap tujuan, sewajarnya kita penasaran dengan umur orang yang mengambil tujuan tersebut. Lalu dibawah disajikan rata-rata umur orang pada setiap tujuan yang dipilih

# In[ ]:


purpose = df.groupby(by='Purpose').mean()['Age']
purpose_df = pd.DataFrame({'Purpose' : purpose.index, 'Age' : purpose.values.astype(int)})
fig = px.line(purpose_df, x="Purpose", y="Age", title='Rata-rata umur pada setiap tujuan kredit')
fig.show()


# Dari grafik diatas terlihat umur terendah ada pada furniture/equipment, namun vacation/other memiliki umur yang paling tinggi.

# In[ ]:


df.groupby(by='Housing').mean()


# Kepemilikan rumah rerata berada pada umur 35 tahun, sedangkan penyewa terbanyak berada pada umur 30an tahun. namun rerata umur 43 tahun tidak memiliki rumah.

# In[ ]:


plt.figure(figsize=(16, 6))
sns.distplot(df['Credit amount'], kde = True, color = 'darkblue', label = 'Credit amount').set_title('Distribution Plot of Credit amount')


# In[ ]:


df_job = df.where(df['Job']==0).dropna()
df_job.where(df_job['Credit amount'] >= df['Credit amount'].mean()).dropna()

Dari hasil query pada dataset diatas, dapat disimpulkan bahwa tiap umur memiliki kecenderungan yang unik. data diatas adalah orang dengan job UnSkilled namun memiliki credit amount lebih dari rata-rata
# # 4. Clustering with k-means
# 
# Feature yang akan digunakan untuk penelitian ini adalah Age, Credit Amount, Duration, dan Cicil**.

# ## 4.1 Importing models and training the data
# Klasterisasi menggunakan K-Means, dikarenakan pengaplikasiannya yang cukup mudah dan memiliki keakurasian yang cukup bagus

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np


# ## 4.2 Feature: Job - Credit amount - Duration

# ### 4.2.1 Ambil Feature dan Normalisasi

# Normalisasi menggunakan Logaritmik

# In[ ]:


X = np.asarray(df[["Job", "Credit amount", "Duration"]])
X[:,0] = X[:,0] + 1
# X[:,1] = np.log(X[:,1])
# X[:,2] = np.log(X[:,2])
X = np.log(X)


# ### 4.2.2 Mencari Nilai K Menggunakan Elbow Method

# In[ ]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(15,10))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# Pada grafik diatas diambil K=3

# ### 4.2.3 Klasterisasi dengan K yang Telah Didapat

# In[ ]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans.labels_


# ### 4.2.4 Evaluasi

# Pada tahap evaluasi ini, menggunakan Silhouette score yang akan dibandingkan dengan feature lain

# In[ ]:


ss_1 = silhouette_score(X, kmeans.labels_, metric='euclidean')


# ### 4.2.5 Visualisasi

# In[ ]:





# In[ ]:


fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(X[:,0], X[:,1], X[:,2], c=kmeans.labels_, cmap='rainbow')
ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c='black')

xLabel = ax.set_xlabel('Jobs', linespacing=3.2)
yLabel = ax.set_ylabel('Credit amount', linespacing=3.1)
zLabel = ax.set_zlabel('Duration', linespacing=3.4)
print("Grafik klasterisasi Jobs - Credit Amount - Duration")


# In[ ]:


df['Risk Jobs'] = kmeans.labels_


# Langkah selanjutnya sama seperti diatas, namun akan dikombinasikan dengan Feature yang telah dipilih diatas

# In[ ]:


X = np.asarray(df[["Age", "Credit amount", "Duration"]])
X = np.log(X)


# In[ ]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(15,10))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans.labels_


# In[ ]:


ss_2 = silhouette_score(X, kmeans.labels_, metric='euclidean')


# In[ ]:


fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(X[:,0], X[:,1], X[:,2], c=kmeans.labels_, cmap='rainbow')
ax.scatter3D(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c='black')

xLabel = ax.set_xlabel('Age', linespacing=3.2)
yLabel = ax.set_ylabel('Credit amount', linespacing=3.1)
zLabel = ax.set_zlabel('Duration', linespacing=3.4)
print("Grafik Klasterisasi Age - Credit Amount - Duration")


# In[ ]:


df['Risk Ages'] = kmeans.labels_


# In[ ]:


X = np.asarray(df[["Credit amount", "Duration"]])
X = np.log(X)


# In[ ]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(15,10))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans.labels_


# In[ ]:


ss_3 = silhouette_score(X, kmeans.labels_, metric='euclidean')


# In[ ]:


fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')

plt.xlabel('Duration')
plt.ylabel('Credit amount')

print("Grafik klasterisasi Duration - Credit Amount")


# In[ ]:


df['Risks'] = kmeans.labels_


# In[ ]:


X = np.asarray(df[["Cicil", "Age"]])
X = np.log(X)


# In[ ]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(15,10))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans.labels_


# In[ ]:


ss_4 = silhouette_score(X, kmeans.labels_, metric='euclidean')


# In[ ]:


fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')

plt.xlabel('Cicil')
plt.ylabel('Age')

print("Grafik klasterisasi Cicil - Age")


# In[ ]:


df['Risk Cicil'] = kmeans.labels_


# In[ ]:


df.sort_values(by='Cicil', ascending=False).head()


# In[ ]:


df_result = df.drop(columns=['Cicil', 'Risk Jobs', 'Risk Ages', 'Risks', 'Risk Cicil'])
df_result.head()


# ### 4.2.6 Summary of Silhouette Score

# In[ ]:


print('silhouette Job - Credit amount - Duration: ', ss_1) 
print('silhouette Age - Credit amount - Duration: ', ss_2) 
print('silhouette Credit amount - Duration: ', ss_3) 
print('silhouette Cicil - Age: ', ss_4) 


# Dari hasil yang didapat, silhouette yang paling bagus menggunakan feature Credit Amount & Duration. Langkah selanjutnya adalah pengecekan K dengan silhouette score

# In[ ]:


X = np.asarray(df[["Credit amount", "Duration"]])
X = np.log(X)


# In[ ]:


silhouette = []
K = range(3,6)
for k in K:
    km = KMeans(n_clusters=k)
    km.fit(X)
    ss = silhouette_score(X, km.labels_, metric='euclidean')
    silhouette.append(ss)
    
pd.DataFrame({'K' : K, 'Silhouette' : silhouette})


# In[ ]:


km = KMeans(n_clusters=4)
km.fit(X)


# In[ ]:


df_result['Risk'] = km.labels_


# Lalu hasil klasifikasi dapat dilihat pada df_result, dan jika ada kesalahan mohon diberi kritik pada kolom komentar

# Pada kolom risk, terdapat value:<br>
# 0 : Bad<br>
# 1 : Medium<br>
# 2 : Good<br>
# 3 : Very God

# In[ ]:


df_result


# In[ ]:




