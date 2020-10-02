#!/usr/bin/env python
# coding: utf-8

# # Classification Problem: Bank Marketing Dataset
# Dataset ini merupakan data Direct Marketing Campaign sebuah Bank di Portugis, Informasi lengkap mengenai dataset ini dapat dilihat di [sini](https://www.kaggle.com/henriqueyamahata/bank-marketing). Sesuai dengan keterangan, permasalahan yang dihadapi adalah untuk membuat model untuk memprediksi apakah seseorang akan berlangganan Deposito Berjanka (Deposito Jangka Panjang).  Saya cukup familiar dengan dataset ini, karena dulu saya pernah mengikuti kompetisi nasional data science yang menggunakan dataset ini sebagai soal. Saya lupa berapa akurasi yang pernah saya dapatkan dan seberapa bagus model yang saya buat, yang saya ingat saya kalah. :). Mari kita lihat apa yang bisa saya lakukan kali ini. 
# 
# **Bagi yang belum tau, solusi model untuk dataset ini sudah banyak yang membuat, tapi kali ini saya mencoba mengerjakannya dengan kemampuan sendiri, hitung-hitung latihan :)**

# In[ ]:


# Import Packages
import pandas as pd # package untuk pengolahan data
import seaborn as sns # package untuk visualisasi data


# In[ ]:


# Import Dataset
df = pd.read_csv("../input/bank-marketing-dataset/bank.csv")

# Menampilkan Informasi Dataset
df.info()


# Dapat kita lihat dataset yang kita memiliki 17 kolom 11162 entry.

# ## Exploratory Data Analysis
# ![Data Science Process: Wikipedia](https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Data_visualization_process_v1.png/350px-Data_visualization_process_v1.png)
# Saat ini kita sudah memiliki, dataset yang rapi/clean (bisa dikatakan begitu). Tahapan selanjutanya adalah melakukan permodelan, atau melakukan Exploratory Data Analysis (EDA). Menurut saya EDA merupakan suatu tahapan penting dalam proses Data Science. EDA merupakan sebuah pendekatan menganalisis dataset untuk merangkum karakteristik data, seringkali dengan metode visualisasi. Tujuan EDA adalah untuk mempelajari permasalahan dalam data serta melihat insight diluar proses permodelan. EDA akan membantu untuk menjawab pertanyaan-pertanyaan dasar terkait permasalahan yang dihadapi. Tahapan awal biasanya saya suka melihat seperti apa data kita.

# In[ ]:


# Menampilkan 10 baris teratas 
df.head(10)


# Target variable dari dataset yang kita miliki adalah variable "deposit". Mari kita lihat untuk distribusi target variablenya. 

# In[ ]:


# Countplot variable 'deposit'
sns.countplot(x='deposit',data=df)


# Seperti yang dapat kita lihat dari visualisasi diatas, kecenderangan "NO" memang lebih besar namun tidak terlalu jauh. Ini cukup keren untuk saya, karena yang berminat berlangganan cukup banyak, ini bisa menandakan direct marketingnya cukup berhasil/atau memang produknya memang bagus. Menurut simple saya, hal ini cukup jarang hal ini terjadi di dunia nyata. 
# 
# Oke kembali ke masalah utama, mari kita lihat apakah ada data/nilai kosong di dataset kita. 

# In[ ]:


# Melihat nilai data yang kosong
df.isnull().sum()


# **Good** sejauh ini datanya terlihat bagus karena tidak ada data/nilai kosong. 
# Dalam dokumentasi data terdapat keterangan bahwa variable 'duration' (nilai lama/durasi telpon terakhir dalam detik) sangat mempengaruhi target variable 'deposit' **(contoh jika duration=0 maka y='no')**. Sangat masuk akal, mari kita explore lebih dalam.

# In[ ]:


# Deskripsi Statistik Variable Duration
df['duration'].describe()


# In[ ]:


# Contingency Table
data_crosstab = pd.crosstab(df['duration'], 
                            df['deposit'],  
                               margins = True) 
data_crosstab.transpose()


# Dapat kita lihat dari 2 analisa diatas, bahwa tidak ada data duration = 0, nilai minimalnya adalah 2 detik dan nilai tertinginya adalah 3881 detik (1 jam 4 Menit).

# In[ ]:


sns.boxplot(x='deposit', y='duration', data=df)


# In[ ]:


# Melihat Distribusi Data Duration
sns.distplot(df['duration'])

