#!/usr/bin/env python
# coding: utf-8

# # Recommendation System (Content Based Filtering)
# 
# Halo teman-teman, bagaimana kabar kalian semua? saya harap kalian semua baik-baik saja ya. Dikesempatan kali ini saya ingin mendemonstrasikan cara membuat Recommendation System berdasarkan kemiripan atribut item (Content Based Filtering).
# 
# Content Based Filtering adalah teknik sistem rekomendasi yang merekomendasikan item berdasarkan kemiripan atribut yang dimiliki suatu item. Contohnya jika seseorang menonton film X, maka dia akan direkomendasikan film yang mirip dengan film X (atributnya memiliki kemiripan, seperti nama artis, alur cerita, dll), dan inilah yang akan kita buat.
# 
# Goal utama yang akan kita lakukan adalah: merekomendasikan beberapa list film, berdasarkan kemiripan film yang di tonton oleh user.
# 
# Biar lebih jelas, kita langsung ke materinya ya.

# ## Import Packages

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


# ## Import Data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/content_by_synopsis.csv")
df.head()


# Data kita memiliki kolom overview yang nantinya akan kita gunakan untuk mengukur tingkat kemiripan suatu film.
# 
# Next, kita buat Bag of Words.

# ## Bag of Words 
# 
# Apa itu BoW ? BoW adalah teknik untuk mengekstrak text menjadi vector. Text yang merupakan unstructured data akan diubah menjadi data tabular (structured), yang nantinya bisa dimanfaatkan untuk kebutuhan machine learning. Dan mohon maaf saya tidak akan menjelaskannya secara detil karena bukan itu fokus bahasan kita kali ini hehe.
# 
# Buat variabel "bank" untuk menampung BoW

# In[ ]:


bow = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
bank = bow.fit_transform(df['overview'])


# ## Modeling
# 
# Berikut adalah step-step yang akan kita lakukan:
# 
# - Step 1: Encode film yang user lihat
# - Step 2: Cari kesamaan
# - Step 3: Rekomendasikan

# ## Step 1: Encode film apa yang user lihat

# Kita gunakan index untuk mengetahui sinopsis dari film yang di tonton user. Contohnya, jika user melihat film "Toy Story", maka kita tidak perlu repot menulis judulnya, cukup gunakan index 0.

# In[ ]:


idx = 0 # Toy Story


# In[ ]:


content = df.loc[idx, "overview"]
content


# Setelah overview dari filmya diambil, selanjutnya kita akan melakukan encoding.

# In[ ]:


code = bow.transform([content])
code


# In[ ]:


pd.DataFrame(code.toarray())


# ## Step 2: Cari kesamaan

# Kita akan gunakan Cosine Distance untuk menghitung similarity nya. Karena secara umum memang Cosine Distancelah yang digunakan untuk mengukur kesamaan document.

# In[ ]:


from sklearn.metrics.pairwise import cosine_distances


# Hitung cosine distance.

# In[ ]:


distance = cosine_distances(code, bank)
pd.DataFrame(distance)


# Bisa dilihat, semua value memiliki jarak antara 0-1, nah distance yang paling kecillah yang akan kita rekomendasikan, distance yang paling kecil artinya film tersebut memiliki tingkat kemiripan paling tinggi.
# 
# Langkah selanjutnya, kita akan sorting 10 film yang memiliki tingkat kemiripan paling tinggi.

# In[ ]:


rec_idx = distance.argsort()[0, 1:11]
rec_idx


# ## Step 3: Rekomendasikan

# Setelah disorting, tinggal direkomendasikan deh.

# In[ ]:


df.loc[rec_idx]


# Namun cara diatas masih memiliki kekurangan, karena tingkat kemiripan hanya dihitung berdasarkan overview saja, sehingga hasilnya tidak begitu bagus. Contohnya film "The 40 Year Old Virgin", film itu tidak memiliki kemiripan sama sekali dengan film "Toy Story", kemiripan yang dimiliki hanya berdasarkan kata "Andy" yang sama-sama sering mucul dikedua film tersebut. Tapi hal yang perlu diingat, dalam rekomendasi sistem tidak ada jawaban yang benar, jadi rekomendasi diatas sah-sah saja.
# 
# Nantinya kita akan coba dengan data yang berbeda dan data campuran, akankah hasilnya akan jadi lebih bagus? kita lihat saja nanti. Sebelum itu, kita rangkum dulu semua code diatas kedalam Class agar code lebih mudah untuk digunakan kembali.

# In[ ]:


from sklearn.metrics.pairwise import cosine_distances

class RecommenderSystem:
    def __init__(self, data, content_col):
        self.df = pd.read_csv(data)
        self.content_col = content_col
        self.encoder = None
        self.bank = None
        
    def fit(self):
        self.encoder = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])
        
    def recommend(self, idx, topk=10):
        content = df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rec_idx] 


# Kita coba tes gunakan Class yang sudah kita buat. Untuk kedepannya Kita akan coba tampilkan film Toy Story, Jumanji, dan Home Alone.

# In[ ]:


recsys = RecommenderSystem("/kaggle/input/content_by_synopsis.csv", content_col="overview")
recsys.fit()


# In[ ]:


recsys.recommend(0) # Toy Story


# In[ ]:


recsys.recommend(1) # Jumanji


# In[ ]:


recsys.recommend(579) # Home Alone


# ## Content Based Filtering menggunakan Metadata yang berbeda

# ### Import Data

# In[ ]:


df = pd.read_csv("/kaggle/input/content_by_multiple.csv")
df.head()


# Data kali ini memiliki kolom "metadata" yang merupakan rangkuman dari keseluruhan kolom-kolom sebelumnya, dan kolom inilah yang akan kita coba gunakan untuk mencari kesamaan.
# 
# Sama seperti sebelumnya, next step kita akan fit dan rekomendasikan.

# In[ ]:


recsys = RecommenderSystem("/kaggle/input/content_by_multiple.csv", content_col='metadata')
recsys.fit()


# In[ ]:


recsys.recommend(0) # Toy Story


# In[ ]:


recsys.recommend(1) # Jumanji


# In[ ]:


recsys.recommend(579) # Home Alone


# Hasilnya sudah lumayan bagus, bisa kita lihat hampir semua rekomendasi memiliki genre yang sama, yang jadi pertanyaan, apakah akan menjadi lebih bagus rekomendasinya jika kita gabungkan kedua data diatas ? dengan asumsi bahwa semakin banyak data = semakin banyak informasi yang diserap = semakin bagus akurasi kemiripannya. Kita coba saja ya :)

# ## Content Based Filtering menggunakan data gabungan

# Caranya sama seperti sebelumnya, yang membedakan kali ini kita ada step menggabungkan data. Kalo di Dragonball, ibaratnya seperti Fusion, hahaha.
# 
# Back to the topic, kita import data pertama.

# In[ ]:


df = pd.read_csv("/kaggle/input/content_by_synopsis.csv")
df.head()


# Import data kedua.

# In[ ]:


df1 = pd.read_csv("/kaggle/input/content_by_multiple.csv")
#df1 = df1[['title','metadata']]
df1.head()


# Kita gabungkan kolom 'overview' dan 'metadata'.

# In[ ]:


df = df1.set_index('title').join(df.set_index('title'))
df['join'] = df['overview'] + df['metadata']
df.reset_index(inplace=True)
df.head()


# Isi nan value dengan string kosong, karena BoW kita tidak menerima nan value.

# In[ ]:


df.fillna('', inplace=True)


# Kita edit sedikit Class yang sudah dibuat sebelumnya, jika sebelumnya Class kita meminta file csv, sekarang kita ganti menjadi dataframe.

# In[ ]:


from sklearn.metrics.pairwise import cosine_distances

class RecommenderSystem_df:
    def __init__(self, data, content_col):
        self.df = pd.DataFrame(data) #sebelumnya csv, sekarang ubah jadi dataframe
        self.content_col = content_col
        self.encoder = None
        self.bank = None
        
    def fit(self):
        self.encoder = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])
        
    def recommend(self, idx, topk=10):
        content = df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rec_idx] 


# Fit dan rekomendasikan.

# In[ ]:


recsys = RecommenderSystem_df(df, content_col='join')
recsys.fit()


# In[ ]:


recsys.recommend(0) # Toy Story


# In[ ]:


recsys.recommend(1) # Jumanji


# In[ ]:


recsys.recommend(579) # Home Alone


# Bagaimana menurut kalian? apakah hasil rekomendasinya lebih bagus atau tidak jika dibandingkan dengan sebelumnya? Sekali lagi saya ingatkan kembali bahwa rekomendasi sistem tidak ada jawaban pastinya, semuanya tergantung subjektifitas kita dalam menilai kemiripan. 
# 
# Kalau saya pribadi, saya lebih suka hasil dari data kedua, karena hasilnya banyak merekomnedasikan genre yang tepat. Penambahan kolom 'overview' di data gabungan ini tidak banyak membantu, malah menambah noise saja, karena kolom 'overview' isinya berupa kalimat, dimana semakin banyak kata yang memiliki frekuensi kemunculan yang tinggi, dia akan berpeluang besar untuk direkomendasikan. Seperti yang sudah saya jelaskan sebelumnya, film "The 40 Year Old Virgin", film itu tidak memiliki kemiripan sama sekali dengan film "Toy Story", kemiripan yang dimiliki hanya berdasarkan kata "Andy" yang sama-sama sering mucul dikedua film tersebut.
# 
# Mungkin itu dulu dari saya, jika ada pertanyaan atau masukan, bisa kalian tulis di kolom komentar atau boleh japri di ig saya: al.fath.terry
# 
# Dan mohon di upvote jika dirasa bermanfaat.
# 
# Terimakasih :)
