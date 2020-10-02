#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering
# 
# Halo teman-teman, bagaimana kabar kalian semua? saya harap kalian semua baik-baik saja ya. Dikesempatan kali ini saya ingin mendemonstrasikan cara membuat Sistem Rekomendasi Collaborative Filtering. 
# 
# Collaborative Filtering adalah teknik rekomendasi sistem yang memanfaatkan preferensi antar user dan juga item similarity sebagai dasar pertimbangan rekomendasinya. Jadi ada dua data yang di manfaatkan, yaitu preferensi similarity antar user dan juga item similarity. Collaborative Filtering ini tidak memerlukan filter apapun, lalu bagaimana cara kerjanya? 
# 
# Bentuk data dari Collaborative Filtering ini berbentuk Sparse data (datanya bolong-bolong), nantinya kita akan memakai teknik Dimentionality Reduction (Dekomposisi Matriks SVD) untuk mengekstrak Sparse Data tadi. Setelah di ekstrak, selanjutnya kita isi data yang bolong dengan prediksi data yang sudah kita ekstrak sebelumnya. Biar lebih jelas, yuk ikuti saja notebook ini sampai selesai.

# ## Import Packages dan Data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("/kaggle/input/collaborative_filtering.csv")
df.head()


# Kita coba cek jumlah user yang ada dan jumlah rata-rata rating yang dilakukan tiap user.

# In[ ]:


jumlah_voting = len(df['userId'])
jumlah_user = df['userId'].nunique()
rata2_rating = jumlah_voting / jumlah_user

print('Jumlah voting = {}'.format(jumlah_voting))
print('Jumlah user = {}'.format(jumlah_user))
print('rata-rata user melakukan rating = {} / {} = {}'.format(jumlah_voting,jumlah_user, rata2_rating))


# ## Training
# 
# Untuk trainingnya kita akan gunakan library Scikit Surprice, Scikit Surprice adalah library yang biasa digunakan untuk membuat model Recommendation System. 
# 
# Nantinya kita akan memanfaatkan:
# - Reader (digunakan untuk membaca Data).
# - Dataset (digunakan untuk loading data).
# - SVD (Singular Value Decomposition, algoritma yang digunakan untuk dekomposisi).
# - Cross_validate (kita tidak akan melakukan split data, yang akan kita lakukan adalah train keseluruhan dataset dan gunakan predict method untuk prediksi).
# 
# Untuk dokumentasi lengkapnya bisa kalian baca di link berikut:
# https://surprise.readthedocs.io/en/stable/dataset.html
# 
# Pertama kita lakukan import dulu.

# In[ ]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# Kita load data dan buat trainset.

# In[ ]:


data = Dataset.load_from_df(df, Reader())
trainset = data.build_full_trainset() 


# Persiapkan modelnya lalu fit.

# In[ ]:


model = SVD() 
model.fit(trainset)


# Lakukan tes prediksi, kita ambil contoh user ke 1 dengan film "Bug's Life, A (1998)". Di data realnya user 1 memberikan rating 5 kepada film itu, apakah prediksi ratingnya akan mendekati 5?.

# In[ ]:


model.predict(1, "Bug's Life, A (1998)")


# Ternyata score prediksi adalah 4.3, sedangkan real datanya adalah 5. artinya terdapat loss sebesar 0.7. But not bad lah ya.. hehe.
# 
# Setelah melakukan tes, sekarang saatnya kita merekomendasikan film yang belum dirating oleh user 1. Namun sebelum melakukan rekomendasi, terdapat beberapa tahapan yang harus kita lakukan:
# 
# - 1 Mencari total jumlah film yang ada di data set.
# - 2 Mencari film apa saja yang sudah dirating oleh user 1.
# - 3 Mencari film apa saja yang belum dirating oleh user 1. < film-film inilah yang akan kita prediksi, lalu kita rekomedasikan ke user 1 berdasarkan nilai rating.
# 
# Dan kita akan gunakan user 1 sebagai contoh.

# In[ ]:


user_id = 1


# ### 1. Mencari total jumlah film yang ada di data set

# In[ ]:


all_movies = pd.DataFrame(df['movie'].unique(), columns=['movie']) #jumlah film
all_movies


# In[ ]:


all_movies.nunique()


# ### 2. Mencari film apa saja yang sudah dirating oleh user 1

# In[ ]:


rated = df[df.userId == 1]
rated.head()


# In[ ]:


print('Film yang sudah dirating oleh user 1 = {}'.format(rated['movie'].nunique()))


# ### 3. Mencari film apa saja yang belum dirating oleh user 1

# In[ ]:


rated_list = pd.Series(rated['movie'])
not_rated = all_movies['movie'].isin( rated_list)
not_rated = all_movies[~not_rated]
not_rated.reset_index(inplace=True)
not_rated.drop(['index'], axis=1, inplace=True)
not_rated.head()


# In[ ]:


print('Film yang belum dirating oleh user 1 = {}'.format(not_rated['movie'].nunique()))


# Setelah mengetahui film yang belum dirating oleh user 1, sekarang kita bisa melakukan prediksi lalu rekomendasi.

# ## Prediksi dan Rekomendasikan

# Hitung score dari film yang belum dirating oleh user 1.

# In[ ]:


score = pd.DataFrame([model.predict(user_id, x).est for x in not_rated['movie']], columns=['rating'])
score.head()


# Kita satukan dataframenya lalu di sorting berdasarkan nilai rating.
# 
# Dan berikut adalah hasil akhir dari rekomendasi kita.

# In[ ]:


result = not_rated.join(score['rating']).sort_values(by='rating', ascending=False)
result.head(10)


# ## Rangkum kedalam Class
# 
# Rangkum code yang sudah kita buat dengan Class Agar lebih rapi dan mudah menggunakannya kembali.

# In[ ]:


class RecommenderSystem:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.all_movies = pd.DataFrame(self.df['movie'].unique(), columns=['movie'])
        self.model = None
        self.rated_list = None
        
    def fit(self):
        data = Dataset.load_from_df(self.df, Reader())
        trainset = data.build_full_trainset()        
        
        self.model = SVD()
        self.model.fit(trainset)
        
    def recommend(self, user_id, topk=10):
        rated = self.df[self.df['userId'] == user_id]
        self.rated_list = pd.Series(rated['movie'])
        not_rated = self.all_movies['movie'].isin(rated_list)
        not_rated = self.all_movies[~not_rated]
        not_rated.reset_index(inplace=True)
        not_rated.drop(['index'], axis=1, inplace=True)
        
        #[movie for movie in self.all_movies if movie not in watched]
        score = pd.DataFrame([self.model.predict(user_id, x).est for x in not_rated['movie']], columns=['rating'])
        #score = [self.model.predict(user_id, movie).est for movie in not_watched]
        
        #result = pd.DataFrame({"movie": not_rated, "pred_score": score}, )
        #result.sort_values("pred_score", ascending=False, inplace=True)
        result = not_rated.join(score['rating']).sort_values(by='rating', ascending=False)
        return result.head(topk)


# In[ ]:


recsys = RecommenderSystem("/kaggle/input/collaborative_filtering.csv")
recsys.fit()


# In[ ]:


recsys.recommend(user_id=1)


# Code kita berhasil ya hehe.
# 
# Mungkin itu dulu dari saya, jika ada pertanyaan atau masukan, bisa kalian tulis di kolom komentar atau boleh japri di ig saya: al.fath.terry
# 
# Dan mohon di upvote jika dirasa bermanfaat.
# 
# Terimakasih :)
