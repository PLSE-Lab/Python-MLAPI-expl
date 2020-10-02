#!/usr/bin/env python
# coding: utf-8

# <h1>Soal 1: Pemahaman Machine Learning</h1>
# 
# Jawab/Kerjakan pertanyaan/perintah di bawah ini dengan bahasa kalian sendiri:
# 
# - Apa itu machine learning?
# - Apa itu data feature dan data target?
# - Apa Perbedaan Supervised Learning dan Unsupervised Learning?
# - Apa Jenis2 yang ada di dalam Supervised Learning? Jelaskan Perbedaannya!
# - Apa perbedaan Hyperparameter dan Parameter?
# - Sebutkan Langkah-langkah dalam mengaplikasikan algoritma apapun dalam machine learning!

# 1. Machine learning adalah ilmu atau studi yang mempelajari tentang algoritma dan model statistik yang digunakan oleh sistem komputer untuk melakukan task tertentu tanpa instruksi eksplisit, sehingga mesin dapat mempelajari data - data yang ada untuk membuat suatu prediksi terhadap data di masa depan
# 
# 2. Data feature merupakan data yang berisi variabel independent, ia berfungsi sebagai predictor yang berperan dalam melakukan prediksi, sedangkan data target merupakan data yang berisi variabel dependent yang digunakan sebagai label dimana label merupakan hasil dari prediksi.
# 
# 3. Supervised learning memiliki data label sehingga outpunya jelas, sedangkan unsupervised learning tidak memiliki kolom label sehingga outputnya tidak jelas
# 
# 4. Jenisnya :
# - Klasifikasi --> outputnya berupa kelas label
# - Regresi --> outputnya berupa kuantitas yang berkelanjutan
# 
# 5. Hyperparameter didefinisikan di awal pembuatan model, sedangkan parameter dihasilkan di akhir dan terdapat pada model di awal tadi
# 
# 6. Tahapannya :
# - Pilih model atau algoritma yang akan digunakan
# - Definisikan hyperparameter
# - Pisahkan antara data feature dan data target
# - Tulis metode.fit() untuk mempelajari data
# - Aplikasikan model dengan .predict() atau transform() dan .predict()

# ---
# 

# <h1>Soal 2: Pemahaman Machine Learning</h1>
# 
# Pelajarilah secara garis besar suatu model/algoritma machine learning, kemudian aplikasikan untuk membuat model bagi data (variable x, y) di bawah ini. Kemudian buatlah prediksi terhadap data training dan data baru di interval 20-30.
# 
# - Plot data asli, data hasil prediksi terhadap data training, data hasil prediksi terhadap data baru
# - Tunjukan beberapa parameter yang dimiliki model yang telah kalian buat

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 20 * rng.rand(50)
y = x**2 + 2 * x + - 1 + rng.randn(50)

# data yang akan di prediksi
x_new = np.arange(20, 30, 0.5)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y)
ax.set_xlabel('X-Value')
ax.set_ylabel('Y-Value')
plt.show()


# Contoh jawaban hasil pembuatan model dengan algoritma DecisionTreeRegressor (jawaban tidak harus sama):
# 
# ![alt text](https://drive.google.com/uc?id=1R0mGnMp3DqEsHDiH2_IL4XSS2Roew0wo)

# Contoh jawaban parameters (tidak harus sama):
# 
# - Feature Importance : array([1.])
# - n features : 1
#   
# Tree Graph :
# 
# ![alt text](https://drive.google.com/uc?id=1GM0ba_qjm_e5oLhE9v6pVt2uRBBW6FH9)

# In[ ]:


# Code here
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 20 * rng.rand(50)
y = x**2 + 2 * x + - 1 + rng.randn(50)

# data yang akan di prediksi
x_new = np.arange(20, 30, 0.5)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y)
ax.set_xlabel('X-Value')
ax.set_ylabel('Y-Value')
plt.show()


from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

dtr = DecisionTreeRegressor(max_depth=4, min_impurity_decrease=0.0, min_impurity_split=None, 
                            random_state=None, splitter='random')

x_train = x.reshape(-1,1)
y_train = y
x_test = x_new.reshape(-1,1)

dtr.fit(x_train, y_train)
y_test = dtr.predict(x_test)
y_predict = dtr.predict(x_train)

fig = plt.subplots(figsize=(12, 8))
plt.scatter(x, y, c='b', label = 'Data Asli')
plt.scatter(x_train, y_predict, c='y', label = 'Prediksi Terhadap Data Training')
plt.plot(x_test, y_test, c='g', label = 'Prediksi Terhadap Data Baru')
plt.legend()
plt.show()
tree.plot_tree(dtr);


# In[ ]:


dtr.feature_importances_


# In[ ]:


rgr.n_features_


# In[ ]:


get_ipython().run_line_magic('pinfo', 'DecisionTreeRegressor')


# In[ ]:




