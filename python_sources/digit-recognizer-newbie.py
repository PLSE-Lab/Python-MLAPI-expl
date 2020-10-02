#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# # Pendahuluan
# 
# Notebook ini dibuat dengan tujuan untuk pembelajaran saya. Saya mengikuti buku Deep Learning with Python, yang kebetulan juga menggunakan dataset Digit Recognizer.

# # Exploratory Data Analysis
# 
# Sebelumnya kita lihat data-nya terlebih dahulu. Data yang dimaksud adalah `test.csv` dan `train.csv`. Kita mulai dari data `train.csv`.

# In[ ]:


train_fname = '../input/digit-recognizer/train.csv'

df_train = pd.read_csv(train_fname)
df_train.head()


# In[ ]:


df_train.shape


# In[ ]:


df_train['label'].unique()


# In[ ]:


fig, ax = plt.subplots()

hist, bin_edges  = np.histogram(df_train['label'])

# atur posisi x-label
bin_edges = np.ceil(bin_edges)
ax.set_xticks(bin_edges[:-1])

# tambah title
ax.set_title('Histogram')
ax.set_xlabel('Digits')

# plot histogram sebagai plot bar
ax.bar(bin_edges[:-1], hist);


# **Komentar**:
# * Ukuran dataframe adalah 42000 x 785. Artinya, jumlah data yang kita miliki ada 42000 buah (untuk data training) dan jumlah fiturnya ada 784 buah (1 kolom untuk label).
# * Label data berupa angka integer 0 - 9. Artinya, data ini adalah multi-class.
# * Karena ini adalah data citra 8-bit, maka nilai piksel berada di-range 0-255.
# * Berdasarkan plot histogram, terlihat bahwa setiap kelas ternyata tidak seimbang jumlah datanya.

# Selanjutnya, kita akan melihat data yang ada di `test.csv`.

# In[ ]:


test_fname = '../input/digit-recognizer/test.csv'

df_test = pd.read_csv(test_fname)
df_test.head()


# In[ ]:


df_test.shape


# **Komentar**:
# * Ukuran dataframe adalah 28000 x 785. Artinya, jumlah data yang kita miliki ada 42000 buah (untuk data training) dan jumlah fiturnya ada 784 buah.
# * Tidak ada kolom label.

# Selanjutnya, kita lihat `sample_submission.csv`

# In[ ]:


sample_fname = '../input/digit-recognizer/sample_submission.csv'

df_sample = pd.read_csv(sample_fname)
df_sample.head()


# In[ ]:


df_sample.shape


# **Komentar**:
# * Dataframe yang kita kumpulkan harus memiliki ukuran 28000 x 2. Dua kolom terakhir berisi `ImageId` dan `Label` yang diprediksi.

# # Pra-pemrosesan

# Kita akan mengubah tipe data dataframe menjadi ndarray.

# In[ ]:


y_train = df_train['label'].to_numpy()

del df_train['label']
X_train = df_train.to_numpy()

X_test = df_test.to_numpy()


# In[ ]:


y_train.shape, X_train.shape, X_test.shape


# Perhatikan bahwa fitur-fitur `X_train` maupun `X_test` adalah nilai piksel yang sudah di-*flatten* (bukan 2D). Jika ingin menggunakan input citra 2D, kita bisa me-*reshape* `X_train` dan `X_test`, namun itu diluar *scope* dari notebook ini.
# 
# Tahap selanjutnya adalah menormalisasi data `X_train` dan `X_test`. Seperti yang sebelumnya sudah saya tulis, range nilai piksel citra adalah 0-255. Namun, neural network akan bekerja optimal jika nilai input-nya berada di range 0-1. 

# In[ ]:


X_train = X_train / 255
X_test = X_test / 255


# Kita transformasikan `y_train` ke dalam bentuk representasi one-hot encoding.

# In[ ]:


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)


# # Desain Arsitektur Deep Learning

# Kita akan menggunakan satu buah FC-layer. FC singkatan dari Fully Connected, artinya setiap neuron pada layer tersebut terkoneksi dengan neuron-neuron pada layer sebelum dan sesudahnya.

# In[ ]:


from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[ ]:


network.summary()


# # Fase Training

# Kita kompilasikan model-nya:

# In[ ]:


network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# Kemudian, kita latih model tersebut menggunakan `X_train`.

# In[ ]:


network.fit(X_train, y_train, epochs=5, batch_size=128,
            validation_split=0.2);


# # Prediksi X_test

# Terakhir, kita buat `y_test` berisi prediksi model terhadap `X_test`.

# In[ ]:


y_test = network.predict_classes(X_test)


# In[ ]:


y_test.shape


# In[ ]:


df_pred = pd.DataFrame({
    'ImageId': range(1, y_test.shape[0] + 1),
    'Label': y_test
})


# In[ ]:


df_pred.head()


# In[ ]:


df_pred.shape


# In[ ]:


df_pred.to_csv('submission.csv', index=False)


# In[ ]:




