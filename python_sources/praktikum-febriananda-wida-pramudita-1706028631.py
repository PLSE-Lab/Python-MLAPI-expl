#!/usr/bin/env python
# coding: utf-8

# # <center>Demo Word Embedding</center>
# <center> Deadline : Kamis , 14 Mei 2020 , 16:00 WIB </center>
# <center>CSC4602354 - Natural Language Processing</center>
# <center>Gasal 2019 / 2020</center>

# ## Code Exercise

# Library preparation. Jangan sentuh.

# In[ ]:


c


# In[ ]:


get_ipython().system('ls /kaggle/input/praktikumword2vecnlp/idwiki.txt')


# In[ ]:


#Preparation

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().system('cp -r /kaggle/input/praktikumword2vecnlp/158 /kaggle/working/158')
get_ipython().system('git clone https://github.com/HIT-SCIR/ELMoForManyLangs')
get_ipython().system('pip install -e ELMoForManyLangs/')
get_ipython().system('cp ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json /kaggle/working/158/config.json')


# In[ ]:


file = open("/kaggle/working/158/config.json", "w")

teks = '{"seed": 1, "gpu": 3, "train_path": "/users4/conll18st/raw_text/Indonesian/id-20m.raw", "valid_path": null, "test_path": null, "config_path": "/kaggle/working/ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json", "word_embedding": null, "optimizer": "adam", "lr": 0.001, "lr_decay": 0.8, "model": "/users4/conll18st/elmo/src/final_models/id.model", "batch_size": 32, "max_epoch": 10, "clip_grad": 5, "max_sent_len": 20, "min_count": 3, "max_vocab_size": 150000, "save_classify_layer": false, "valid_size": 0, "eval_steps": 10000}'

file.write(teks)
file.close()


# In[ ]:


import ELMoForManyLangs.elmoformanylangs as elmo


# Persiapan sudah selesai. Untuk kepentingan latihan kali ini, kalian akan menggunakan Word2Vec yang sudah dilatih dengan bahasa Indonesia.

# In[ ]:


from gensim.models import Word2Vec

modelword2vec = Word2Vec.load("/kaggle/input/praktikumword2vecnlp/idwiki_word2vec_300.model")
import os
from IPython.display import FileLink
FileLink(r'/kaggle/input/praktikumword2vecnlp/idwiki.txt')


# In[ ]:


def plot(model, words):
    
    arr = np.empty((0,100), dtype='f')
    word_labels = []
    
    for word in words:
        wrd_vector = model[word]
        word_labels.append(word)
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
        
    # find tsne coords for 2 dimensions
    pca = PCA(n_components=2, copy=False, whiten=True)
    Y = pca.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


# Pertama-tama, mari kita bermain dengan vektor-vektor tersebut. Kita bisa mengakses dengan cara sebagai berikut

# In[ ]:


print(modelword2vec["ui"])


# Soal :
#     
# [1] Dimensi vektor Word2Vec dapat ditentukan saat proses training awal. Berapakah dimensi vektor word2vec yang kita gunakan sekarang?

# ~ Jawaban Teks Soal 1

# In[ ]:


# Jawaban kode Soal 1
print("dimensi dari vektor word2vec yang kita gunakan sekarang adalah:", modelword2vec["ui"].shape[0])


# Salah satu hal yang membuat Word2Vec spesial adalah vektor-vektor tersebut sebenarnya secara implisit menyimpan konteks dari kata tersebut. Cara termudah adalah dengan melakukan plotting terhadap vektor tersebut.

# In[ ]:


plot(modelword2vec, ["jakarta", "bandung" , "bekasi" , "serpong" ,# Barat Jawa
             "surabaya" , "malang" , "yogyakarta"  , # Timur Jawa
             "banjarmasin" , "balikpapan" , "samarinda" , # Kalimantan
             "medan" , "palembang" , "jambi" , # Sumatera
             "manado" , "gorontalo" , "palu"  , # Sulawesi
            "ambon" , "sofifi" , "tual" , # Maluku
             "fakfak" , "jayapura" , "mamuju" ]) # Papua


# Dari plot diatas, kita bisa lihat bahwa kota-kota tercluster secara geografi. meskipun tidak sempurna. Kota dari Barat Jawa di kanan atas. Timur Jawa di kanan bawah. Indonesia timur ada di sebelah kiri.  Sumatra & Kalimantan tercampur di tengah, tapi kemungkinan besar karena masalah visualisasi.
# 
# Uniknya, word2vec seperti sadar kalau Serpong itu bukan kota, tapi sebenarnya kecamatan, makanya dibuang ke atas.
# 
# Salah satu penjelasan mengapa ini terjadi karena kota yang terletak berdekatan lebih sering muncul bersama-sama. Sebagai contoh, wajar untuk membaca artikel yang membahas Jakarta dan Depok, tetapi langka untuk membaca artikel yang membahas Jakarta dan Fakfak. Jika dua buah kota sering muncul di kalimat yang sama, berati training dataset CBOW / Skipgramnya banyak yang sama. Ini mengakibatkan kota-kota tersebut mirip vektornya.
# 
# Soal :
# 
# [2] Dari visual di atas, saya bisa membuat hipotesis bahwa Word2Vec dapat digunakan untuk task "prediksi lokasi sebuah kota hanya dari namanya". Kita sudah belajar Pos-Tagging dan Syntatic Parsing. Buat sebuah visual yang membuktikan Word2Vec bisa digunakan untuk task lain. Untuk nomor 2, anda tidak harus meng-cover task Pos-Tagging dan Syntatic Parsing. Anda boleh pilih task NLP lain atau buat task lain dari kreativitas. 

# ~Jawaban Teks Nomor 2

# In[ ]:


# Jawaban kode nomor 2
plot(modelword2vec, ["sungai", "danau", "laut", "rawa", # tempat alami di perairan
                     "sabana", "stepa", "hutan", "gurun", "pegunungan", # tempat alami di daratan
                     "waduk", "kolam", # tempat buatan manusia di perairan
                     "kebun", "sawah", "pabrik", "rumah", # tempat buatan manusia di daratan
             ])


# > terlihat bahwa komponen di atas terkluster, 
# tempat alami di perairan cenderung berada di bawah kiri
# tempat buatan di perairan cenderung berada di bawah kanan
# tempat alami di daratan cenderung berada di atas kanan
# tempat buatan di perairan cenderung berada di atas kanan
# 

# Selain itu, kita bisa menggunakan model tersebut untuk mencari kata yang paling mirip berdasarkan seberapa mirip vektornya

# In[ ]:


modelword2vec.most_similar(positive = ["presiden"], topn=5)


# In[ ]:


modelword2vec.most_similar(positive = ["makan"], topn=5)


# Selain itu, kita bisa melakukan "vector composition" di mana kita melakukan penjumlahan dan pengurangan vektor kata untuk mencapai kata lain yang relasinya masih sama seperti Raja - Pria + Wanita = Ratu atau Jakarta - Indonesia + Inggris = London
# ![](https://i.imgur.com/l4Uawww.png)

# In[ ]:


modelword2vec.most_similar(positive = ["inggris", "jakarta"], negative = ["indonesia"], topn=5)


# Bagaimana hal itu bisa terjadi? Entah mengapa, Word2Vec bisa menyimpan hubungan tersebut dalam bentuk arah dan jarak antara dua vektor tersebut. Jika kita lihat di grafik di bawah, jika kamu tarik garis dari ibu kota ke negaranya masing-masing, garisnya hampir konsisten.

# In[ ]:


plot(modelword2vec, ["inggris" , "london",
                     "filipina" , "manila" ,
                    "rusia" , "moscow" , 
                    "jepang" , "tokyo",
                    "taiwan" , "taipei",
                    "kanada" , "ottawa"])


# Tentu saja kita dapat mencari seberapa mirip dua buah kata dari seberapa mirip vektornya.

# In[ ]:


modelword2vec.similarity('zebra' , 'refrigerator')
modelword2vec.similarity('zebra' , 'house')


# Word2Vec dapat digunakan untuk mencari kalimat yang "beda" sendiri dengan mencari vektor mana yang paling jauh dari lainnya

# In[ ]:


modelword2vec.doesnt_match(['jokowi' , 'sby' , 'suharto' , 'sule'])


# Soal:
# 
# [3] 
# 
# Soal 3 ada 2 versi. Cek nomor bundel Tugas Akhir yang kalian gunakan. Jika nomor bundel Tugas Akhir kalian adalah ganjil, gunakan versi 1. Jika nomor bundel Tugas Akhir kalian adalah genap, gunakan versi 2.
# 
# Versi 1 :
# Gunakan Word2Vec untuk menjawab soal TPA SBMPTN tersebut.
# 
# ![](https://i.imgur.com/xXBPdvD.png)
# 
# Versi 2 :
# Gunakan Word2Vec untuk menjawab soal TPA SBMPTN tersebut.
# 
# ![](https://www.zenius.net/blog/wp-content/uploads/2015/02/dompet.png)
# 
# 
# [4]  Fungsi model.similiarity menggunakan rumus cosine distance. Implementasikan sebuah fungsi yang menerima dua buah kata dan meng-return cosine distance antara vektor-vektor tersebut. Jika implementasinya benar, saat fungsi tersebut digunakan untuk menghitung raja dan presiden, jawabannya mendekati 0.3518753.

# ~ Jawaban teks nomor 3

# In[ ]:


# Jawaban kode nomor 3
plot(modelword2vec, [
    "semut", "gula",
    "harimau", "hutan",
    "tentara", "perang",
    "kerbau", "kandang",
    "burung", "langit",
    "penjara", "narapidana"
])


# In[ ]:


print('setelah diplot yang paling mirip arah dan jaraknya adalah kerbau:kandang')


# In[ ]:


# Jawaban kode nomor 4
def cosine_similarity(word1, word2):
    vec1 = modelword2vec[word1]
    vec2 = modelword2vec[word2]
    return (vec1.dot(vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cosine_similarity('raja', 'presiden')


# Kita bisa menggunakan Word2Vec sebagai fitur untuk kebutuhan klasifikasi. Sebagai contoh, kita bisa mengambil rata-rata Word2Vec setiap kata sebuah kalimat untuk membentuk "vektor kalimat" tersebut

# In[ ]:


# Ambil word2vec setiap kata
w2v = dict(zip(modelword2vec.wv.index2word, modelword2vec.wv.syn0))

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec['dan'])
        
    def tokenize(self, sentences):
        return [sentence.lower().split(" ") for sentence in sentences]

    
    def transform(self, X):
        # Ambil kata-katanya lalu rata-rata
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
vectorizer = MeanEmbeddingVectorizer(w2v)


# In[ ]:


# Task : Sentimen Analisis
# 0 negatif , 1 positif

train_teks = ["Saya sedih karena warung pasta ditutup" ,
              "Sekarang adalah waktunya untuk berbahagia dan bersyukur" , 
              "Bangun segan , mati tidak mau ketika menghadapi sprint" ,
              "OH MY GOD AKU DAPAT TANDA TANGAN LISA DARI BLACKPINK" ,
              "NLP itu seru !" ,
              "Gue bahagia karena keterima magang" ,
              "' Mampus aku bisnis aku bakal bangkrut ' , pikir CEO Traveloka" , 
              "Cacing di perut mencuri semua nutrisi penting"
             ]

train_y = [0 , 1, 0 ,1 , 1,  1, 0 , 0]

train_X = vectorizer.transform(vectorizer.tokenize(train_teks))

test_teks = ["Memang tidak salah untuk berharap , tapi aku harus tahu kapan berhenti" ,
              "Mengapur berdebu , kita semua gagal , ambil s'dikit tisu , bersedihlah secukupnya" , 
              "Ini adalah waktunya kamu untuk bersinar" ,
             "Kita akan berhasil menghadapi cobaan "
             ]

test_y = [0 , 0 , 1 , 1]

test_X = vectorizer.transform(vectorizer.tokenize(test_teks))


# Soal:
# 
# [5] Soal ini ada 2 versi. Cek NPM kalian. Jika NPM kalian ganjil, kerjakan versi pertama. Jika NPM kalian genap, kerjakan versi kedua.
# 
# Versi 1 :
# 
# Train model SVM Linear menggunakan data training dan testing di atas. Laporkan akurasi model. Anda boleh menggunakan model milik https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# 
# Versi 2 :
# 
# Train model Logistic Regression menggunakan data training dan testing di atas. Laporkan akurasi model. Anda boleh menggunakan model milik https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# 
# 
# Hint : Karena datanya sedikit, kita sebaiknya menggunakan model klasik. Tugas Akhir akan memiliki dataset yang jauh lebih besar, kita mungkin bisa menggunakan deep learning.

# Jawaban teks nomor 5

# In[ ]:


# Jawaban kode nomor 5
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

SVCpipe = Pipeline([('scale', StandardScaler()),
                   ('SVC',LinearSVC())])

# Gridsearch to determine the value of C
param_grid = {'SVC__C':[10**(i/8) for i in range(-24, 25)]}
clf = GridSearchCV(SVCpipe,param_grid,cv=4,return_train_score=True)
clf.fit(train_X,train_y)
print(clf.best_params_)
#linearSVC.coef_
#linearSVC.intercept_

bestlinearSVC = clf.best_estimator_
bestlinearSVC.fit(train_X,train_y)
bestlinearSVC.coef_ = bestlinearSVC.named_steps['SVC'].coef_
bestlinearSVC.score(test_X,test_y)


# **Materi tambahan!**
# Kalian hanya perlu mengerjakan lima soal tersebut. Materi di bawah adalah bonus pengetahuan =)
# 

# ![](https://media.giphy.com/media/1iqPlDCyyXeiQRvqWI/giphy.gif)

# ### Training Word2Vec dari awal

# Model Word2Vec yang kalian gunakan ditrain menggunakan data dari seluruh artikel di Wikipedia Indonesia. Kalian boleh menggunakan model ini untuk tugas kelompok NLP, tapi bagaimana jika ingin membuat model baru dari awal?

# In[ ]:


from gensim.test.utils import common_texts
from gensim.models import Word2Vec


# 1. Teks harus sudah ditokenisasi pada tingkat dokumen (sesuai kebijaksanaan sendiri apakah pada tingkat dokumen, paragraf, atau kalimat). Lalu teks tersebut ditokenisasi pada tingkat kata. (Lihat contoh di bawah)

# In[ ]:


common_texts


# 2. Lalu kita train. Akan tetapi, ada beberapa parameter yang perlu kita perhatikan:
# 
# 
# * size => ukuran vektor
# 
# * window => ukuran window
# 
# * min_count => berapa banyak kali suatu kata harus muncul sebelum disimpan
# 
# * epoch => berapa kali data ditrain ulang
# 
# * workers => berapa banyak thread yang akan digunakan
# 
# Tidak ada *heuristic* spesifik untuk mencari parameter yang tepat. Akan tetapi,
# 
# * Mayoritas paper yang menggunakan embedding umumnya ukurannya >= 100
# * Window umumnya memiliki ukuran 2 dan 3 , tetapi jika kita merasa datanya kurang banyak , kita bisa meningkatkan window ke 4 atau 5.

# In[ ]:





# In[ ]:


get_ipython().system('ls /kaggle/input/praktikumword2vecnlp/idwiki.txt')


# In[ ]:


get_ipython().system('wc -l /kaggle/input/praktikumword2vecnlp/idwiki.txt')
import nltk
common_texts = []
f = open('/kaggle/input/praktikumword2vecnlp/idwiki.txt')
counter = 0
for line in f:
    counter += 1
    if counter % 1000 == 0:
        print(counter * 100 / 392172, '%')
    common_texts.append(nltk.word_tokenize(line))
print(len(common_texts))


# In[ ]:


model = Word2Vec(common_texts, size=300, window=3, min_count=1, workers=4)
get_ipython().system('ls')
model.save("word2vec.300.model")
get_ipython().system('ls')


# In[ ]:


get_ipython().system('cp word2vec.300.model* /kaggle/input/')


# In[ ]:


get_ipython().system('ls /kaggle/')


# # Elmo
# 
# Word2Vec memiliki kelemahan. Sebuah kata mungkin memiliki definisi yang berbeda, tetapi word2vec untuk definisi yang berbeda tetaplah yang sama karena kata yang sama. Sebagai contoh, kata "kali" bisa berarti perkalian matematika atau sungai. Meskipun kali mungkin memiliki definisi yang berbeda, vector word2vec yang merepresentasikan tetap sama. Elmo menghasilkan embedding ala Word2Vec, tetapi embedding kata tersebut juga berdasarkan kata-kata di sebelahnya. Mari kita coba.

# In[ ]:


e = elmo.Embedder('/kaggle/working/158')


# In[ ]:


def encode_elmo(elmo,  kalimat):
    vektor = elmo.sents2elmo([kalimat.split(" ")])
    return vektor[0]


# In[ ]:


matematika = encode_elmo(e, "Tujuh kali dua sama dengan empat belas")[1]
sungai1 = encode_elmo(e, "Saya tinggal di samping kali Ciliwung")[4]
sungai2 = encode_elmo(e, "Indonesia Lawyers Club mempertanyakan kualitas kali yang menjadi water way")[5]
sekarang = encode_elmo(e, "Untuk kali ini dia yang kena batunya")[1]
perbandingan = encode_elmo(e, "Sejak korona , harga telur menjadi dua kali lipat")[7]
frekuensi = encode_elmo(e, "Saya sudah tidur lima kali")[4]


# In[ ]:


arr = np.empty((0,1024), dtype='f')
arr = np.append(arr, np.array([matematika]), axis=0)
arr = np.append(arr, np.array([sungai1]), axis=0)
arr = np.append(arr, np.array([sungai2]), axis=0)
arr = np.append(arr, np.array([sekarang]), axis=0)
arr = np.append(arr, np.array([perbandingan]), axis=0)
arr = np.append(arr, np.array([frekuensi]), axis=0)

pca = PCA(n_components=2, copy=False, whiten=True)
Y = pca.fit_transform(arr)

x_coords = Y[:, 0]
y_coords = Y[:, 1]

plt.scatter(x_coords, y_coords)

nama_label = ['matematika' , 'sungai1' , 'sungai2' , 
              'sekarang' , 'perbandingan' , 'frekuensi']
for label, x, y in zip(nama_label, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.show()


# Dari contoh di atas, embedding "kali" untuk definisi yang berbeda terpisah, tetapi sungai1 dan sungai2, kalimat yang definisi "kali" sebagai "sungai", vektornya berdekatan.
