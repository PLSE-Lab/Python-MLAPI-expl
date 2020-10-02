#!/usr/bin/env python
# coding: utf-8

# <h2>**Mengimport Library**</h2>

# In[ ]:


# Mengimport library yang dibutuhkan program
import pandas as pd # Library untuk menangani data
import numpy as np # library numpy array
from sklearn.model_selection import train_test_split # fungsi untuk membagi dataset menjadi data training dan data testing
from sklearn.tree import DecisionTreeClassifier # fungsi untuk menggunakan model decision tree dalam klasifikasi
from sklearn import metrics # untuk kalkulasi akurasi model
import joblib # Library untuk mengeksport model yang sudah di training ke dalam bentuk file


# <h2>**Membaca Data**</h2>

# In[ ]:


# Path dataset yang ada (datapanas.csv) disimpan kedalam variabel skripsiku
skripsiku = '../input/datapanas/datapanas_processed_lagi.csv'

# Membaca dataset dengan dungsi read_csv pada library pandas (pd)
# pada path yang tersimpan di variabel skripsiku
# data csv yang ada disimpan di variabel hotSpotData
hotSpotData = pd.read_csv(skripsiku)

# hotSpotData2 = pd.read_csv(skripsiku)
# frames = [hotSpotData, hotSpotData2]
# seluruhData = pd.concat(frames)

# Untuk melihat rangkuman dari dataset yang tersimpan pada variabel hotSpotData.
# Rangkuman yang dimaksud itu sebenernya rangkuman secara statistik (kaya count atau jumlah data 
# pada kolom tsb, mean atau rata2 pada kolom tsb, std atau standar deviasi pada kolom tsb, 
# 25% itu kuartil bawah pada kolom tsb, 50% itu median pada kolom tsb, 75% kuartil atas pada kolom tsb, 
# max itu nilai maksimum pada kolom tsb, min itu nilai minimum pada kolom tsb)
hotSpotData.describe()
# seluruhData.describe()

# Catatan: Rangkuman yang diliat dengan fungsi describe() itu cuman untuk kolom yang tipenya angka
# makanya kolom2 kaya provinsi sama kota engga ada di rangkuman di bawah


# In[ ]:


# untuk ngeliat 5 data pertama dari hotSpotData (fungsi head() buat ngeliat 5 data pertama))
hotSpotData.head()


# <h2>**Menangani Nilai Non-Angka dan Menangani Nilai Kosong**</h2>
# Kalo seandainya didataset ada nilai yang non-angka dan nilai kosong, library sklearn enggak bisa memproses data tersebut dan akan memberikan error. Jadi sebelum diolah lebih lanjut data harus dilakukan praproses dulu, yaitu menangani nilai non-angka dan nilai kosong.
# 
# Untuk menangani nilai non-angka, library pandas punya fungsi namanya get_dummies(). Untuk menangani nilai kosong, library sklearn punya fungsi namanya SimpleImputer().
# 
# <h3>**Cara Kerja Pandas get_dummies()**</h3>
# ![Table Full](http://miro.medium.com/max/700/1*psCS6W7FNKJ_auc9fdnA1g.png)
# 
# Di tabel atas, liat ada data non-angka di kolom sx. Nah pas fungsi get_dummies dipake, pandas bakal nambahin kolom baru (disebut kolom dummy, makanya nama fungsinya get_dummies) seperti tabel di bawah.
# 
# ![Tabel Dummy](https://miro.medium.com/max/700/1*HfhgywtwXtxVcUmQuyu-_w.png)
# 
# Dia bakal buat kolom female sama male, kalo misalnya di kolom sx nilainya male maka di kolom male di kasih 1 dan kolom female 0 dan berlaku sebaliknya.

# In[ ]:


# Menentukan kolom mana aja yang dipake untuk fitur-fitur dalam prediksi.
# Disini kolom CurahHujan engga dipake karena CurahHujan itu
# kolom yang akan diprediksi.
# Kolom-kolom yang dipake sebagai fitur untuk prediksi itu X
# Kolom yang akan diprediksi itu Y
features = ['Provinsi', 'Kota', 'Long', 'Lat', 
            'Hari', 'FFMC', 'DMC', 'DC', 'ISI',
            'Temperatur', 'KecepatanAngin']

# variabel x cuman nyimpen nilai kolom-kolom yang ditentuin 
# di variabel features dari hotSpotData
x = hotSpotData[features]

# ngeliat 5 data pertama dari variabel x (fungsi head() buat ngeliat 5 data pertama))
# x ini belum di praproses.
x.head()


# In[ ]:


#Variabel y hanya menyimpan nilai-nilai kolom Luas Area dari hotSpotData
y = hotSpotData['target']

#meelihat 5 data pertama dari variabel y.
y.head()


# In[ ]:


#mengatasi nilai non\-angka pada kolom\-kolom yang tersimpan di variabel x
x = pd.get_dummies(x)
x.head()


# Bisa diliat di tabel di atas ada kolom tambahan baru seperti kolom Provinsi_BALI, provinsi_BANGKA-BELITUNG, Hari_Kamis, dll. Nah kolom-kolom tambahan itu yang disebut sebagai kolom dummy.

# <h2>**Membagi Data Menjadi Data Training dan Data Testing**</h2>

# In[ ]:


# Untuk membagi data menjadi data training dan testing
# dipake fungsi train_test_split
# train_test_split butuh 2 parameter utama yaitu
# nilai kolom-kolom untuk melakukan prediksi (X) dan
# nilai kolom yang diprediksi (Y)
# parameter yang lain seperti shuffle dan test_size itu optional
# shuffle dipake supaya pengambilan datanya engga random
# test_size dipake untuk nentuin seberapa banyak data yang mau dipake untuk testing
# test_size kalo ga ditentuin defaultnya 0.25 (jd nantinya data trainingnya 0.75,
# data testingnya 0.25)
# trainX sama trainY itu data training
# valX sama valY itu data testing
trainX, valX, trainY, valY = train_test_split(
    x,y)
print('Data Training')
print(trainX)
print('Data Test')
print(trainY)

print('data')
print (valX)
print('datacoba')
print(valY)
# Ini kalo bagi data manual
# trainX = x[:299]
# trainY = y[:299]
# valX = x[299:]
# valX = y[299:]

# untuk ngeliat apakah benar data yang diambil 0.75 training dan 0.25 testing
# dilakukan print pada variabel x (data utuh) dan data training beserta data testing

# print data training
# karena keseluruhan data ada 400 baris. berarti 0.75nya dipake training
# perhitungannya 0.75 * 400 - 1 = 299 data yang dipakai untuk training
# untuk memastikan kebenarannya diprint data di x dari index 0 sampai 300
#print("Sumber data sebelum train_test_split (index 0-298)")
# python sistem indexing arraynya explisit artinya walau ditulis 299, 
# 299 itu tidak termasuk yang di print (cuman sampe 298)
#print(x[:299])
#print()
#print("Data Training")
# ini print data yang ada di variabel trainX (hasil train_test_split)
#print(trainX)


# In[ ]:


# Ini buat ngeliat data test
# data test 0.25 berarti 0.25 * 400 + 1 = 101 data
print("Sumber data sebelum train_test_split (index 299-399)")
print(x[299:])
print()
print("Data Testing")
print(valX)


# <h2>**Menginisialisasi Model**</h2>

# In[ ]:


# Model yang dipakai adalah Decision Tree untuk regresi jadi library yang dipake
# itu DecisionTreeRegressor-nya sklearn
# random_state itu parameter supaya Decision Tree nya ga berubah-berubah saat
# programnya dijalanin ulang
# max_leaf_nodes itu parameter untuk nentuin nilai maksimal leaf pada treenya berapa,
# ditentuin karena kalau terlalu banyak leafnya biasanya malah makin tidak akurat modelnya
hotspotPredictorModel = DecisionTreeClassifier(random_state=1, max_leaf_nodes=100)


# <h2>**Melatih Model (Training)**</h2>

# In[ ]:


# Model yang tadi difit (istilah lainnya disuruh belajar)
# menggunakan data training trainX dan trainY
hotspotPredictorModel.fit(trainX, trainY)

# Nah terus setelah modelnya di training, modelnya di test akurasinya
# dengan menggunakan data test valX dan valY
valPredictions = hotspotPredictorModel.predict(valX)
print("Akurasi = ", metrics.accuracy_score(valY, valPredictions))


# In[ ]:


besar_kecil = np.where(valPredictions == 1, 'BESAR', np.where(valPredictions == 2, 'SEDANG', 'KECIL'))
result = pd.DataFrame({'hasil_prediksi':valPredictions, 'keterangan':besar_kecil})
print(result)


# In[ ]:


data_test_satu_doang = valX.iloc[9,:]
nilai_y_asli = [valY.iloc[9]]
predict2 = hotspotPredictorModel.predict([data_test_satu_doang])
print('nilai hasil prediksi = ', predict2)
print('nilai asli = ', nilai_y_asli)


# In[ ]:



joblib.dump(hotspotPredictorModel, 'my_model.pkl', compress=9)

prediction = pd.DataFrame(valPredictions, columns=['HASIL_PREDIKSI']).to_csv('prediction.csv')
result = result.to_csv('hasil.csv')
train = pd.DataFrame(trainX).to_csv('train.csv')

