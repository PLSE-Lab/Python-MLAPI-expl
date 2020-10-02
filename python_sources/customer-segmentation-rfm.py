#!/usr/bin/env python
# coding: utf-8

# # Segmentasi Customer (Metode RFM)

# Halo teman-teman, bagaimana kabar kalian semua? saya harap kalian semua baik-baik saja ya. Kali ini saya ingin mendemonstrasikan cara mensegmentasikan customer (Metode RFM) menggunakan library Lifetimes dan juga K-means. 
# 
# tapi sebelum masuk ke RFM nya, saya ingin menunjukkan terlebih dahulu tipe-tipe segmentasi yang ada:

# ### Tipe-Tipe Segmentasi:
# 
# - Demographic: variabelnya: usia, gender, status pernikahan, pekerjaan.
# - Geographic: variabelnya: lokasi, wilayah, pedesaan.
# - Behavioral: variabelnya: pengeluaran, kebiasaan customer, penggunaan produk, produk yang dibeli sebelumnya.
# - Psychographic: variabelnya: status sosial, gaya hidup, karakter personal.

# Oh ya, sebetulnya ngga harus pake K-means sih, kalian bisa pakai algoritma clustering apapun, berhubung data penting yang akan kita modelkan adalah numerical semua (Recency, Frequency, Monetary Value, dan Invoicedate adalah data numerical ), dan juga K-means ini adalah algoritma yang simpel, jadi ya saya pakai K-means deh. 
# 
# Alasan lain yang menguatkan saya untuk memilih K-means adalah karena saya teringat pesan dari kak Wira: "always start with simple model". Siapa itu kak Wira? beliau adalah instruktur saya, hehe. Kenapa harus start with simple model? karena.... alasannya banyakkk, dan saya tidak akan menjabarkannya disini (mungkin akan saya ceritakan dilain notebook). Oke lanjut ke RFM nya ya.
# 
# RFM adalah metode yang digunakan untuk mensegmentasikan customer berdasarkan kebiasaan perilaku pembeliannya. RFM ini termasuk kedalam tipe segmentasi Behavioral. Berikut adalah definisi masing-masing komponen dari RFM:
# 
# - Recency: Recency adalah jarak waktu tidak aktifnya customer setelah mereka melakukan pembelian yang paling terbaru(recent). *jika customer hanya melakukan satu kali pembelian, maka recency-nya adalah 0. 
# 
# 
# - Frequency: Angka pembelian *ulang*/ repetisi pembelian yang dilakukan customer. Atau sama dengan total pembelian dikurangi satu.
# 
# 
# - Mean Moneter Value: rata-rata nilai uang yang telah dihabiskan oleh pelanggan dalam periode waktu tertentu. ini sama dengan jumlah semua pembelian customer dibagi dengan angka total pembelian (berapa kali customer melakukan pembelian).
# 
# RFM dapat digunakan untuk mengenal basis customer dengan baik. Salah satu  contoh pemanfaatnya adalah, jika sebelumnya kalian memberikan treatment yang sama terhadap semua jenis customer yang ada, maka dengan RFM ini kalian bisa lebih spesifik dalam mengidentifikasi segmen yang ada, lalu menerapkan strategi pendekatan yang berbeda antar segmen. Agar lebih jelas, yuk ikuti terus kelanjutannya.

# ## Import Packages dan Data

# Kita akan gunakan library Lifetimes terlebih dahulu, Lifetimes adalah library Python yang bisa digunakan untuk analisa customer, selain itu Lifetimes juga bisa membuat model RFM berdasarkan data transaksional, dan fitur inilah yang akan kita gunakan.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install lifetimes')
get_ipython().system('pip install jcopml')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.utils import summary_data_from_transaction_data
from jcopml.plot import plot_missing_value


# Import data.

# In[ ]:


df = pd.read_csv("/kaggle/input/onlineretail/OnlineRetail.csv",error_bad_lines=False, encoding='unicode_escape')
df.head()


# ## Quick EDA (Exploratory Data Analysis)

# Terdapat 4 kolom penting yang dapat kita gunakan untuk membuat segmentasi customer RFM, yaitu:
# 
# - Quantity
# - UnitPrice
# - CustomerID
# - InvoiceDate
# 
# Kita coba lakukan beberapa pengecekkan untuk mengetahui gambaran secara umum kondisi data kita.

# ### Data Types

# In[ ]:


df.dtypes


# ### Plot Missing Value

# In[ ]:


plot_missing_value(df, return_df = True)


# Sangat disayangkan value CustomerID banyak yang hilang, padahal library Lifetimes bergantung pada CustomerID untuk membuat segmentasi RFM. Mau tidak mau nantinya kita akan lakukan drop out terhadap value CustomerID yang bolong. Meskipun nantinya Lifetimes akan melakukannya secara otomatis, namun kita akan melakukannya secara manual.

# ### Describe Numerical Data

# In[ ]:


df.describe()


# Dari snapshot diatas, hal pertama yang menarik perhatian saya adalah value min dan max dari Quantity, karena valuenya simetris, min -80995 dan max 80995, ada apa gerangan? kira-kira kejadian apa yang menyebabkan hal ini terjadi?
# 
# Lalu ada juga UnitPrice dengan value negatif, apakah ini hutang?
# 
# Kita lanjut identifikasi outliers dan buat scatter plot untuk memahami hubungan antar keduanya

# ### Identifikasi Outliers

# Kita identifikasi outliers menggunakan IQR score, rumusnya adalah:
# - IQR = Q3 - Q1
# - lower bound = Q1 - 1.5 * IQR
# - upper bound = Q3 + 1.5 * IQR 

# #### Outliers Quantity

# In[ ]:


q1_quan = df['Quantity'].quantile(0.25)
q3_quan = df['Quantity'].quantile(0.75)
iqr_quan = q3_quan - q1_quan
lb_quan = float(q1_quan) - (1.5 * iqr_quan)
ub_quan = float(q3_quan) + (1.5 * iqr_quan)

print('Q1 = {}'.format(q1_quan))
print('Q3 = {}'.format(q3_quan))
print('IQR = Q3 - Q1 = {}'.format(iqr_quan))
print('lower bound = Q1 - 1.5 * IQR = {}'.format(lb_quan))
print('upper bound = Q3 + 1.5 * IQR = {}'.format(ub_quan))


# #### Outliers UnitPrice

# In[ ]:


q1_unit = df['UnitPrice'].quantile(0.25)
q3_unit = df['UnitPrice'].quantile(0.75)
iqr_unit = q3_unit - q1_unit 
lb_unit = float(q1_unit) - (1.5 * iqr_unit)
ub_unit = float(q3_unit) + (1.5 * iqr_unit)

print('Q1 = {}'.format(q1_unit))
print('Q3 = {}'.format(q3_unit))
print('IQR = Q3 - Q1 = {}'.format(iqr_unit))
print('lower bound = Q1 - 1.5 * IQR = {}'.format(lb_unit))
print('upper bound = Q1 - 1.5 * IQR = {}'.format(ub_unit))


# ### Hubungan antara Quantity dan UnitPrice

# In[ ]:


sns.scatterplot(df['UnitPrice'], df['Quantity'])
plt.title('Quantity x UnitPrice', fontsize = 20);


# Setelah di plot, sekarang makin terlihat jelas hubungan antar kedua kolom. 
# 
# Data kita memiliki beberapa outliers ekstrim: 
# 
# - titik kelompok paling atas menggambarkan customer yang membeli banyak barang dengan harga yang tidak terlalu mahal, titik kelompok yang paling bawah adalah customer yang banyak .... mengembalikan barang? apakah ini retur?. Jika dicermati, bentuknya terlihat simetris dengan titik kelompok data yang paling atas. Hmm.. kira-kira apa ya maksudnya? menurut analisa saya, kemungkinan besar itu adalah Reseller, karena jika dilihat dari Quantitynya, tidak mungkin orang secara pribadi membeli barang sebegitu banyaknya. Reseller tersebut kemudian secara rutin mengambil barang dagangan dengan item yang sama, kemudian mengembalikan item yang tidak laku. Namun untuk membuktikan benar tidaknya, lebih baik kita tanyakan kepada Data Analis mengenai arti dari data itu, atau jangan-jangan ada kesalahan input? siapa tau. Dan sayang sekali penyedia data kita tidak menyertakan nomenklatur yang bisa menjelaskan data secara lebih detil.
# 
# 
# - Untuk titik kelompok yang paling kiri, saya tidak faham artinya apa, apakah itu pribadi yang berhutang. Sekali lagi cara terbaik untuk mengetahui kejelasan data kita adalah dengan menanyakannya kepada Data Analis.
# 
# 
# - Untuk titik kelompok yang paling kanan, itu adalah customer yang membeli barang dengan harga yang mahal, kemungkinan besar barang itu digunakan untuk diri sendiri karena Quantitynya tidaklah ekstrim. Cukup normal menurut saya, tidak begitu aneh.
# 
# Secara umum data kita berisikan customer yang beraneka ragam, ada yang membeli barang dengan rentang harga yang luas namun Quantitynya rendah (kemungkinan besar digunakan untuk pribadi), dan ada juga customer yang membeli dengan Quantity tinggi namun rentang harga yang rendah(kemungkinan adalah Reseller)
# 
# Tapi saya penasaran bagaimana bentuk scatter plotnya jika tanpa outliers, yuk kita cek dulu. 

# In[ ]:


dx = df[df['Quantity']>0] #hilangkan value negatif
dy = df[df['UnitPrice']>0] #hilangkan value negatif

filtered_quantity = dx.query('(@q1_quan - 1.5 * @iqr_quan) <= Quantity <= (@q3_quan + 1.5 * @iqr_quan)')
filtered_unitprice = dy.query('(@q1_unit - 1.5 * @iqr_unit) <= UnitPrice <= (@q3_unit + 1.5 * @iqr_unit)')

sns.scatterplot(filtered_unitprice['UnitPrice'], filtered_quantity['Quantity'])
plt.title('Quantity x UnitPrice', fontsize = 20);


# Secara umum perseberannya terlihat sama saja, namun jika diperhatikan dengan seksama terlihat kalau tingkat kepadatan di titik sebelah kanan atas mulai mengurang, artinya semakin tinggi UnitPrice semakin sedikit tingkat Quantity nya.
# 
# Namun plot diatas kurang begitu jelas gambarannya, biar lebih pasti, coba kita plot dengan scope yang lebih jauh.

# In[ ]:


q1_quan_custom = df['Quantity'].quantile(0.5)
q3_quan_custom = df['Quantity'].quantile(0.95)
iqr_quan_custom = q3_quan_custom - q1_quan_custom

q1_unit_custom = df['UnitPrice'].quantile(0.5)
q3_unit_custom = df['UnitPrice'].quantile(0.95)
iqr_unit_custom = q3_unit_custom - q1_unit_custom

dx = df[df['Quantity']>0] #hilangkan value negatif
dy = df[df['UnitPrice']>0] #hilangkan value negatif

filtered_quantity = dx.query('(@q1_quan_custom - 1.5 * @iqr_quan_custom) <= Quantity <= (@q3_quan_custom + 1.5 * @iqr_quan_custom)')
filtered_unitprice = dy.query('(@q1_unit_custom - 1.5 * @iqr_unit_custom) <= UnitPrice <= (@q3_unit_custom + 1.5 * @iqr_unit_custom)')

sns.scatterplot(filtered_unitprice['UnitPrice'], filtered_quantity['Quantity'])
plt.title('Quantity x UnitPrice', fontsize = 20);


# ### Distribusi Quantity dan UnitPrice

# In[ ]:


sns.distplot(df['Quantity'])
plt.title('Distribusi Quantity', fontsize = 20)
plt.xlabel('Quantity')
plt.ylabel('count');


# In[ ]:


sns.distplot(df['UnitPrice'])
plt.title('Distribusi Unit price', fontsize = 20)
plt.xlabel('Unit Price')
plt.ylabel('count');


# ### Plot 5 negara Terbesar

# Plot 5 negara terbesar yang melakukan pembelian retail secara online (berbasis di UK)

# In[ ]:


x = df['Country'].value_counts().head(5)
sns.barplot(x = x.values, y = x.index, )
plt.title('5 negara terbesar', fontsize = 20)
plt.xlabel('Count')
plt.ylabel('Nama Negara');


# Berikut detil jumlah negara dan jumlah transaksi yang terjadi 

# In[ ]:


x = df['Country'].nunique()
print("Terdapat total {} negara".format(x))

country = pd.DataFrame(df['Country'].value_counts()).reset_index()
country.columns = ['Negara', 'Jumlah Transaksi']
country


# Setelah melakukan EDA sekarang saatnya kita memodelkan data menjadi RFM. Yang pertama saya akan melakukannya dengan library Lifetimes, yang kedua saya akan menggunakan algoritma K-means.

# # Segmentasi Customer menggunakan library Lifetimes

# Secara garis besar, berikut adalah step yang dilakukan dalam membuat segmentasi customer menggunakan Lifetimes:
# 
# - 1. Persiapkan data
# - 2. Buat RFM
# 
# Kita hanya akan melakukan segmentasi terhadap United Kingdom saja (karena datanya yang paling banyak, hehe)
# 
# Sebelum melakukan permodelan, kita persiapkan datanya terlebih dahulu menjadi format dataframe yang di inginkan Lifetimes.

# ## Persiapkan Data

# Step 1. Buang negara selain UK.

# In[ ]:


df = df[df['Country'] == 'United Kingdom']
df.head()


# Step 2. Transform InvoiceDate ke python object datetime

# In[ ]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.head()


# Step 3. Buang value dari kolom 'CustomerID' yang datanya bolong.
# 
# sebetulnya kita bisa saja melakukan input data kosong tersebut dengan data dummy. tapi kali ini kita akan drop out saja. Step drop out ini akan mengurangi jumlah data lumayan signifikan.

# In[ ]:


df = df[~df['CustomerID'].isna()]
df.head()


# Step 4. Buang value dari kolom 'Quantity' dan 'UnitPrice' yang memiliki value 0 dan negatif.

# In[ ]:


df = df[df['Quantity']>0]
df = df[df['UnitPrice']>0]
df.head()


# Step 5. Buat kolom Revenue 
# 
# Definisi Revenue dari Lifetimes adalah:
# - Revenue = UnitPrice * Quantity

# In[ ]:


df['Revenue'] = df['Quantity'] * df['UnitPrice']
df.head()


# Step 6. Persiapkan format dataframe yang di inginkan Lifetimes.
# 
# Buat dataframe baru, agregatkan semua data berdasarkan InvoiceNo dan InvoiceDate.

# In[ ]:


orders = df.groupby(['InvoiceNo', 'InvoiceDate']).sum().reset_index()
orders.head()


# Data sudah siap, sekarang waktunya lakukan segmentasi

# ## Buat RFM Model

# Berikut adalah langkah-langkah dalam membuat RFM model:
# 
# - Step 1, hitung RFM Value
# - Step 2, hitung RFM Score
# - Step 3, hitung score RFM secara keseluruhan
# - Step 4, labeling
# 
# Kalau kalian bingung kenapa ada RFM Value dan RFM Score? apa bedanya? RFM Value adalah nilai dari RFM itu sendiri, sedangkan RFM Score adalah Score yang diberikan kepada RFM Value. agar lebih jelas, ikuti saja dulu notebook ini. Oke kita lanjut ya.

# ### Step 1: Hitung RFM Values

# Di step ini Lifetimes akan membuatkan dataframe baru secara otomatis berdasarkan dataframe yang sudah kita persiapkan sebelumnya.
# 
# outputnya terdapat kolom frequency, recency, T, dan monetary_value. 
# 
# Untuk saat ini abaikan saja kolom CustomerID dan T, karena kita tidak akan menggunakannya. Jika kalian penasaran kolom apa T itu, berikut penjelasannya:
# 
# - T adalah umur customer dari waktu pembelian pertama sampai akhir periode waktu yang ditentukan. representasi unit waktunya bisa menggunakan jam, hari, minggu, dll, tergantung kebijakan kalian.
# 
# lalu apa bedanya dengan recency? kalian bisa membandingkannya dengan penjelasan recency sebagai berikut:
# 
# - Recency adalah umur customer ketika mereka melakukan pembelian yang paling terbaru(recent). ini sama dengan durasi pembelian pertama(first purchase) sampai pembelian terbaru (latest purchase). *jika customer hanya melakukan satu kali pembelian, maka recency-nya adalah 0.

# In[ ]:


rfm = summary_data_from_transaction_data(orders, 'CustomerID', 'InvoiceDate', monetary_value_col='Revenue').reset_index()
rfm


# Kita dapati terdapat banyak sekali kolom yang valuenya 0, apa yang terjadi?.
# 
# Semuanya berawal dari recency, masih ingat definisi recency?.
# 
# - Recency: Recency adalah umur customer ketika mereka melakukan pembelian yang paling terbaru (recent). Atau sama dengan durasi pembelian pertama(first purchase) sampai pembelian terbaru (latest purchase). *jika customer hanya melakukan satu kali pembelian, maka recency-nya adalah 0.
# 
# jika dilihat dari definisi diatas, maka kebanyakan customer tidak melakukan pembelian ulang, alias hanya membeli satu kali saja. efeknya, nilai frequency pun ikut menjadi 0, kok bisa? coba baca definisi frequency berikut:
# 
# - Frequency: Angka pembelian ulang yang dilakukan customer. Atau sama dengan total pembelian dikurangi satu.
# 
# Jika total pembelian yang dilakukan customer hanya 1 kali, maka: 1-1= 0. Dan tentu saja efek ini berimbas ke monetary value. kalau recency dan frequency nya saja 0, lalu apa yang bisa dihitung untuk dimasukkan ke monetary value?.
# 
# Maka dari itu kita akan melakukan sedikit penyesuaian lagi. Sekarang coba kita cek kondisi frequency dengan plot histogram.

# In[ ]:


plt.hist(rfm['frequency'])
plt.title('Frequency')
plt.ylabel('Jumlah Customer' )
plt.xlabel('Frequency');


# Bisa dilihat, customer yang tidak melakukan transaksi ulang jumlahnya sangat mendominasi. Untuk membuat segmen yang lebih masuk akal, kita akan membuang data customer yang tidak melakukan pembelian ulang.

# In[ ]:


rfm = rfm[rfm['frequency']>0]
rfm.head()


# Buat histogram lagi untuk mengetahui distribusi kolom yang baru. sekarang distribusinya terlihat lebih baik, meskipun masih terjadi ketimpangan.

# In[ ]:


plt.hist(rfm['frequency'])
plt.title('Frequency')
plt.ylabel('Jumlah Customer', )
plt.xlabel('Frequency');


# Sekarang gantian kita cek distribusi monetary_value dengan histogram

# In[ ]:


plt.hist(rfm['monetary_value'])
plt.title('Monetary Value')
plt.ylabel('Jumlah Customer', )
plt.xlabel('Monetary Value');


# Oke, terlihat keseluruhan customer memiliki monetary value diantara 0 sampai 500. tetapi ada juga customer yang memiliki monetary value sebanyak 2000 keatas, bahkan sampai 16000an.

# In[ ]:


rfm = rfm[rfm['monetary_value']<2000]
rfm.head()


# Berikut adalah distribusi monetary value tanpa outliers. *seharusnya kita melakukan investigasi lebih lanjut terhadap outliers, dan tidak membuangnya begitu saja, tapi untuk kali ini, kita buang saja outliersnya.

# In[ ]:


plt.hist(rfm['monetary_value'])
plt.title('Monetary Value')
plt.ylabel('Jumlah Customer', )
plt.xlabel('Monetary Value');


# ## Step 2: Hitung Individual RFM Score

# Menghitung RFM individual Score dapat dilakukan dengan beberapa cara, kalian bisa menghitungnya menggunakan rumus perhitungan bisnis kalian sendiri yang kira-kira cocok dengan basis customer kalian. Kali ini kita akan menggunakan metode statistikal Quartil (membagi Score menjadi empat bagian)

# In[ ]:


quartiles = rfm.quantile(q=[0.25, 0.5, 0.75])
quartiles


# In[ ]:


def recency_score (data):
    if data <= 60:
        return 1
    elif data <= 128:
        return 2
    elif data <= 221:
        return 3
    else:
        return 4

def frequency_score (data):
    if data <= 1:
        return 1
    elif data <= 1:
        return 2
    elif data <= 2:
        return 3
    else:
        return 4

def monetary_value_score (data):
    if data <= 142.935:
        return 1
    elif data <= 292.555:
        return 2
    elif data <= 412.435:
        return 3
    else:
        return 4

rfm['R'] = rfm['recency'].apply(recency_score )
rfm['F'] = rfm['frequency'].apply(frequency_score)
rfm['M'] = rfm['monetary_value'].apply(monetary_value_score)
rfm.head()


# Setelah Score individual didapatkan, sekarang saatnya menghitung Score RFM  secara keseluruhan

# ## Step 3: Hitung RFM Score secara keseluruhan

# Untuk menghitung RFM Score secara keseluruhan kita cukup menjumlahkan value dari RFM individual score. 

# In[ ]:


rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)
rfm.head()


# ## Step 4: Berikan Label untuk RFM_Score

# Kalian boleh melabeli score dengan nama apapun dan range berapapun, tergantung kebijakan kalian. Kali ini kita akan mensegmentasikannya menjadi 5 bagian, dengan urutan label sebagai berikut:
# 'Bronze' sebagai yang terendah, diikuti dengan 'Silver', 'Goldr', 'Platinum', dan 'Diamond' yang tertinggi

# In[ ]:


rfm['label'] = 'Bronze' 
rfm.loc[rfm['RFM_score'] > 4, 'label'] = 'Silver' 
rfm.loc[rfm['RFM_score'] > 6, 'label'] = 'Gold'
rfm.loc[rfm['RFM_score'] > 8, 'label'] = 'Platinum'
rfm.loc[rfm['RFM_score'] > 10, 'label'] = 'Diamond'

rfm.head()


# Dan beginilah gambaran segmentasi customer yang sudah kita buat

# In[ ]:


barplot = dict(rfm['label'].value_counts())
bar_names = list(barplot.keys())
bar_values = list(barplot.values())
plt.bar(bar_names,bar_values)
print(pd.DataFrame(barplot, index=[' ']))


# Jika sebelumnya bagian marketing memberlakukan strategi yang sama terhadap semua customer, maka sekarang pihak marketing bisa memanfaatkan segmentasi RFM untuk membuat strategi yang lebih spesifik targetnya.
# 
# Contohnya seperti, tindakan apa yang harus dilakukan untuk mempertahankan customer agar selalu berada di kelas Diamond? jawabannya bisa dengan memberikan potongan harga yang spesial, buatkan personal merchandise, dll.
# 
# Sampai tahap ini kita bisa menyerahkan ke bagian marketing hasil segmentasi yang sudah dibuat untuk dilakukan perencanaan strategi dan analisa yang lebih mendalam.

# # Segmentasi Customer Menggunakan K means

# Tahapannya sama seperti ketika membuat RFM menggunakan library Lifetimes: 
# 
# - A. Persiapkan data
# - B. Buat RFM

# ## A. Persiapkan data

# Import data

# In[ ]:


df2 = pd.read_csv("/kaggle/input/onlineretail/OnlineRetail.csv",error_bad_lines=False, encoding='unicode_escape')
df2.head()


# Transform InvoiceDate ke python object datetime

# In[ ]:


df2['InvoiceDate'] = pd.to_datetime(df2['InvoiceDate'])
df2.head()


# Kita gunakan United Kingdom saja sama seperti sebelumnya.

# In[ ]:


uk = df2[df2['Country'] == 'United Kingdom']
uk.head()


# ## B. Hitung RFM Value

# Berikut adalah langkah-langkah dalam membuat RFM model menggunakan algoritma K means:
# 
# - Step 1, Hitung RFM Value, Tentukan Jumlah Cluster, dan Hitung RFM Score.
# - Step 2, Hitung score RFM secara keseluruhan
# - Step 3, Labeling

# Jika sebelumnya kita memanfaatkan library Lifetimes untuk menghitung RFM Value, maka kali ini kita akan menggunakan algoritma machine learning, K means. Algoritma K means ini termasuk kedalam algoritma unsupervised learning, yang artinya kita akan mengolah data yang tak berlabel, dan nanti kita akan melabeli outputnya sesuai dengan analisis kita sendiri.

# ## Recency

# ### 1. Hitung Recency Value

# Untuk mengetahui value recency nya, kita harus mengetahui jumlah hari tidak aktifnya customer (tidak melakukan pembelian) sejak pembelian terakhir. caranya? sebagai berikut:

# Ambil setiap ID yang ada

# In[ ]:


main_df = pd.DataFrame(df2['CustomerID'].unique())
main_df.columns = ['CustomerID']
main_df.head()


# Ambil waktu pembelian paling akhir (latest purchase) untuk setiap ID

# In[ ]:


latest_purchase = uk.groupby('CustomerID').InvoiceDate.max().reset_index()
latest_purchase.columns = ['CustomerID','LatestPurchaseDate']
latest_purchase.head()


# Hitung Recency, rumusnya:
# 
# Recency = titik waktu observasi - jumlah hari pembelian terakhir
# 
# *kita gunakan waktu yang paling terkini (the latest) sebagai titik waktu observasi.

# In[ ]:


latest_purchase['Recency'] = (latest_purchase['LatestPurchaseDate'].max() - latest_purchase['LatestPurchaseDate']).dt.days
latest_purchase.head()


# Munculkan kolom CustomerID dan Recency
# 
# Caranya bebas terserah kalian, kali ini saya akan gunakan merge.

# In[ ]:


main_df = pd.merge(main_df, latest_purchase[['CustomerID','Recency']], on='CustomerID')
main_df.head()


# Plot distribusi Recency valuenya

# In[ ]:


sns.distplot(main_df['Recency'], kde=False, bins=50)
plt.title('Distribusi Value Recency', fontsize = 20)
plt.xlabel('Recency')
plt.ylabel('count');


# Distribusi Recencynya skewed ke kanan

# ### 2. Tentukan Jumlah Cluster

# Untuk menentukan jumlah clusternya kita akan pakai inertia analysis

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    member = kmeans.fit_predict(np.array(main_df['Recency']).reshape(-1, 1))
    score.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), score)
plt.ylabel("Inertia")
plt.xlabel("n_clusters");


# Setelah melihat plot diatas, saya putuskan untuk menggunakan 4 kelas, karena 4 merupakan siku yang paling tidak tajam penurunannya.
# 
# Step selanjutnya, hitung Recency score

# ### 3. Hitung Recency Score

# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(main_df[['Recency']])
main_df['RecencyCluster'] = kmeans.predict(main_df[['Recency']])
main_df.head()


# cek snapshot (describe) Recency yang sudah kita buat
# 
# Ingat, cluster dibawah belum diurutkan, kita belum mengetahui kasta clusternya.

# In[ ]:


main_df.groupby('RecencyCluster')['Recency'].describe()


# Untuk mengetahui kasta urutan valuenya kita akan menggunakan function berikut

# In[ ]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# Berikut adalah cluster yang sudah diurutkan:

# In[ ]:


main_df = order_cluster('RecencyCluster', 'Recency',main_df,False)
main_df.head()


# In[ ]:


main_df.groupby('RecencyCluster')['Recency'].describe()


# Step selanjutnya lakukan hal yang sama kepada Frequency dan Monetary value

# ## Frequency

# ### 1. Hitung Frequency Value
# 
# Untuk menghitung Frequency, cukup gunakan kolom Frequency yang ada

# Ambil kolom CustomerID dan Frequency

# In[ ]:


frequency = uk.groupby('CustomerID').InvoiceDate.count().reset_index()
frequency.columns = ['CustomerID','Frequency']
frequency.head()


# Merge dengan dataframe recency yang tadi sudah dibuat

# In[ ]:


main_df = pd.merge(main_df, frequency, on='CustomerID')
main_df.head()


# Cek snapshot Frequency

# In[ ]:


main_df.Frequency.describe()


# Plot Distribusi Value Frequency

# In[ ]:


sns.distplot(main_df['Frequency'], kde=False, bins=50)
plt.title('Distribusi Value Frequency', fontsize = 20)
plt.xlabel('Frequency')
plt.ylabel('count');


# ### 2. Tentukan Jumlah Cluster

# In[ ]:


score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    member = kmeans.fit_predict(np.array(main_df['Frequency']).reshape(-1, 1))
    score.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), score)
plt.ylabel("Inertia")
plt.xlabel("n_clusters");


# Sama seperti Recency, kita akan gunakan 4 cluster

# ### 3. Hitung Frequency Score

# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(main_df[['Frequency']])
main_df['FrequencyCluster'] = kmeans.predict(main_df[['Frequency']])
main_df.head()


# Urutkan
# 
# Berikut adalah hasil cluster Frequency yang sudah dibuat

# In[ ]:


main_df = order_cluster('FrequencyCluster', 'Frequency',main_df,True)
main_df.head()


# Kondisi detil cluster Frequency

# In[ ]:


main_df.groupby('FrequencyCluster')['Frequency'].describe()


# ## Monetary Value (Revenue)

# ### 1. Hitung Monetary Value
# 
# Untuk menghitung Monetary Value, cukup kalikan UnitPrice dengan Quantity

# In[ ]:


uk['Revenue'] = uk['UnitPrice'] * uk['Quantity']
revenue = uk.groupby('CustomerID').Revenue.sum().reset_index()
revenue.head()


# merge dataframe

# In[ ]:


main_df = pd.merge(main_df, revenue, on='CustomerID')
main_df.head()


# In[ ]:


main_df['Revenue'].describe()


# In[ ]:


sns.distplot(main_df['Revenue'], kde=False, bins=50)
plt.title('Distribusi Revenue', fontsize = 20)
plt.xlabel('Revenue')
plt.ylabel('count');


# ### 2. Tentukan Jumlah Cluster

# In[ ]:


score = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    member = kmeans.fit_predict(np.array(main_df['Revenue']).reshape(-1, 1))
    score.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 5))
plt.plot(range(1, 15), score)
plt.ylabel("Inertia")
plt.xlabel("n_clusters");


# ### 3. Hitung Monetary Score

# In[ ]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(main_df[['Revenue']])
main_df['RevenueCluster'] = kmeans.predict(main_df[['Revenue']])
main_df.head()


# Urutkan

# In[ ]:


main_df = order_cluster('RevenueCluster', 'Revenue',main_df,True)
main_df.head()


# Berikut adalah gambaran cluster Monetary Value

# In[ ]:


main_df.groupby('RevenueCluster')['Revenue'].describe()


# ## Hitung Score RFM secara keseluruhan 

# Aturan perhitungan score:
# 
# - Recency: semakin rendah value Recency, semakin tinggi score nya.
# - Frequency: semakin tinggi value Frequency, semakin tinggi scorenya.
# - Monetary: semakin tinggi value Monetary, semakin tinggi scorenya.

# In[ ]:


main_df['RFM_score'] = main_df['RecencyCluster'] + main_df['FrequencyCluster'] + main_df['RevenueCluster']
main_df.head()


# In[ ]:


main_df['RFM_score'].unique()


# Score tertinggi yang kita miliki adalah 8 (dari total 9, R=3, F=3, M=3)

# In[ ]:


main_df.groupby('RFM_score')['Recency','Frequency','Revenue'].mean()
main_df.head()


# Binning.
# 
# Dibawah 2 adalah low value
# Diantara 2-4 adalah mid value
# Di

# In[ ]:


main_df['label'] = 'Bronze' 
main_df.loc[main_df['RFM_score'] > 1, 'label'] = 'Silver' 
main_df.loc[main_df['RFM_score'] > 2, 'label'] = 'Gold'
main_df.loc[main_df['RFM_score'] > 3, 'label'] = 'Platinum'
main_df.loc[main_df['RFM_score'] > 5, 'label'] = 'Diamond'

main_df.head()


# Berikut adalah bar plot dari label yang sudah dibuat

# In[ ]:


barplot = dict(main_df['label'].value_counts())
bar_names = list(barplot.keys())
bar_values = list(barplot.values())
plt.bar(bar_names,bar_values)
print(pd.DataFrame(barplot, index=[' ']))


# ## Kesimpulan 

# Segmentasi RFM adalah metode yang mudah untuk membuat segmentasi pelanggan. Outputnya intuitif, sehingga mudah untuk difahami dan di interpretasikan oleh pihak marketing nantinya. Namun dibalik kemudahannya, metode RFM memiliki beberapa kekurangan, yaitu:
# 
# - Perhitungan segmentasi RFM hanya memperhatikan tiga faktor saja (Recency, Frequency, dan Monetary Value), dan mengabaikan faktor lain yang sama atau mungkin bisa jadi lebih penting (seperti rincian demografis, jenis produk, dll) 
# 
# - Segmentasi customer RFM adalah segmentasi yang memakai metode historical, yang artinya penilaian segmentasi hanya berdasarkan data masa lalu, yang mungkin tidak bisa menggambarkan kondisi customer di masa depan dengan baik. 
# 
# Jadi, jangan mengambil keputusan hanya berdasarkan segmentasi RFM saja, perhatikan juga analisa model yang lain agar lebih luas point of view kita dalam mengambil keputusan.
# 
# Lalu antara Library Lifetimes dan K means, mana yang lebih baik? Jika dataset kalian tidak memiliki missing values, saya pribadi lebih merekomendasikan untuk pakai library saja, karena lebih cepat, mudah, dan praktis. Jika dataset kalian ada missing valuesnya, kalian bisa lakukan impute untuk mengisi missing valuesmya, tapi tentu saja hal itu akan mengurangi kepraktisan yang kita harapkan dari library.
# 
# Mungkin itu saja dari saya, jika ada pertanyaan atau masukan, bisa kalian tulis di kolom komentar atau boleh japri di ig saya: al.fath.terry 
# 
# Dan... mohon di upvote jika dirasa bermanfaat. 
# 
# Terimakasih :)
