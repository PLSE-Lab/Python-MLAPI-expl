#!/usr/bin/env python
# coding: utf-8

# In[615]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Pertama-tama, mari kita load data train ke notebook

# In[616]:


df = pd.read_csv('../input/train.csv')
df.head() #Liat 5 elemen teratas


# Kolom unnamed: 0 ternyata hanyalah nilai index, mari ubah dia menjadi index

# In[617]:


df['id'] = df['Unnamed: 0'] #buat kolom baru namanya id
df = df.drop(columns='Unnamed: 0') #ilangin kolom Unnamed: 0
df = df.set_index(df['id'])
df = df.drop(columns='id') #ilangin kolom id
df.head()


# Mari kita lihat info dari datanya

# In[618]:


df.info()


# Ternyata ada 299 data dengan 13 kolom, dengan data quantitatif sebanyak 5 dan qualitatif 8. Mari kita lihat data secara statistikal

# In[619]:


df.describe()


# Pilih input (x) dan output (y), untuk mempersimpel tutorial ini langsung pakai saja yang quantitative sebagai input

# In[620]:


x = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
y = df['Loan_Status']


# Standarisasi X

# In[621]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# Train Test Split

# In[622]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2) #Split train test


# Contoh kali ini saya akan menggunakan algoritma Logistic Regression untuk mengklasifikasi

# In[623]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
train = clf.fit(x_train, y_train)
pred = train.predict(x_test)


# In[624]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# Mari kita periksa akurasi dari model kita

# Jangan lupakan confusion matrixnya

# In[625]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)


# Apabila prediksi kita tepat, seharusnya bentuk dari confusion matrix kita adalah
#  <pre>[nilai, 0
#  0,    nilai]</pre>
# 

# Sehingga model kita ternyata tidak jelek-jelek amat

# # Bikin CSV Buat dikumpulin

# In[626]:


soal = pd.read_csv('../input/test.csv')
soal.head()


# In[627]:


soal['id'] = soal['Unnamed: 0'] #buat kolom baru namanya id
soal = soal.drop(columns='Unnamed: 0') #ilangin kolom Unnamed: 0
soal = soal.set_index(soal['id'])
soal = soal.drop(columns='id') #ilangin kolom id
soal.head()


# In[628]:


x_soal = soal[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]


# Jangan lupa standarisasi soalnya juga ya

# In[629]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
soal_scaled = scaler.fit_transform(x_soal)


# Soalnya tinggal kita prediksi pake model kita

# In[630]:


jawab = train.predict(soal_scaled)


# Ubah jawaban jadi DataFrame

# In[631]:


submission = pd.DataFrame({'Loan_Status' : jawab})
submission = submission.set_index(soal.index)
submission.head()


# Simpen Jawaban jadi file csv buat disubmit ke kaggle

# In[632]:


submission.to_csv('submission.csv')


# Tinggal klik commit dan output siap dikirim buat kompetisi ~

# Update :
# V4. Fixed Some Typos
