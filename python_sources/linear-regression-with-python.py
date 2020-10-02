#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# Diambil dari materi pembelajaran kelas Python for Data Science di Purwadhika Startup and Coding School.
# Tujuan pembuatan model: Untuk memperkirakan harga sebuah rumah di USA dengan melihat beberapa variabel independen atau features yang mempengaruhi harga rumah tsb.
# Beberapa variabel yang mempengaruhi harga rumah adalah:
# 1. Average Income resident of the city where the house is located : Rata2 pendapatan orang yang tinggal di lokasi rumah.
# 2. Average age of the house in the city : Rata2 umur dari orang yang tinggal di kota yang sama lokasi rumah.
# 3. Average number of rooms in a house in the city : Rata2 jumlah ruangan di rumah tsb.
# 4. Average number of bedrooms in a house in the city : Rata2 jumlah tempat tidur di rumah tsb.
# 5. Population of the city that the house is located : Populasi dari kota dimana lokasi rumah berada.
# 6. Price of the house that is sold : Harga rumah.
# 7. Address : Alamat rumah.

# **1. Import semua dictionary yang dibutuhkan.**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/USA_Housing.csv') #Import data


# In[ ]:


df.head() #Melihat 5 data pertama


# In[ ]:


df.describe() #Melihat statistik setiap feature (variabel independen)


# In[ ]:


df.columns #Untuk melihat nama-nama kolom feature atau variabel independen


# In[ ]:


sns.pairplot(df) #Untuk melihat visualisasi hubungan (korelasi) antar feature atau variabel independen


# In[ ]:


sns.distplot(df.Price) #Melihat distribusi Price


# **2. Training the model**

# In[ ]:


X = df.drop(['Price','Address'], axis=1) #Untuk mendapatkan variabel independen atau feature. Karena Address tidak mempengaruhi harga rumah maka di-drop dan Price adalah variabel dependen atau target.


# In[ ]:


X.head()


# In[ ]:


y = df.Price


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split #Untuk split data menjadi data pembuatan model dan data untuk test.


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #test size 0.3 atau 30% berarti 30% dari data yg ada digunakan untuk test, dan sisanya digunakan untuk membuat model.


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X_train, y_train) #Memasukan data untuk membentuk model


# **3. Model Evaluation**

# In[ ]:


print('b =', model.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])


# In[ ]:


coeff_df


# **4. Model Prediction**

# In[ ]:


pred = model.predict(X_test) #Memprediksi sebagian data (X Test) menggunakan model


# In[ ]:


plt.scatter(y_test,pred)


# In[ ]:


sns.distplot((y_test-pred),bins=50)


# **5. Model Evaluation Metrics**

# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test,pred)


# In[ ]:


metrics.mean_squared_error(y_test,pred)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test, pred))


# In[ ]:


abs(y_test-pred).mean()


# In[ ]:


((y_test-pred)**2).mean()


# In[ ]:


np.sqrt(((y_test-pred)**2).mean())


# In[ ]:


metrics.r2_score(y_test,pred)


# In[ ]:




