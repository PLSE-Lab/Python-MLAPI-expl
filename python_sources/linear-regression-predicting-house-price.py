#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Build Machine Learning Model menggunakan Linear Regression
# Dataset yang digunakan adalah harga rumah Bengaluru
# Model diadaptasi dari tutorial Youtube Codebasic
# Link youtube = https://www.youtube.com/watch?v=rdfbcdP75KI&list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg
# Github = https://github.com/codebasics


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/bengaluru_house_prices.csv")
df.head()
df.info()


# Pada model Linear Regression kita akan menggunakan features dengan tipe data numeric dan 1 feautre tipe categori, feature yang akan kita gunakan adalah (location,size, total_sqft, dan bath). Kolom 'price' akan kita gunakan sebagai target.

# ## Drop Column

# In[ ]:


df2 = df.drop(['area_type','availability','society','balcony'], axis='columns')
df2.head()


# ## Treating Missing Values

# In[ ]:


display(df2.info())
display(df2.isnull().sum())

df2 = df2.dropna()
df2.info()


# ## Cleaning Data (Kolom bhk)

# In[ ]:


df2['bhk'] = df2['size'].apply(lambda i: int(i.split(' ')[0]))
df2 = df2.drop(['size'], axis=1)
df2.info()


# ## Cleaning Data (kolom sqft)

# In[ ]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df2[~df2['total_sqft'].apply(is_float)]


# In[ ]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 
df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)


# In[ ]:


display(df3.info())
df3 = df3.dropna()
df3.info()


# ## Dimensional Reduction (kolom Location)

# In[ ]:


df4 = df3.copy()
display(len(df4.location.unique()))
df4['location'] = df4['location'].apply(lambda i: i.strip())
x = df4['location'].value_counts()
display(x.shape)
display(x[x<=10].shape)
len(x[x>10])


# Dari data di atas terdapat 1287 lokasi berbeda, terdapat 1047 lokasi yang hanya memiliki rumah tidak lebih dari 10 unit, dimensi ini bisa kita reduction dengan mengubah lokasi dibawah 10 unit menjadi 'other'

# In[ ]:


loc_less10 = x[x<=10]
len(df4['location'].unique())


# In[ ]:


df4['location'] = df4['location'].apply(lambda i: 'other' if i in loc_less10 else i)
len(df4['location'].unique())


# In[ ]:


display(df4.info())
df4.head()


# ## Outlier By Total sqft

# __Remove Outlier By Bisnis Mind__
# 
# Sebagai data scientist, saat anda berdiskusi dengan manager bisnis anda, dia akan mengatakan luas bhk secara normal adalah 300 square per feet. Jika ada dalam dataset tersebut yang luas kamarnya kurang dari 300 sqft maka akan kita anggap sebagai outlier.

# In[ ]:


df5 = df4[~(df4['total_sqft']/df4['bhk'] < 300)] ### ~ artinya komplement
df5.info()


# __Remove Outlier By Range__

# In[ ]:


df6 = df5.copy()
df6['price_per_sqft'] = df6['price']*100000 / df6['total_sqft']
df6.price_per_sqft.describe()


# Nilai maximum dari price_per_sqft 176470, sedangkan nilai minimum : 267. Dalam dataset ini jelas terlihat range yang besar, untuk itu kita akan menghapus outlier per location berdasar mean dan standard deviasinya

# In[ ]:


df7 = pd.DataFrame()
for key, sub in df6.groupby('location'):
    m = sub['price_per_sqft'].mean()
    st = sub['price_per_sqft'].std()
    new_df = sub[(sub['price_per_sqft'] > (m-st))&(sub['price_per_sqft'] < (m+st))]
    df7 = pd.concat([df7,new_df], ignore_index=True)
df7                                     


# ## Outlier by Bath

# In[ ]:


df7.bath.unique()
df7.shape


# Dalam suatu rumah mungkin semua kamar memiliki kamar mandi pribadi plus 1 kamar mandi tamu, tapi kecil kemungkinan dalam 1 rumah memiliki 2 kamar mandi lebih banyak daripada kamar tidur. Rumah yang memiliki 2 kamar mandi lebih banyak daripada kamar tidur kita anggap outlier

# In[ ]:


df8 = df7[df7['bath'] < df7['bhk']+2]
df8.shape


# In[ ]:


import plotly.express as px
fig = px.box(y=df8['bath'])
# fig = px.hist(y=df7['bath'])
fig.show()
fig = px.histogram(x=df8['bath'])
fig.show()


# ## Using One Hot Encoder for Feature Location

# In[ ]:


dummies = pd.get_dummies(df8.location)
dummies.head()


# In[ ]:


df9 = pd.concat([df8,dummies], axis='columns')
df9.head()


# In[ ]:


df10 = df9.drop(['location','price_per_sqft','other'], axis=1)
display(df10.head())
df10.shape


# ## Build Model

# In[ ]:


X = df10.drop(['price'],axis=1)
Y = df10.price
display(X.head())
display(Y.head())


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test) ## Default using R^2 Scoring


# Skor R^2 dari model Linear Rrgression yang dibuat adalah 0.83. Saya rasa ini cukup baik, mari kita coba skor model menggunakan Cross Validation

# ### Use Cross Validation to score our model

# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
cross_val_score(LinearRegression(),X,Y,cv=cv).mean()


# Skor model yang sudah kita buat 0.79, ini bukan hasil yang baik. Mari kita coba meningkatkan model menggunakan Ridge Regression dan Lasso 

# In[ ]:


## Ridge Regression
from sklearn.linear_model import Ridge

cross_val_score(Ridge(),X,Y,cv=cv).mean()


# In[ ]:


## Lasso Regression
from sklearn.linear_model import Lasso

cross_val_score(Lasso(),X,Y,cv=cv).mean()


# R^2 yang kita dapat dari 3 model diatas adalah LinearRegression = 0.788, RidgeRegression = 0.787, LassoRegression = 0.665. Untuk itu kita akan menggunakan Linear Regression sebagai model terbaik

# ## Test model for view properties

# In[ ]:


import numpy as np
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[ ]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[ ]:




