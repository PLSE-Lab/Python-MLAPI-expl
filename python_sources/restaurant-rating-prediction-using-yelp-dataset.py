#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# >P.S. I decided to write this in Bahasa since there already exist many resources using English, but just very few ones in Bahasa. Hopefully by using Bahasa, local people (especially newbies like me) can understand it better and faster :) 
# 
# 
# Di notebook ini saya akan berbagi pengalaman/proyek saya terkait pembangunan sistem rekomendasi menggunakan beberapa teknik. Asumsi saya, pembaca sudah memahami sedikit mengenai konsep sistem rekomendasi. 
# 
# Sistem rekomendasi bisa dikategorikan sebagai *Supervised Learning* karena memiliki target kelas yang akan diprediksi. Pendekatan yang saya gunakan pada pembangunan sistem rekomendasi kali ini yaitu *multi-class prediction*, dimana untuk masing-masing pasangan *item*-pengguna akan diprediksi berapa *rating* yang akan diberikan, dengan *range* 1 - 5.
# 
# Data yang saya gunakan yaitu data [Yelp Academic dataset version 6](https://www.kaggle.com/yelp-dataset/yelp-dataset/version/6), disediakan oleh Yelp yang merupakan platform review berbagai bisnis di Amerika Serikat. Untuk versi terbaru (version 9) juga sudah tersedia, namun karena keterbatasan *resources* pada Kaggle, pada notebook ini saya masih menggunakan versi 6 dengan data yang lebih sedikit.

# # Business Understanding
# 
# Sistem rekomendasi menjadi salah satu *tool* untuk menyajikan informasi yang sesuai dengan preferensi pengguna. Pada use case kali ini, akan dilakukan rekomendasi terhadap restoran. Rekomendasi akan dilakukan dengan memanfaatkan data historis review/ulasan serta penilaian pengguna yang telah dilakukan sebelumnya. Dari data ini, dapat diprediksi bagaimana preferensi pengguna terhadap restoran-restoran lain yang terdapat pada platform tersebut. Restoran yang diprediksi memiliki *rating* yang tinggi akan disajikan sebagai rekomendasi terhadap user terkait

# # Data Understanding
# 
# Data yang akan dimanfaatkan yaitu data informasi terkait bisnis, pengguna, serta data ulasan terkait.
# 
# Pertama kita mulai dengan membaca dan menyimpan data-data tersebut ke *dataframe* pandas

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Business Data

# In[ ]:


# Membaca data bisnis
biz=pd.read_csv('/kaggle/input/yelpversion6/yelp_business.csv')
biz.head()


# In[ ]:


biz.shape


# terdapat 174.567 data bisnis dengan 13 atribut

# ### Memilih sebagian data 
# 
# kali ini kita akan mencoba  membangun rekomendasi hanya untuk bisnis **restoran**, sehingga data yang ada perlu di-*filter* dulu.
# 
# karena kolom kategori terdiri dari beberapa jenis, kita akan membaginya (split) menjadi satu kolom per kategori 

# In[ ]:


#memisahkan masing-masing kategori ke kolom berbeda
df_category_split = biz['categories'].str.split(';', expand=True)[[0,1,2]]

# nama kolom yang baru
df_category_split.columns = ['category_1', 'category_2', 'category_3']
biz = pd.concat([biz, df_category_split], axis=1)

# menghapus kolom lama 'categories'
biz = biz.drop(['categories'], axis=1)


# In[ ]:


biz.head()


# Setelah membagi kategori ke masing-masing kolom, kita akan melakukan *filter* pada data, disini kita akan memilih data **Restaurants** di *state* **PA (Pennsylvania)** dengan status *is_open* **True / 1**

# In[ ]:


# Filter dataset, 'kategori: Restaurants, 'state': PA, dan 'is_open' : 1
resto = biz.loc[(biz['category_1'] == 'Restaurants') | (biz['category_2'] == 'Restaurants') | (biz['category_3'] == 'Restaurants')]
resto = resto.loc[(resto['state'] == 'PA')]
resto = resto.loc[(resto['is_open'] == 1)]
print(resto.shape)


# untuk meminimalisir memori, kita bisa menghapus data biz awal yang sudah tidak lagi digunakan

# In[ ]:


#menghapus variabel yang tidak digunakan dan garbage collection
del biz

import gc
gc.collect()


# kemudian, kita akan memilih kolom-kolom yang akan digunakan. Disini kita akan memanfaatkan 3 kolom saja, yaitu **business_id, review_count, stars** 

# In[ ]:


resto.head()


# In[ ]:


#menghapus kolom yang tidak digunakan
resto=resto.drop(['name', 'neighborhood', 'address', 'city', 'state',
       'postal_code', 'latitude', 'longitude','is_open', 'category_1', 'category_2', 'category_3'],axis=1)
resto.reset_index(drop=True, inplace=True)


# In[ ]:


print(resto.info())
resto.head()


# Data resto final yang kita gunakan adalah data di atas 

# ## Users Data

# In[ ]:


# Membaca data user
user=pd.read_csv('/kaggle/input/yelpversion6/yelp_user.csv')
user.head()


# In[ ]:


print(user.shape)


# ### Memilih sebagian data 
# 
# Sama dengan sebelumnya, kali ini kita akan mem-*filter* data user. kolom *name* tidak akan digunakan. Selain itu ada baris yang akan kita buang, yaitu data **user yang tidak pernah melakukan review** (review_count=0) 

# In[ ]:


## Filter dataset

# Menghapus kolom 'name'
user=user.drop('name',axis=1)

# Memilih data user yang review_count nya >0
user = user.loc[(user['review_count'] > 0)]
print(user.shape)


# In[ ]:


print(user.info())
user.head()


# ## Reviews Data

# In[ ]:


# Membaca data review
reviews=pd.read_csv('yelp_review.csv')
reviews.head()


# In[ ]:


print(reviews.shape)
reviews.columns


# Disini kita tidak memanfaatkan informasi 'text', sehingga kolom tersebut akan dihapus

# In[ ]:


reviews=reviews.drop('text',axis=1)


# ## Joined Data

# Ketiga *dataframe* ini yaitu **resto, user, dan review** perlu digabung menjadi satu *dataframe*. Hal ini dilakukan dengan *inner join* atau *merge* yang akan menghubungkan ketiga data tersebut berdasarkan id nya

# In[ ]:


#join resto dan reviews
yelp_join=pd.merge(resto,reviews,on='business_id',how='inner')

#join resto, reviews dengan user
yelp_join=pd.merge(yelp_join,user,on='user_id',how='inner')
print(yelp_join.shape)
yelp_join.head()


# In[ ]:


#menghapus variabel yang tidak digunakan dan garbage collection
del resto
del reviews
del user

import gc
gc.collect()


# In[ ]:


yelp_join.head()


# # Data Preparation
# 
# Selanjutnya kita masuk ke tahap data preparation, yaitu menyiapkan data sehingga sesuai dengan kebutuhan model
# 
# Pertama, perlu dilakukan beberapa penyesuaian terhadap tipe data

# In[ ]:


print(yelp_join.dtypes)

#tipe data datetime
yelp_join['date']=pd.to_datetime(yelp_join['date'])
yelp_join['yelping_since']=pd.to_datetime(yelp_join['yelping_since'])


# Untuk masing-masing id yang bertipe string, akan dilakukan perubahan (index) menjadi id bertipe integer, yang dilakukan untuk penyederhanaan serta minimalisir memori 

# In[ ]:


#indexing id to number for simplicity
bizID = pd.Categorical((pd.factorize(yelp_join.business_id)[0] + 1))
userID = pd.Categorical((pd.factorize(yelp_join.user_id)[0] + 1))
reviewID = pd.Categorical((pd.factorize(yelp_join.review_id)[0] + 1))

bizID=bizID.astype(int)
userID=userID.astype(int)
reviewID=reviewID.astype(int)

yelp_join['business_id']=bizID
yelp_join['user_id']=userID
yelp_join['review_id']=reviewID


# Setelah itu dilakukan penyaringan untuk data yang tidak valid atau tidak sesuai dengan kebutuhan model. Diantaranya yaitu:
# 1. Data dengan tanggal review **(date)** lebih awal dibandingkan tanggal mendaftar **(yelping_since)** dianggap tidak valid, sehingga tidak disertakan dalam observasi
# 2. 

# In[ ]:


## Filter dataset
# Menghapus  incosistency: review date < yelping since
yelp_join = yelp_join.loc[((yelp_join['date'] > yelp_join['yelping_since']) == True)]
print(yelp_join.shape)


# In[ ]:


print(yelp_join.shape)
yelp_join.head()


# In[ ]:


print(yelp_join.business_id.nunique())
print(yelp_join.user_id.nunique())


# In[ ]:


yelp_join['user_id'].value_counts()


# In[ ]:


yelp_join['business_id'].value_counts()


# ## Filtering data : >1 user reviewing and >1 business reviewed

# In[ ]:


min_resto_ratings = 1
filter_resto = yelp_join['business_id'].value_counts() > 1
filter_resto = filter_resto[filter_resto].index.tolist()

min_user_ratings = 1
filter_users = df_new['user_id'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()


# In[ ]:


df_new = yelp_join[(yelp_join['business_id'].isin(filter_resto)) & (yelp_join['user_id'].isin(filter_users))]
print('The original data frame shape:\t{}'.format(yelp_join.shape))
print('The new data frame shape:\t{}'.format(df_new.shape))


# In[ ]:


print(yelp_join.business_id.nunique())
print(yelp_join.user_id.nunique())


# In[ ]:


yelp_join.head()


# ## Exporting data for collaborative filtering (PENDING)

# In[ ]:


cf=yelp_join


# In[ ]:


print(yelp_join.shape)
yelp_join.columns


# In[ ]:


#drop unused columns
#cf = cf.drop(['business_id'], axis=1)
cf = cf.drop(['stars_x'], axis=1)
cf = cf.drop(['review_count_x'], axis=1)
#cf = cf.drop(['review_id'], axis=1)
#cf = cf.drop(['user_id'], axis=1)
cf = cf.drop(['date'], axis=1)
cf = cf.drop(['useful_x'], axis=1)
cf = cf.drop(['funny_x'], axis=1)
cf = cf.drop(['cool_x'], axis=1)
cf = cf.drop(['review_count_y'], axis=1)
cf = cf.drop(['yelping_since'], axis=1)
cf = cf.drop(['friends'], axis=1)
cf = cf.drop(['cool_y'], axis=1)
cf = cf.drop(['useful_y'], axis=1)
cf = cf.drop(['funny_y'], axis=1)
cf = cf.drop(['fans'], axis=1)
cf = cf.drop(['elite'], axis=1)
cf = cf.drop(['average_stars'], axis=1)
cf = cf.drop(['compliment_hot'], axis=1)
cf = cf.drop(['compliment_more'], axis=1)
cf = cf.drop(['compliment_profile'], axis=1)
cf = cf.drop(['compliment_cute'], axis=1)
cf = cf.drop(['compliment_list'], axis=1)
cf = cf.drop(['compliment_note'], axis=1)
cf = cf.drop(['compliment_plain'], axis=1)
cf = cf.drop(['compliment_cool'], axis=1)
cf = cf.drop(['compliment_funny'], axis=1)
cf = cf.drop(['compliment_writer'], axis=1)
cf = cf.drop(['compliment_photos'], axis=1)
cf.columns


# In[ ]:


cf.to_csv('collaborative.csv')


# ## Data Exploration

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


print(yelp_join.shape)
yelp_join.head()
yelp_join.dtypes


# ### Rating Distribution

# In[ ]:


yelp_join=predictor


# In[ ]:


stars=yelp_join['rating'].value_counts()
stars=stars.to_frame().reset_index()
stars.columns=['rating','count']
print(stars)
stars.sort_values(by=['rating'],ascending=True).plot.bar(x='rating',y='count')


# ### Yelping Since

# In[ ]:


yr=yelp_join.groupby('yelping_since')[['user_id']].count()
yr


# In[ ]:


# df is defined in the previous example

# step 1: create a 'year' column
yelp_join['year_of_yelping'] = yelp_join['yelping_since'].map(lambda x: x.strftime('%Y'))

# step 2: group by the created columns
grouped_df = yelp_join.groupby('year_of_yelping')[['user_id']].count()

grouped_df


# In[ ]:


yr=grouped_df.reset_index()
yr


# In[ ]:


yr.plot.bar(x='year_of_yelping',y='user_id')


# ### Review Date

# In[ ]:


yr=yelp_join.groupby('date')[['review_id']].count()
yr


# In[ ]:


# df is defined in the previous example

# step 1: create a 'year' column
yelp_join['year_of_review'] = yelp_join['date'].map(lambda x: x.strftime('%Y'))

# step 2: group by the created columns
grouped_df = yelp_join.groupby('year_of_review')[['review_id']].count()

grouped_df


# In[ ]:


yr=grouped_df.reset_index()
yr


# In[ ]:


yr.plot.bar(x='year_of_review',y='review_id')


# ### User

# In[ ]:


us=yelp_join.groupby('user_id')[['business_id']].count()
us


# In[ ]:


us=yelp_join.groupby('user_id')[['review_id']].count()
us


# ## Filtering Dataset

# ### Review Date

# In[ ]:


yelp_join.shape


# In[ ]:


# Only the last 3 years
#yelp_join=yelp_join.loc[(yelp_join['date'] >= '2005-10-01')]
sdfsd=yelp_join.loc[(yelp_join['date'] >= '2005-10-01')]
#yelp_join.shape


# In[ ]:


#x=yelp_join.loc[(yelp_join['date'] >= '2015-01-01')]
#x=x.loc[(x['yelping_since'] <'2015-02-01')]
print(sdfsd['date'].sort_values(ascending=True))


# In[ ]:


yelp_join=yelp_join.loc[(yelp_join['date'] < '2015-01-01')]
yelp_join.shape


# In[ ]:


yelp_join.shape


# In[ ]:


print(yelp_join.business_id.nunique())
print(yelp_join.user_id.nunique())


# ### Yelping Since

# In[ ]:


#x=yelp_join.loc[(yelp_join['yelping_since'] >= '2015-01-01')]
x=yelp_join.loc[(yelp_join['yelping_since'] <'2015-01-01')]
print(x['yelping_since'].sort_values(ascending=True))
print(x.shape)


# In[ ]:


yelp_join=yelp_join.loc[(yelp_join['yelping_since'] < '2015-02-01')]
yelp_join.shape


# In[ ]:


print(yelp_join['yelping_since'].sort_values(ascending=False))


# ## Derived Columns

# In[ ]:


yelp_join['no_friends']=0
yelp_join.loc[yelp_join['friends'] == 'None', ['no_friends']] = 1


# In[ ]:


yelp_join


# In[ ]:


yelp_join.shape


# In[ ]:


yelp_join['year_of_yelping']=yelp_join['year_of_yelping'].astype(int)
yelp_join['year_of_review']=yelp_join['year_of_review'].astype(int)
yelp_join.dtypes


# In[ ]:


#Check check

#check recent date
print(yelp_join[['date','yelping_since','review_id']].sort_values(by='date',ascending=False).head())
#print(yelp_join['date'].loc[yelp_join['index']==29524])


# In[ ]:


from datetime import datetime

d_base = datetime(2015, 1, 1)
print(d_base)


# In[ ]:


print(yelp_join['date'].loc[yelp_join['review_id']==53024])
days=(d_base-(yelp_join['date'].loc[yelp_join['review_id']==53024]))
print(days)


# In[ ]:


yelp_join.shape


# In[ ]:


yelp_join.columns


# In[ ]:


#derive columns
df = pd.DataFrame([])
for index, row in yelp_join.iterrows():
    #total friends
    number=row['friends'].count(",")+1
    #days been yelping since
    days=(d_base-row['yelping_since']).days
    #total compliments
    compnum=row['compliment_hot']+row['compliment_more']+row['compliment_cute']+row['compliment_note']+row['compliment_cool']+row['compliment_funny']+row['compliment_writer']+row['compliment_photos']
    #total votes per user
    votes=row['funny_y']+row['useful_y']+row['cool_y']
    #review age
    age=(d_base-row['date']).days
    print(days)
    df = df.append(pd.Series([row['review_id'],row['no_friends'],number,compnum,days,age,votes]),ignore_index=True)
    
df.columns=['review_id','no_friends','total_friends','total_compliments','days_since','review_age','total_votes']
df.shape


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.loc[df['no_friends'] == 1, ['total_friends']] = 0
df = df.drop(['no_friends'], axis=1)


# In[ ]:


yelp_join=pd.merge(yelp_join,df,on='review_id',how='inner')
yelp_join.shape


# In[ ]:


yelp_join.head()


# In[ ]:


print(yelp_join[['date','review_age','review_id']].sort_values(by='date',ascending=False).head())
print(yelp_join[['yelping_since','days_since','review_id']].sort_values(by='yelping_since',ascending=False).head())


# In[ ]:


yelp_join=yelp_join.rename(columns={"stars_x": "biz_avg_rating",
                          "review_count_x": "biz_total_rvw",
                          "stars_y": "rating",
                          "useful_x": "review_useful",
                          "funny_x": "review_funny",
                          "cool_x": "review_cool",
                          "review_count_y": "user_total_rvw",
                          "average_stars" : "user_avg_rating",
                         })


# In[ ]:


yelp_join['elite']


# In[ ]:


z=yelp_join.describe()
z
z.to_csv('descriptive-stats.csv')
#2019-07-27 17:36


# In[ ]:


skew=yelp_join.skew(axis=0,numeric_only=True)
skew.to_csv('skew.csv')
#2019-07-27 17:36


# In[ ]:


#export data to master file
yelp_join.to_csv('yelp_join_added_columns.csv')
#2019-07-27 17:36


# In[ ]:


yelp_join.head()


# In[ ]:


#Read data from file
yelp_join=pd.read_csv('yelp_join_added_columns.csv',index_col=0)
yelp_join.head()


# In[ ]:





# ## Additional

# In[ ]:


yelp_join.business_id.nunique()


# In[ ]:


#for recent-ness column of item
bizpopularity=yelp_join.groupby('business_id')[['review_age']].mean()
bizpopularity


# In[ ]:


yelp_join=pd.merge(yelp_join,bizpopularity,on='business_id',how='inner')
yelp_join.head()
yelp_join.shape


# In[ ]:


yelp_join.columns


# In[ ]:


#review metadata columns for user feature
userreview=yelp_join[['review_id','user_id','review_useful','review_funny','review_cool','user_total_rvw']]
#c=userreview.groupby('user_id')[['cool_y','funny_y','useful_y','review_count_y']].mean()


# In[ ]:


userreview.sort_values(by='user_id')


# In[ ]:


yelp_join.business_id.nunique()


# In[ ]:


c=userreview.groupby('user_id').agg({'review_useful' : 'sum','review_funny' : 'sum','review_cool' : 'sum'})
c


# In[ ]:


yelp_join=pd.merge(yelp_join,c,on='user_id',how='inner')
yelp_join.head()
yelp_join.shape


# In[ ]:


yelp_join.columns
yelp_join.shape


# In[ ]:


yelp_join.columns


# ## Final

# In[ ]:


print(yelp_join.shape)
print(yelp_join.business_id.nunique())
print(yelp_join.user_id.nunique())


# In[ ]:


yelp_join.groupby(['business_id','user_id']).size()


# In[ ]:


#export
yelp_join.to_csv('resto_full.csv')
#2019-07-27 18:36


# In[ ]:


#import
import pandas as pd
yelp_join=pd.read_csv('resto_full.csv',index_col=0)


# In[ ]:


yelp_join.columns


# In[ ]:


yelp_join.shape


# # Separate Users

# ## Test data

# In[ ]:


z=yelp_join.groupby('user_id').agg({'review_id' : 'count'})
z=z.sort_values(by='review_id',ascending=True)
z


# In[ ]:


z.to_csv('user-review-count.csv')


# In[ ]:


z.describe()


# In[ ]:


##Split test user, top 10%
from sklearn.model_selection import train_test_split
user_train, user_test = train_test_split(z,test_size=0.1,shuffle=False)


# In[ ]:


user_test=user_test.reset_index()


# In[ ]:


user_test.sort_values(by='user_id',ascending=True)


# In[ ]:


yelp_join.shape


# In[ ]:


forsample = yelp_join[(yelp_join['user_id'].isin(user_test['user_id']))]
forsample.shape


# In[ ]:


forsample.head()


# In[ ]:


#get 20% from the whole dataset
train, test = train_test_split(forsample,test_size=0.425,shuffle=True)


# In[ ]:


test


# In[ ]:


test.to_csv('test_dataset.csv')


# ## Train data: except test data

# In[ ]:


yelp_join.shape


# In[ ]:


print(yelp_join.user_id.nunique())
print(yelp_join.business_id.nunique())


# In[ ]:


train = yelp_join[(~yelp_join['review_id'].isin(test['review_id']))]
train


# In[ ]:


#for training the neural network model
train.to_csv('train_dataset.csv')


# In[ ]:


yelp_join.business_id.nunique()


# In[ ]:


print(yelp_join.user_id.nunique())
print(yelp_join.business_id.nunique())


# In[ ]:


print(train.user_id.nunique())
print(train.business_id.nunique())


# In[ ]:


#import
import pandas as pd
yelp_join=pd.read_csv('resto_full.csv',index_col=0)
train=pd.read_csv('train_dataset.csv',index_col=0)
test=pd.read_csv('test_dataset.csv',index_col=0)


# In[ ]:


train.shape


# In[ ]:


train.sha


# In[ ]:


train['rating'].hist()


# In[ ]:


test.shape


# In[ ]:


test['rating'].hist()


# ## Dataset to be predicted: to make full rating

# In[ ]:


print(yelp_join.shape)
print(train.shape)
print(test.shape)


# In[ ]:


yelp_join.user_id.nunique()


# In[ ]:


nuser=yelp_join.user_id.unique()
nuser


# In[ ]:


userset = pd.DataFrame({'user_id':nuser[:]})


# In[ ]:


userset


# In[ ]:


nbiz=yelp_join.business_id.unique()
nbiz


# In[ ]:


bizset = pd.DataFrame({'business_id':nbiz[:]})


# In[ ]:


bizset


# In[ ]:


userset['key'] = 0
bizset['key'] = 0

df_cartesian = userset.merge(bizset,on='key',how='outer')
df_cartesian = df_cartesian.drop(columns=['key'])
df_cartesian


# In[ ]:


iddata=train[['user_id','business_id']]
iddata


# In[ ]:


df_1_2 = df_cartesian.merge(iddata,on=['user_id','business_id'], how='left',indicator=True)
df_1_not_2 = df_1_2[df_1_2["_merge"] == "left_only"].drop(columns=["_merge"])
df_1_not_2


# In[ ]:


iddata=test[['user_id','business_id']]
iddata


# In[ ]:


df_1_2 = df_1_not_2.merge(iddata,on=['user_id','business_id'], how='left',indicator=True)
df_final = df_1_2[df_1_2["_merge"] == "left_only"].drop(columns=["_merge"])
df_final


# In[ ]:


bizf=yelp_join[['business_id','biz_avg_rating','biz_total_rvw','review_age_y']]
bizf.sort_values(by='business_id')
bizf=bizf.drop_duplicates()
bizf


# In[ ]:


df_final= df_final.merge(bizf,on='business_id', how='left')
df_final


# In[ ]:


userf=yelp_join[['user_id','fans','user_total_rvw','user_avg_rating','days_since','total_friends','total_compliments','total_votes','review_useful_y', 'review_funny_y', 'review_cool_y']]
userf.sort_values(by='user_id')
userf=userf.drop_duplicates()
userf


# In[ ]:


df_final= df_final.merge(userf,on='user_id', how='left')
df_final.head()


# In[ ]:


df_final.sort_values(by='user_id')


# In[ ]:


df_final.to_csv('full-mat-predict.csv')


# In[ ]:


import pandas as pd


# In[ ]:


df_final=pd.read_csv('full-mat-predict.csv')


# In[ ]:


df_final.shape


# In[ ]:


df_final=df_final.drop('Unnamed: 0',axis=1)


# In[ ]:


df_final.head()


# In[ ]:


yelp_join.columns


# # Clean data

# ## Predictor

# In[ ]:


predictor=yelp_join[['rating','biz_avg_rating', 'biz_total_rvw',
       'review_age_y', 'fans', 'user_total_rvw', 'user_avg_rating',
       'days_since', 'total_friends', 'total_compliments', 'total_votes',
       'review_useful_y', 'review_funny_y', 'review_cool_y']]
predictor.shape


# In[ ]:


predictor.columns


# In[ ]:


predictor.shape


# In[ ]:


import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
corr = predictor.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# In[ ]:


import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
corr = predictor.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# In[ ]:


descdata=predictor.describe()
descdata.to_csv('predictor-descriptive.csv')


# ## Finally!

# In[ ]:


rating=predictor['rating']
pred = predictor.drop(['rating'], axis=1)

#export
pred.to_csv('predictor-new.csv')
rating.to_csv('target-new.csv',header=False)


# In[ ]:


import pandas as pd


# In[ ]:


#import
x=pd.read_csv('predictor-new.csv',index_col=0)
y=pd.read_csv('target-new.csv',index_col=0,header=None)


# In[ ]:


print(x.shape)
x.head()


# In[ ]:


print(y.shape)
y.head()


# # Pre-processing data

# In[ ]:


df_final.columns


# In[ ]:


df_final.shape


# In[ ]:


x


# In[ ]:


fullmatpred=df_final[['biz_avg_rating', 'biz_total_rvw',
       'review_age_y', 'fans', 'user_total_rvw', 'user_avg_rating',
       'days_since', 'total_friends', 'total_compliments', 'total_votes',
       'review_useful_y', 'review_funny_y', 'review_cool_y']]


# In[ ]:


test=test[['biz_avg_rating', 'biz_total_rvw',
       'review_age_y', 'fans', 'user_total_rvw', 'user_avg_rating',
       'days_since', 'total_friends', 'total_compliments', 'total_votes',
       'review_useful_y', 'review_funny_y', 'review_cool_y','rating']]


# In[ ]:


test


# In[ ]:


x_test=test[['biz_avg_rating', 'biz_total_rvw', 'review_age_y', 'fans',
       'user_total_rvw', 'user_avg_rating', 'days_since', 'total_friends',
       'total_compliments', 'total_votes', 'review_useful_y', 'review_funny_y',
       'review_cool_y']]


# In[ ]:


y_test=test[['rating']]


# In[ ]:


x_test


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.describe().to_csv('train-describe.csv')


# In[ ]:


test.describe().to_csv('test-describe.csv')


# In[ ]:


#encode target to 5 columns
# import preprocessing from sklearn
from sklearn import preprocessing
from tensorflow.python import keras
enc = preprocessing.LabelEncoder()

# 2. FIT
enc.fit(y)

# 3. Transform
labels = enc.transform(y)
labels.shape
y=keras.utils.to_categorical(labels)
# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data


# In[ ]:


#encode target to 5 columns
# import preprocessing from sklearn
from sklearn import preprocessing
from tensorflow.python import keras
enc = preprocessing.LabelEncoder()

# 2. FIT
enc.fit(y_test)

# 3. Transform
labels = enc.transform(y_test)
labels.shape
y_test=keras.utils.to_categorical(labels)
# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data


# In[ ]:


y.shape


# In[ ]:


y_test.shape


# In[ ]:


x.shape


# In[ ]:


##handling outliers
import numpy as np
import numpy.ma as ma
from scipy.stats import mstats

low = .05
high = .95
quant_df = x.quantile([low, high])
print(quant_df)


# In[ ]:


quant_df.head()


# In[ ]:


##handling outliers

# Winsorizing
x['biz_avg_rating']=mstats.winsorize(x['biz_avg_rating'], limits=[0.05, 0.05])
x['biz_total_rvw']=mstats.winsorize(x['biz_total_rvw'], limits=[0.05, 0.05])
x['review_age_y']=mstats.winsorize(x['review_age_y'], limits=[0.05, 0.05])
x['fans']=mstats.winsorize(x['fans'], limits=[0.05, 0.05])
x['user_total_rvw']=mstats.winsorize(x['user_total_rvw'], limits=[0.05, 0.05])
x['user_avg_rating']=mstats.winsorize(x['user_avg_rating'], limits=[0.05, 0.05])
x['days_since']=mstats.winsorize(x['days_since'], limits=[0.05, 0.05])
x['total_friends']=mstats.winsorize(x['total_friends'], limits=[0.05, 0.05])
x['total_compliments']=mstats.winsorize(x['total_compliments'], limits=[0.05, 0.05])
x['total_votes']=mstats.winsorize(x['total_votes'], limits=[0.05, 0.05])
x['review_useful_y']=mstats.winsorize(x['review_useful_y'], limits=[0.05, 0.05])
x['review_funny_y']=mstats.winsorize(x['review_funny_y'], limits=[0.05, 0.05])
x['review_cool_y']=mstats.winsorize(x['review_cool_y'], limits=[0.05, 0.05])


# In[ ]:


quant_df.head()


# In[ ]:


fullmatpred.loc[fullmatpred['biz_avg_rating'] < 2.5, 'biz_avg_rating'] = 2.5
fullmatpred.loc[fullmatpred['biz_avg_rating'] > 4.5, 'biz_avg_rating'] = 4.5
fullmatpred.loc[fullmatpred['biz_total_rvw'] < 15, 'biz_total_rvw'] = 15
fullmatpred.loc[fullmatpred['biz_total_rvw'] > 561, 'biz_total_rvw'] = 561
fullmatpred.loc[fullmatpred['review_age_y'] < 255, 'review_age_y'] = 255
fullmatpred.loc[fullmatpred['review_age_y'] > 1177, 'review_age_y'] = 1178
fullmatpred.loc[fullmatpred['fans'] < 0, 'fans'] = 0
fullmatpred.loc[fullmatpred['fans'] > 76, 'fans'] = 76
fullmatpred.loc[fullmatpred['user_total_rvw'] < 6, 'user_total_rvw'] = 6
fullmatpred.loc[fullmatpred['user_total_rvw'] > 862, 'user_total_rvw'] = 862
fullmatpred.loc[fullmatpred['user_avg_rating'] < 2.88, 'user_avg_rating'] = 2.88
fullmatpred.loc[fullmatpred['user_avg_rating'] > 4.43, 'user_avg_rating'] = 4.43
fullmatpred.loc[fullmatpred['days_since'] < 281, 'days_since'] = 281
fullmatpred.loc[fullmatpred['days_since'] > 2667, 'days_since'] = 2667
fullmatpred.loc[fullmatpred['total_friends'] < 0 , 'total_friends'] = 0
fullmatpred.loc[fullmatpred['total_friends'] > 589 , 'total_friends'] = 589
fullmatpred.loc[fullmatpred['total_compliments'] < 0 , 'total_compliments'] = 0
fullmatpred.loc[fullmatpred['total_compliments'] > 966 , 'total_compliments'] = 676
fullmatpred.loc[fullmatpred['total_votes'] < 0 , 'total_votes'] = 0
fullmatpred.loc[fullmatpred['total_votes'] > 4728 , 'total_votes'] = 4728
fullmatpred.loc[fullmatpred['review_useful_y'] > 201 , 'review_useful_y'] = 201
fullmatpred.loc[fullmatpred['review_funny_y'] > 78 , 'review_funny_y'] = 78
fullmatpred.loc[fullmatpred['review_cool_y'] > 78 , 'review_cool_y'] = 78


# In[ ]:


x_test.loc[x_test['biz_avg_rating'] < 2.5, 'biz_avg_rating'] = 2.5
x_test.loc[x_test['biz_avg_rating'] > 4.5, 'biz_avg_rating'] = 4.5
x_test.loc[x_test['biz_total_rvw'] < 15, 'biz_total_rvw'] = 15
x_test.loc[x_test['biz_total_rvw'] > 561, 'biz_total_rvw'] = 561
x_test.loc[x_test['review_age_y'] < 255, 'review_age_y'] = 255
x_test.loc[x_test['review_age_y'] > 1177, 'review_age_y'] = 1178
x_test.loc[x_test['fans'] < 0, 'fans'] = 0
x_test.loc[x_test['fans'] > 76, 'fans'] = 76
x_test.loc[x_test['user_total_rvw'] < 6, 'user_total_rvw'] = 6
x_test.loc[x_test['user_total_rvw'] > 862, 'user_total_rvw'] = 862
x_test.loc[x_test['user_avg_rating'] < 2.88, 'user_avg_rating'] = 2.88
x_test.loc[x_test['user_avg_rating'] > 4.43, 'user_avg_rating'] = 4.43
x_test.loc[x_test['days_since'] < 281, 'days_since'] = 281
x_test.loc[x_test['days_since'] > 2667, 'days_since'] = 2667
x_test.loc[x_test['total_friends'] < 0 , 'total_friends'] = 0
x_test.loc[x_test['total_friends'] > 589 , 'total_friends'] = 589
x_test.loc[x_test['total_compliments'] < 0 , 'total_compliments'] = 0
x_test.loc[x_test['total_compliments'] > 966 , 'total_compliments'] = 676
x_test.loc[x_test['total_votes'] < 0 , 'total_votes'] = 0
x_test.loc[x_test['total_votes'] > 4728 , 'total_votes'] = 4728
x_test.loc[x_test['review_useful_y'] > 201 , 'review_useful_y'] = 201
x_test.loc[x_test['review_funny_y'] > 78 , 'review_funny_y'] = 78
x_test.loc[x_test['review_cool_y'] > 78 , 'review_cool_y'] = 78


# In[ ]:


x.shape


# In[ ]:


fullmatpred.shape


# In[ ]:


#Normalization
from sklearn.preprocessing import MinMaxScaler
#Normalize data
scaler = MinMaxScaler()
# Fit only to the training data
x = scaler.fit_transform(x)
# Now apply the transformations to the data:
#x_test= scaler.transform(x_test)


# In[ ]:


fullmatpred


# In[ ]:


fullmatpred= scaler.transform(fullmatpred)


# In[ ]:


x_test= scaler.transform(x_test)


# In[ ]:


test.columns


# In[ ]:


##Split training and test set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,stratify=y)


# In[ ]:


x_test.shape


# In[ ]:


x_val.shape


# In[ ]:


x_test[6]


# # Modelling

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
 
# Do other imports now...


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
from sklearn.metrics import log_loss, confusion_matrix

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("GPU Available: ", tf.test.is_gpu_available())


# ## 13 features final

# In[ ]:


csv_logger = CSVLogger('log-final-2.csv', append=True, separator=';')
from sklearn import metrics
from sklearn.metrics import log_loss, confusion_matrix


# In[ ]:


model = Sequential()
model.add(Dense(4,input_dim=13, activation='tanh',use_bias=True, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(4, activation='tanh',use_bias=True, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(8, activation='tanh',use_bias=True, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax'))
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9)


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50,restore_best_weights=True)
#checkpoint
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(x_train, y_train,
          epochs=3000,batch_size=200,validation_data=(x_val, y_val),callbacks=[csv_logger,es,checkpoint]
          )

y_pred=model.predict(x_test)
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
pd.DataFrame(matrix).to_csv("result-final-2.csv",header=False,index=False)

from tensorflow.keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model-final-2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model-final-2.h5")
print("Saved model to disk")
 
# later...


# In[ ]:


# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print("Created model and loaded weights from file")


# In[ ]:


y_pred_val=model.predict(x_val)


# In[ ]:


acc=metrics.accuracy_score(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))
acc


# In[ ]:


matrix


# In[ ]:


len(history.history['loss'])


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# load json and create model
from tensorflow.keras.models import model_from_json

json_file = open('model-final-2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model-final-2.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[ ]:


y_pred=loaded_model.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


test.rating.value_counts()


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(rms)
mae = mean_absolute_error(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(mae)
conf=metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(conf)


# In[ ]:


df_final.shape


# In[ ]:


fmpred=loaded_model.predict(fullmatpred)


# In[ ]:


fmpredrating=fmpred.argmax(axis=1)
fmpredrating.min()


# In[ ]:


fmpredratingdf=pd.DataFrame({'col1':fmpred[:,0],'col2':fmpred[:,1],'col3':fmpred[:,2],'col4':fmpred[:,3],'col5':fmpred[:,4]})
fmpredratingdf


# In[ ]:


fmpredratingdf["rating"] = fmpredratingdf[["col1","col2","col3","col4","col5"]].max(axis=1)


# In[ ]:


fmpredratingdf


# In[ ]:


def get_status(df):
    if df['rating'] == df['col1']:
        return 1
    elif df['rating'] == df['col2']:
        return 2
    elif df['rating'] == df['col3']:
        return 3
    elif df['rating'] == df['col4']:
        return 4
    else:
        return 5

fmpredratingdf['star'] = fmpredratingdf.apply(get_status, axis = 1)


# In[ ]:


fmpredratingdf


# In[ ]:


full_matrix_id=df_final[['user_id','business_id']]
full_matrix_id


# In[ ]:


rat=fmpredratingdf[['star']]
rat=rat.rename({'star':'rating'},axis=1)
rat


# In[ ]:


fullpreddata=pd.concat([full_matrix_id, rat], axis=1)


# In[ ]:


fullpreddata


# In[ ]:


fullpreddata.to_csv('cf-predicted.csv',header=False,index=None)


# In[ ]:


train=pd.read_csv('train_dataset.csv',index_col=0)
train.head()


# In[ ]:


train_cf=train[['user_id','business_id','rating']]
train_cf.shape


# In[ ]:


fullpreddata.head()


# In[ ]:


test=pd.read_csv('test_dataset.csv',index_col=0)
test.head()


# In[ ]:


test_cf=test[['user_id','business_id','rating']]
test_cf.shape


# In[ ]:


full_total=fullpreddata.append(train_cf,sort=False)
full_total
full_total.to_csv('cf-full.csv',header=False,index=None)


# In[ ]:





# In[ ]:





# In[ ]:


full_total


# # -------------------- 

# 

# ## Alternatively: using sklearn

# In[ ]:


##Import library
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#model=MLPClassifier(hidden_layer_sizes=(4,4,8),max_iter=3000,solver='sgd',activation='tanh',alpha=0.01,learning_rate_init=0.0001,verbose=True)
#model=MLPClassifier(hidden_layer_sizes=(8,8,8),max_iter=3000,solver='adam',activation='tanh',verbose=True)
#model=KNeighborsClassifier(n_neighbors=1000)
#model=LogisticRegression(multi_class='auto',solver='sag')
#model=DecisionTreeClassifier()
model=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=3000,verbose=True,activation='tanh')


# In[ ]:


model.fit(x_train,y_train)
#Test the model
y_pred = model.predict(x_test)
#Print final result
print(confusion_matrix(y_test,y_pred))


# In[ ]:


model.n_layers_


# In[ ]:


accuracy = model.score(x_test,y_test)
print(accuracy*100,'%')


# In[ ]:


accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[ ]:




