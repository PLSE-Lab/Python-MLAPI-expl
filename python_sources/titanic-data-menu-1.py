#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Get The Data from csv

# In[ ]:


data_train=pd.read_csv("/kaggle/input/titanic/train.csv")
data_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_test=pd.read_csv("/kaggle/input/titanic/test.csv")


# ## Analyze the titanic data

# In[ ]:


data_train


# In[ ]:


data_test


# In[ ]:


data_gender


# In[ ]:


data_train.head()#ilk belli sayidakiler
pd.options.display.max_rows=None # display all the table of data
pd.options.display.max_columns=None
display(data_train)


# In[ ]:


pd.reset_option("display.max_rows") # reset the periveous setting
pd.reset_option("display.max_columns")
pd.get_option("display.max_rows")# see default row lines


# # merge the datas

# In[ ]:


##LECTURE2
data_gender.info()


# In[ ]:


data_genderTestMarged=pd.merge(data_gender,data_test,on='PassengerId')
data_genderTestMarged # passengerId ayni oldundan data genderdaki survived sutunu almak icin merge yaptik


# In[ ]:


# bu merged ile data_tarin i merge yapacagiz columnlari uyusuyor mu kontrol edelim
data_genderTestMarged.columns==data_train.columns


# In[ ]:


# data_genderTestMarged ile data_tarin i concat.index devam etsin diye true dedik
data=pd.concat([data_train,data_genderTestMarged],ignore_index=True)
data


# In[ ]:


data.describe() # sadece int ve float olan degerler ile ilgili bilgi


# In[ ]:


data.info() # ilk  sayilari karsilastirip eksik var mi kontol ediyoruz


# In[ ]:


## transpoz. -to make rows to columns
data.describe().T


# In[ ]:


# yaslarin ortalamasi
data['Age'].mean()


# In[ ]:


# yuzde listesi describe bilgileimize rows olarak ekliyoruz
perc=[.10,.20,.40,.60,.80,.90]
include1_list =['object','float','int'] # object olanlari da ekleyince farkli verilerde ekleniyor tabloya
include_list =['float','int']  # --25,50,75 olan kismi degistirdik

desc=data.describe(percentiles=perc,include=include_list)
desc


# In[ ]:


# data setimizin koralsyonunu inceliyoruz.iki veri arasinda iliskiyi +- olarak inceliyor
korelasyon = data.corr()
korelasyon


# In[ ]:


# koralsyon tablosunda 1 den kucuk maksimum sayilari nasil listeleriz. +- 0.5 bi iliski oldugu izlenimini verir
korelasyon.abs()[korelasyon<1].max()


# In[ ]:


korelasyon[korelasyon<1].abs().max()


# In[ ]:


#LECTURE3
data[data.Cabin.isna()] # nan olan satirlari iste


# In[ ]:


# Cabin bilgisi dolu olanlari listele
data[data.Cabin.isna()==False]


# In[ ]:


# cabin sutununda tekrar eden verilerin sayisi
data["Cabin"].value_counts()


# In[ ]:


# name sutununda tekrar eden verilerin sayisi
data["Name"].value_counts()


# In[ ]:


# ayni gorulen 2 ismi kontol edelim gercekten ayni kisi mi ===sonuc===farkli kisiler
data[(data["Name"]=='Connolly, Miss. Kate') | (data["Name"]=='Kelly, Mr. James')]


# In[ ]:


# Ticket sutununda tekrar eden verilerin sayisi
data["Ticket"].value_counts()


# In[ ]:


# 11 tane ayni bilet numarasindan var inceleyelim ===sonuc==9 cocuklu anne baba ayni bilet no almis
data[data['Ticket']=='CA. 2343']


# In[ ]:


# yapilmis siniflandirmalari gorme --unique
data.Survived.unique() # 0 ve 1 degerleri
data.Pclass.unique() # 3,2,1 degerleri
data.Sex.unique() # male female
data.SibSp.unique() # 1, 0, 3, 4, 2, 5, 8
data.Parch.unique() # 0, 1, 2, 5, 3, 4, 6, 9
data.Embarked.unique() # 'S', 'C', 'Q', nan


# In[ ]:


# titanicteki yanliz seyhat eden kisilerin listesi
data[(data['SibSp']==0)&(data['Parch']==0)]


# In[ ]:


## ortalama yanlis seyhat etmeyen
a=data[(data['SibSp']>=1)|(data['Parch']>=1)]['Survived'].mean()  
b=data['Survived'].mean()
c=data[(data['SibSp']==0)&(data['Parch']==0)]['Survived'].mean()
a,b,c


# In[ ]:


data.isna().sum() # nan verilerin toplami sutunlarda


# In[ ]:



# kolon ismi degistime
#data.rename(columns={"PassengerId":'YolcuId',    ....}) hepsini tum clumnlari yazmak gerek


# In[ ]:


# LECTURE 4
# kacindeki nan olanlari belirsizle doldurduk
data['Cabin'].fillna('Belirsiz',inplace=True)
data


# In[ ]:


# icerik degistirme
#data["Survived"].replace(0,'Die',inplace=True)
#data["Survived"].replace(1,'Survive',inplace=True)
data["Survived"].replace('Die',0,inplace=True)
data["Survived"].replace('Survive',1,inplace=True)
data


# In[ ]:


# icerigi liste olarak degistirme
data['Embarked'].replace(["C","S",'Q'],["Cherbourg",'Southampton','Queenstown'],inplace=True)
data


# In[ ]:


# icerigi Dictionary olarak degistirme
data['Sex'].replace({'male': 'M' ,'female':"F"},inplace=True)
data.head()


# In[ ]:


# Nan degerlerini doldurmak
# yas da Nan olanlar ortalamaya katilmiyor.Ancak Nanlari ortalama ile degisebililriz
data['Age'].fillna(data['Age'].mean(),inplace=True)
data


# In[ ]:


# groupby ornekleri
data.Sex.groupby(data.Sex).count()


# In[ ]:


data.groupby(data.Sex).count()


# In[ ]:


data.groupby("Pclass")['Survived'].describe()


# In[ ]:


# aggergate valuelerini istedimiz memtodu yazabilirz, mean sum max 
data.groupby('Pclass').aggregate({'Fare':'mean','Age':'max'}) 


# In[ ]:


# pvot tableler  cinsiyet ve siniflara gore yasam degiskeninin ortalamisini gormek istiyoruz
data.pivot_table('Survived',index='Sex',columns='Pclass')


# In[ ]:


# cinsiyet ve bilet siniflandirmasina gore 0-25,25-80 yas araliginda yasam ortalamsi
yas = pd.cut(data['Age'],[0,25,80])
data.pivot_table('Survived',['Sex',yas],'Pclass')


# In[ ]:


# veriyi ortalamaya gore 2 gruba bolme-ortalama alti bir sinif ustu diger sinif
cost=pd.qcut(data['Fare'],2)
cost


# In[ ]:


#yas degiskenindeki durumlara ve bilet fiyat siniflandirmasina gore
# hayatta kalma durumu
#piotable(istedimisSiniflandirma,satirda ne olacak,columsda ne olcak)
data.pivot_table("Survived",('Sex',cost),(yas,"Pclass"))


# In[ ]:


data.pivot_table(index="Sex", columns="Pclass",aggfunc={"Survived":'mean',"Fare":'max'})


# In[ ]:


# cinsiyet ve bilet siniflarina gore yasam oranlari ve toplam duruma gore bakis
data.pivot_table("Survived",index="Sex",columns="Pclass",margins=True)


# In[ ]:




