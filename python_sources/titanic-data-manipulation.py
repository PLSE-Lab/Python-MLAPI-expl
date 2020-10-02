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


# In[ ]:


#bir data 3 e bolunmus. 
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")    
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
data_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


data_gender.head()


# In[ ]:


# daginik olan test verisini birlestirelim
data_test2 = pd.merge(data_test, data_gender, on = 'PassengerId') 
data_test2.head()


# In[ ]:


len(data_test2.columns)


# In[ ]:


len(data_train.columns)


# In[ ]:


data_test2.columns==data_train.columns


# In[ ]:


data_train.columns


# In[ ]:


data_test2.columns


# In[ ]:


#kolonlari yer degistirdik. train i test ile ayni sirada yaptik.

data_train2 = data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
data_train2.tail()


# In[ ]:


data_test2.columns == data_train2.columns


# In[ ]:


data_test2.head()


# In[ ]:


data_train2.head()


# In[ ]:


#esitledigimiz datalari alt alta concat ettik
data = pd.concat([data_train2,data_test2], ignore_index=True)
data.head(3)


# In[ ]:


data_train.PassengerId.count()


# In[ ]:


data_test.PassengerId.count()


# In[ ]:


data_train.PassengerId.count() + data_test.PassengerId.count()


# In[ ]:


data.PassengerId.count()


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data.describe().T #sadece int ve float alir


# In[ ]:


#korelasyon

data.corr()


# In[ ]:


#cabin de kac cesit unique deger var.    
data.Cabin.unique()


# In[ ]:


data.Name.unique().size    #ile toplam sayiya bakabilirsiniz.


# In[ ]:


# Name sutununda tekrar eden degerleri sorgular
data["Name"].value_counts()


# In[ ]:


data[data["Name"]=="Kelly, Mr. James"]


# In[ ]:


data[data["Name"]=="Connolly, Miss. Kate"]


# In[ ]:


# Ticket sutununda tekrar eden degerleri sorgular

data["Ticket"].value_counts()


# In[ ]:


data[data["Ticket"]=="CA. 2343"]

#not bu 11 kisiye bakip yorum yapmaya calisin.


# In[ ]:


# sutunlardaki bos/NaN degerleri toplar.
data.isna().sum()


# In[ ]:


#kolon isimlerini turkcelestirelim.

data.rename(columns={"PassengerId": "YolcuID", 
                     "Pclass": "Sinif",
                     "Name": "Ad_Soyad",
                     "Sex": "Cinsiyet",
                     "Age" : "Yas",
                     "SibSp":"Aile_1",
                     "Parch" : "Aile_2",
                     "Ticket" : "BiletID",
                     "Fare" : "Fiyati",
                     "Cabin" : "KoltukNO",
                     "Embarked" : "Liman",
                     "Survived" : "Yasam"
                    }, inplace=True)   


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


#icerikleri replace edelim.

data["Yasam"].replace(0,"oldu", inplace=True)
data["Yasam"].replace(1,"yasiyor", inplace=True)
data["Liman"].replace("S","Southampton", inplace=True)
data["Liman"].replace("C","Cherbourg", inplace=True)
data["Liman"].replace("Q","Queenstown", inplace=True)
data["Cinsiyet"].replace("male","Erkek", inplace=True)
data["Cinsiyet"].replace("female","Kadin", inplace=True)

data.head(10)


# In[ ]:


# NaN Degerleri dolduralim.   > fillna

data["KoltukNO"].fillna("Belirsiz", inplace=True)
data.head()


# In[ ]:


#cinsiyete gore gruplayip, sayalim

data.Cinsiyet.groupby(data.Cinsiyet).count()


# In[ ]:


# Yasam durumuna gore gruplayip, sayalim

data.Yasam.groupby(data.Yasam).count()


# In[ ]:





# In[ ]:


#sinifi ==1 ve cinsiyeti == kadin olan degerler. coklu suzme

data[(data['Sinif'] == 1) & (data['Cinsiyet'] == 'Kadin')]


# In[ ]:


#bazi sutunlari drop edelim

data.drop(["Aile_1","Aile_2"],axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


#kadin ve erkekleri ayri ayri tutalim.
erkekler = data[data["Cinsiyet"]=="Erkek"]
kadinlar = data[data["Cinsiyet"]=="Kadin"]
cocuklar = data[data["Yas"]<=18]

kadinlar.head()


# In[ ]:


kadinlar.Yas.mean()


# In[ ]:


erkekler.Yas.mean()


# In[ ]:


# kac cocuk var? 
cocuklar.YolcuID.count()


# In[ ]:


cocuklar.Yas.mean()


# In[ ]:


sinif_1 = data[data["Sinif"] == 1]
sinif_2 = data[data["Sinif"] == 2]
sinif_3 = data[data["Sinif"] == 3]


# In[ ]:


# verideki siniflarin yas ve fiyat ortalamalarini alacak.
data.groupby("Sinif")["Yas","Fiyati"].mean()


# demekki zenginler daha yasli :) 

# **Istatiksel islemler icin veride bazi degisiklikler yapalim. **
# 
# Data uzerinde bilerek yapmadik. Cunku; Amacimiz ML degil, veri uzerinde pandas metodlari uygulamaktir

# In[ ]:


#farkli islemler yapalim. Onun icin hazir veriyi copy edelim
veri = data.copy()
veri.head()


# In[ ]:


#verideki NaN degerleri ve o satiri atacak
veri.dropna(inplace=True)


# In[ ]:


veri.count()


# In[ ]:





# In[ ]:


#yasi integer yaptik ve tekrar yas sutunu ile degisim yaptik.
veri["Fiyati"] = veri.Fiyati.astype('int64')


# In[ ]:


veri.head()


# In[ ]:


#her kolondaki en yuksek veriyi getirir.
veri.apply(np.max)


# In[ ]:


# Yasi 80 olan kisi
veri[veri["Yas"] >= 80]


# In[ ]:


veri["Yas"]


# In[ ]:


a = veri.iloc[:,[2,3]]


# In[ ]:


a.T


# In[ ]:


# pivot almak, satirlari sutun yapmak

veri.pivot(index ='Ad_Soyad', columns ='YolcuID')


# In[ ]:




