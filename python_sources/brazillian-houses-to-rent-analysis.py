#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv",index_col=False)
df1=pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv",index_col=False)
df
#df.city.unique()
#df.area.unique()


# In[ ]:


#df1.city.unique()
#df1.area.unique()
df1


# In[ ]:


df.area.unique()==df1.area.unique()


# In[ ]:


df.columns


# In[ ]:


df1.columns


# In[ ]:


df.drop('Unnamed: 0',axis=1)
df.city.isna().sum()


# In[ ]:


df.area.isna().sum()


# In[ ]:


for c in df.columns:
    print(c)

    


# In[ ]:


# Lets calculate the number of null values in all the columns
df.isnull().sum()


# In[ ]:


df.floor.unique()


# In[ ]:


df.floor.value_counts()


# In[ ]:


# we know that there are a lot of '-' values in the floor column, so we will change them to 0
df.floor=df.floor.replace('-',0)


# In[ ]:


df.floor.value_counts()


# In[ ]:


# Also in animal column we have the spelling of accept wrong , so let's change it

df.animal=df.animal.replace({'acept':'accept','not acept':'not accept'})


# In[ ]:


df.animal.unique()


# In[ ]:


df.furniture.unique()


# In[ ]:


def remove_dollor(x):
    a =  x[2:] #removes first two chr
    result = ""
    for i in a:
        if i.isdigit() is True:
            result = result + i
    return result #returns only digits (excludes special character)


# In[ ]:


df["hoa"] = pd.to_numeric(df["hoa"].apply(remove_dollor), errors= "ignore")
print(df.hoa.head())


# # In the hoa column in order to analyze the value we need it in the numeric form in order to do that we need to remove R$ 
# import re
# pattern=r'R\$'
# text=""
# #df.hoa=df.hoa.str.replace("R$"," ")
# df['hoa']=df['hoa'].str.replace("$","")
# df['hoa']=df['hoa'].str.replace("R","")
# df['hoa']=df['hoa'].str.replace(",","")
# df['hoa']=df['hoa'].replace('Sem info',)
# #df['hoa']=pd.to_numeric(df['hoa'])
# print(df['hoa'][35])
# print(df.hoa.value_counts())
# df["hoa"].unique()

# In[ ]:


#df['hoa']=pd.to_numeric(str(df['hoa']).str[:-1])
#df.hoa.unique()


# In[ ]:


df['furniture']= df['furniture'].map({'not furnished':'unfurnished','furnished':'furnished'})

df.furniture.unique()


# In[ ]:


import re
df['total']=df['total'].map(lambda x : re.sub(r'\D+','',str(x)))
df['total']=df['total'].astype(float)
df.total.unique()


# In[ ]:


#df['hoa'] = df['hoa'].map(lambda x: re.sub(r'\D+', '', x))

df['rent amount'] = df['rent amount'].map(lambda x: re.sub(r'\D+', '', str(x)))
df['rent amount']=df['rent amount'].astype(float)
df['rent amount'].unique()

df['fire insurance'] = df['fire insurance'].map(lambda x: re.sub(r'\D+', '', str(x)))
df['fire insurance']=df['fire insurance'].astype(float)
df['fire insurance'].unique()


# In[ ]:


df['property tax'].value_counts()
df["property tax"] = pd.to_numeric(df["property tax"].apply(remove_dollor), errors= "ignore")


# In[ ]:


df['property tax'].max()


# In[ ]:


import seaborn as sns
df1=df1.rename(columns={"rent amount (R$)":"rent","hoa (R$)":"hoa","property tax (R$)":"property_tax","fire insurance (R$)":"fire_insurance","total (R$)":"total"})
df1.columns
sns.lineplot(x="city",y="rent",data=df1)
plt.title("Rent Comparison per City")
plt.show()


# In[ ]:


# Extra expenses per city

x1=df1['city']
x2=df1['city']

y1=df1['rent']
y2=df1['hoa']

plt.subplot(2,1,1)
plt.plot(x1,y1)
plt.show()

plt.subplot(2,1,2)
plt.plot(x2,y2)
plt.show()


# In[ ]:


fig = plt.figure()

fig.add_subplot(1,1,1)
ax1 = fig.add_subplot(1,1,1)
ax1.plot(df1['rent'],df1['city'])


# In[ ]:


sns.boxplot(df['total'])


# In[ ]:



plt.figure(figsize = (7,7))
sns.set(style = "whitegrid")
f = sns.barplot(x = "rooms", y = "total", data = df1)
f.set_title("Before Removing Outliers")
f.set_xlabel("No. of Rooms")
f.set_ylabel("Total Cost")


# In[ ]:


plt.figure(figsize=(7,7))
sns.set(style="whitegrid")
z=sns.barplot(y='rent',x='rooms',data=df1)
z.set_title("Rent analysis according to the number of rooms")
z.set_ylabel("Rent")
z.set_xlabel("Rooms")


# In[ ]:


plt.figure(figsize=(7,7))
sns.set(style="whitegrid")
z=sns.barplot(y='total',x='city',data=df1)
z=sns.lineplot(y='rent',x='city',data=df1)
z=sns.lineplot(y='property_tax',x='city',data=df1)
z=sns.lineplot(y='fire_insurance',x='city',data=df1)
z.set_title("Rent analysis w.r.t city")
z.set_ylabel("Rent")
z.set_xlabel("city")


# # Here i would like to make a stacked barplot that shows me bifergation of different living costs in different cities
# plt.figure(figsize=(10,7))
# cities=df1.iloc[:0]
# t_rent=df1.iloc[:9]
# ren=df1.iloc[:12]
# plt.bar(x=cities,height=t_rent,width=0.35)
# plt.bar(x=cities,height=ren,width=0.35,bottom=t_rent)
# 
# plt.show()

# In[ ]:


columns = ["city","rooms","bathroom","parking spaces", "animal", "furniture"]
plt.figure(figsize = (30,30))
for i,var in enumerate(columns,1):
    plt.subplot(2,4,i)
    f = sns.barplot(x = df1[var], y = df1["total"])
    f.set_xlabel(var.upper())
    f.set_ylabel("Total Cost")


# In[ ]:


df_g=df1.groupby(['city'])[['total','rent','property_tax','fire_insurance']].mean()
#df_g1=df1.groupby(['total','rent','property_tax','fire_insurance'])[['city']].mean()
df_g1.head(5)


# In[ ]:


df_g.plot(kind='pie',subplots=True,legend=False,figsize=(20,30))


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(df1.corr(),annot=True)


# In[ ]:




