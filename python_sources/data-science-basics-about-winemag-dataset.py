#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_wine = pd.read_csv('../input/winemag-data_first150k.csv')


# Basic informations about this dataset.
# 

# In[ ]:


data_wine.info()


# In[ ]:


data_wine.corr()


# In[ ]:


f , ax = plt.subplots(figsize=(14,14))
sns.heatmap(data_wine.corr() , annot=True , linewidths=0.6 , fmt='.1f' , ax=ax)
plt.show() 


# In[ ]:


data_wine.head()


# scatter

# In[ ]:


#data_wine.columns
plt.scatter(data_wine.index , data_wine.price , color='red' , alpha=0.5)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Price')


# histogram
# 

# In[ ]:


data_wine.price.plot(kind='hist',bins=50 , figsize=(5,5))
plt.show()


# pandas

# In[ ]:


x=data_wine['price']>1500

data_wine[x].head()


# filtering

# In[ ]:


data_wine[np.logical_and(data_wine['country']=='US' , data_wine['price']>500)]


# filtering with & operator

# In[ ]:


c=data_wine[(data_wine['country']=='France') & (data_wine['price']>500)]
c


# scope

# In[ ]:


x=5
def g():
    y=2**x
    return y
print(g())


# nested function
# 

# In[ ]:


def volume():
    def add():
        x=3
        y=4
        z=x+y
        return z
    return add()**3
print(volume())
    


# In[ ]:


def f(a,b=5,c=7):
    h=a+b+c
    return h
print(f(3))
print(f(4,6,8))


# In[ ]:


def f(*args):
    for i in args:
        print(i)  
f(1)
print("")
f(1,2,3,4)


# In[ ]:


z= lambda x,y : y+x
print(z(4,5))


# In[ ]:


number_list=[2,3,5]
z = map(lambda x:x**2,number_list)
print(list(z))


# Cleaning Data
# 

# In[ ]:


data_wine.head()


# In[ ]:


data_wine.tail()


# In[ ]:


data_wine.columns


# In[ ]:


data_wine.shape


# In[ ]:


data_wine.info()


# In[ ]:


print(data_wine['region_2'].value_counts(dropna = False))


# In[ ]:


data_wine.describe()


# In[ ]:


data_wine.boxplot(column='price' , by='points')
plt.show()


# In[ ]:


data_new=data_wine.head()
data_new


# In[ ]:


melted = pd.melt(frame=data_new , id_vars='country' , value_vars=['points','price'])
melted


# In[ ]:


data1 = data_wine['points'].head()
data2= data_wine['price'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col


# In[ ]:


#data_wine[:50]


# In[ ]:


data_wine['region_2'] = data_wine['region_2'].astype('category')
data_wine['region_1'] = data_wine['region_1'].astype('bool')


# In[ ]:


data_wine.dtypes


# In[ ]:


data_wine["region_2"].value_counts(dropna =False)


# In[ ]:


data1=data_wine
data1["region_2"].dropna(inplace = True)


# In[ ]:


assert 1==1


# In[ ]:


assert  data1['region_2'].notnull().all()


# In[ ]:


assert  data1['region_2'].notnull().all()
assert 1==1


# Data frames from dictionary
# 

# In[ ]:


country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label , list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# Add new columns

# In[ ]:


df["capital"] = ["Madrid","Paris"]
df


# BROADCASTING

# In[ ]:


df["income"] = 0 #Broadcasting entire column
df


# Subplots

# In[ ]:


data1.plot(subplots = True)
plt.show()


# Scatter plot

# In[ ]:


data1.plot(kind = "scatter" , x = "price" , y = "points")
plt.show()


# time lists
# 

# In[ ]:


time_list = ["1995-01-03","1996-01-03"]
print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


data2 = data_wine.head()
date_list = ["1995-01-03","1996-01-03","1995-02-03","1996-03-03","1996-04-03"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2 = data2.set_index("date")
data2


# In[ ]:


print(data2.loc["1996-01-03"])

print(data2.loc["1996-01-03" : "1996-04-03"])


# In[ ]:


data2.resample("A").mean()
data2.resample("M").mean()


# In[ ]:


data2.resample("M").first().interpolate("linear")


# In[ ]:


data2.resample("M").mean().interpolate("linear")


# In[ ]:


data2


# In[ ]:


data3 = data_wine


# In[ ]:


boolean = data3.price>2000
data3[boolean]


# In[ ]:


first_filter = data3.price>1500
second_filter = data3.points>90
data3[first_filter & second_filter]


# In[ ]:


data3.price[data3.points>98]


# In[ ]:


def div(n):
    return n/2
data3.head(10).points.apply(div)


# In[ ]:


data3.head().price.apply(lambda n : n/2)


# In[ ]:


data3["max_point"] = data3.price + data3.points
data3.head()


# In[ ]:


data3.index.name = "index_name"
print(data3.index.name)
data3.head()


# In[ ]:


data3.index=data3["index_name"]


# In[ ]:


data3

