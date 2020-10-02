#!/usr/bin/env python
# coding: utf-8

# **Section4- Python Data Science Tool Box**

# ![](https://i.etsystatic.com/9993958/r/il/d2bc14/1526829006/il_794xN.1526829006_lg0b.jpg)

# In this noteboook I'll analyse Iris Species Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Read the Iris data from Csv
iris=pd.read_csv("/kaggle/input/iris/Iris.csv")


# In[ ]:


# Display content of Iris data 
iris.info()


# In[ ]:


#Data overview
iris.describe()


# In[ ]:


#Data correlation
iris.corr()


# High correlation between the PetalWidthCm and PetalLengthCm.
# Negative correlation between the SepalWidthCm and SepalLenght and Petalx

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


iris


# In[ ]:


iris.columns


# In[ ]:


iris["Species"].unique()


# In[ ]:


seri=iris["SepalLengthCm"]
df=iris[["SepalLengthCm"]]


# In[ ]:


print(type(seri))
print(type(df))


# In[ ]:


iris[iris.SepalLengthCm>7.4]


# In[ ]:


iris[(iris.SepalLengthCm>7.4) & (iris.PetalLengthCm<6.7)]


# In[ ]:


iris[np.logical_and(iris["SepalLengthCm"]>7.4, iris["PetalLengthCm"]<6.7)]


# In[ ]:


#Add new features to the DataFrame
iris["TotalLenght"]=iris["SepalLengthCm"]+iris["PetalLengthCm"]
iris["TotalWidth"]=iris["SepalWidthCm"]+iris["PetalWidthCm"]
iris


# In[ ]:


#Adding new column to Dataframe with using if and for loop(list comprehension)
iris["Compare"]=[1 if i>14 or j>=6 else 0 for i,j in zip(iris.TotalLenght,iris.TotalWidth)]
iris


# In[ ]:


iris[iris.Compare==1]


# In[ ]:


list1=[i for i in iris.SepalLengthCm]
list2=[i for i in iris.SepalWidthCm]
zip_Sepal=zip(list1,list2)


# In[ ]:


print(zip_Sepal)


# In[ ]:


list_Sepal=list(zip_Sepal)
print(list_Sepal)


# In[ ]:


print(list_Sepal[0])
i,j=list_Sepal[0]
print("SepalL:",str(i),"SepalW:",str(j))


# In[ ]:


# iteration example
it=iter(list_Sepal)
print(next(it))
print(next(it))
print(*it)


# In[ ]:


un_zip=zip(*list_Sepal)
u_list1,u_list2=list(un_zip)
print(list(u_list1[0:2]), u_list2[0:2])


# In[ ]:


#Anonymous function
u_list1=list(u_list1)
u_list1
u_list1_double=map(lambda i:i*2, u_list1)
list(u_list1_double)


# In[ ]:




