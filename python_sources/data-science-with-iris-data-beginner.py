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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/Iris.csv')


# In[ ]:


data.columns


# In[ ]:


data.head()


# **DESCRIPTIVE STATISTICS**

# In[ ]:


data['Species']=data['Species'].astype('category')
data.dtypes


# In[ ]:


data.groupby("Species").describe()


# In[ ]:


print(data.SepalLengthCm.describe())
print(data.SepalWidthCm.describe())
print(data.PetalLengthCm.describe())
print(data.PetalWidthCm.describe())


# **CORRELATION COEFFICIENTS**

# In[ ]:


no_id=data.drop(['Id'],axis=1)
no_id.head()
no_id.corr()


# In[ ]:


print(data.Species.unique())


# In[ ]:


setosa=data[data.Species=='Iris-setosa']
versicolor=data[data.Species=='Iris-versicolor']
virginica=data[data.Species=='Iris-virginica']


# In[ ]:


print(data['Species'].value_counts())
df1=pd.DataFrame(data['Species'].value_counts())
df1


# **PIE CHART**

# In[ ]:


data.Species.value_counts().plot(kind='pie')


# **BAR PLOT**

# In[ ]:


x=['Iris-setosa','Iris-versicolor','Iris-virginica']
y=[setosa.SepalLengthCm.mean(),versicolor.SepalLengthCm.mean(),virginica.SepalLengthCm.mean()]
plt.bar(x,y, color='red')
plt.xlabel('Species')
plt.ylabel('SepalLengthCm Mean')
plt.title('bar plot')


# **LINE PLOT**

# In[ ]:


data.SepalLengthCm.plot(kind='line',color='green',label='sepal length cm',alpha=0.5,grid=True,linestyle=':')
plt.xlabel('Id')
plt.ylabel('sepal length cm')
plt.title('line plot')


# In[ ]:


plt.plot(setosa.Id,setosa.SepalLengthCm,color='red',label='setosa-sepal length cm')
plt.plot(versicolor.Id,versicolor.SepalLengthCm,color='blue', label='versicolor sepal length cm')
plt.plot(virginica.Id,virginica.SepalLengthCm, color='green', label='virginica sepal length cm')
plt.xlabel('Id')
plt.ylabel('sepal length cm')
plt.legend()


# **SCATTER PLOT**

# In[ ]:


plt.scatter(data.PetalLengthCm,data.PetalWidthCm, color='pink')
plt.xlabel('petal length cm')
plt.ylabel('petal width cm')
plt.title('petal length cm and petal width cm scatter plot')


# **FILTER EXERCISE**

# In[ ]:


filter1=data.SepalLengthCm>data.SepalLengthCm.mean()
filter2=data.SepalWidthCm>data.SepalWidthCm.mean()
filter3=data.PetalLengthCm>data.PetalWidthCm.mean()
filter4=data.PetalWidthCm>data.PetalWidthCm.mean()
data[filter1&filter2&filter3&filter4]


# In[ ]:




