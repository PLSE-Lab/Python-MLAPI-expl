#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/googleplaystore.csv')
data.head()


# In[ ]:


data_columns=data.columns.str.replace(" ","_")


# In[ ]:


print(data.columns)
print(data_columns)


# In[ ]:


print("Shape of the data and the types")
print(data.shape)
print(data.dtypes.value_counts())


# In[ ]:


#The data has 12 objects and 1 numeric values 
#Exploring each data types individually : 

#1. Size
data.Size.value_counts().head()


# In[ ]:


#2. Installs
data.Installs.value_counts()
data.Installs=data.Installs.apply(lambda x: x.strip('+'))
data.Installs=data.Installs.apply(lambda x: x.replace(',',''))
data.Installs=data.Installs.replace('Free',np.nan)


# In[ ]:


data.Installs.value_counts()


# In[ ]:


data.Installs.str.isnumeric().sum() 
data.Installs=pd.to_numeric(data.Installs)


# In[ ]:


data.Installs


# In[ ]:


data.Installs.hist()
plt.xlabel('Number of Installs')
plt.ylabel('Frequency')


# In[ ]:


#3.Reviews 
data.Reviews.str.isnumeric().sum()


# In[ ]:


data[~data.Reviews.str.isnumeric()]


# In[ ]:


data=data.drop(data.index[10472])#To check if the row is deleted? 
data[10471:10475]


# In[ ]:


data.Reviews=pd.to_numeric(data.Reviews)
data.Reviews.hist()
plt.xlabel("Number of Reviews")
plt.ylabel("Frequency")


# In[ ]:


#Rating 
#Find the range of the Ratings 
print("Range:",data.Rating.min(),"-",data.Rating.max())


# In[ ]:


print(data.Rating.isna().sum())
data.Rating.hist()
plt.xlabel("Rating")
plt.ylabel("Frequency")


# In[ ]:


#Category
data.Category.unique()


# In[ ]:


data.Category.value_counts().plot(kind='bar')


# In[ ]:


Category_list = list(data.Category.unique())
ratings=[]

for category in Category_list:
    x=data[data.Category==category]
    rating_rate= x.Rating.sum()/len(x)
    ratings.append(rating_rate)
dat=pd.DataFrame({'Category':Category_list,'Rating':ratings})
new_index=(dat['Rating'].sort_values(ascending=False)).index.values
sorted_data=dat.reindex(new_index)


# In[ ]:


sorted_data


# In[ ]:


import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import seaborn as sns
plt.xticks(rotation=45)
x=Category_list
sns.barplot(x=sorted_data.Category,y=sorted_data.Rating)
plt.xlabel("Application category")
plt.ylabel('Ratings')
plt.title('Average Rating by Category')
plt.show()


# In[ ]:


y=ratings
le=preprocessing.LabelEncoder()
y_encoded=le.fit_transform(x)


# In[ ]:


le = preprocessing.LabelEncoder()
x_encoded=le.fit_transform(x)
xaa = x_encoded.reshape(-1,1)
print(xaa)


# In[ ]:


result=KNeighborsClassifier(n_neighbors=3)
result.fit(xaa,y_encoded)


# In[ ]:





# In[ ]:




