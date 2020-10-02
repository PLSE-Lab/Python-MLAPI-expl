#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.describe(include='all')


# In[ ]:


df['Stars'].describe()


# In[ ]:


#Since the 'Stars' is in object data type instead of int, Convert it into int
df['Stars'] = pd.to_numeric(df['Stars'],errors='coerce')
df.describe(include='all')


# In[ ]:


df['Stars'].describe()


# In[ ]:


#check if there are any NULL values in the data
df.isnull().sum()


# In[ ]:


#replace the null values with '0'
df.fillna(0,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Top Ten'].head()


# In[ ]:


#Drop the Columns that are not usefull
df.drop('Top Ten',axis=1,inplace=True)


# In[ ]:


df.columns


# In[ ]:


#convert brand to lower case
df['Brand'] = df['Brand'].str.lower()
df.head()


# In[ ]:


df.Country


# In[ ]:


df.Country.unique()


# In[ ]:


#number of countries 
print(len(df.Country.unique()))


# In[ ]:


#Different styles on ramen 
df['Style'].unique()


# In[ ]:


df['Style'].value_counts()


# In[ ]:


#number of ramen brands given
print(len(df.Brand.unique()))


# In[ ]:


#top ten brands of ramen noodles
df['Brand'].value_counts()[:10]


# In[ ]:


#lets select the style of ramen noodles with stars above 4
style = df.Style[df.Stars > 4]


# In[ ]:


#the different styles of noodles with Stars above 4
style.value_counts()


# In[ ]:


# Ramen noodles in Japan with Stars above 4
j = df.loc[(df.Country=='Japan') & (df.Stars>4)]
j


# In[ ]:


sns.set(style='whitegrid')
f,ax = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='Brand',data=j)
plt.xticks(rotation=90)
plt.show()


# In Conclusion 'Nissin' is most preffered brand of Ramen Noodles in Japan succeeded by 'Myojp'.
