#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


filename = '/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'
df = pd.read_csv(filename)
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.drop(['last_review'], axis=1, inplace=True)
df.dropna(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


X = df['longitude'].values
Y = df['latitude'].values
hue = df['neighbourhood_group'].values
size = df['price'].values
plt.figure(figsize=(20, 10))
sns.scatterplot(X, Y, hue=hue, size=size, sizes=(50, 200))
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend(loc=0)


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(18,8))
df['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.1,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Share of Neighborhood')
ax[0].set_ylabel('Neighborhood Share')
sns.countplot('neighbourhood_group',data=df,ax=ax[1])
ax[1].set_title('Share of Neighborhood')
plt.show()


# In[ ]:


X = df['longitude'].values
Y = df['latitude'].values
hue = df['availability_365']
size = df['price'].values
plt.figure(figsize=(20, 10))
sns.scatterplot(X, Y, hue=hue)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend(loc=0)


# In[ ]:


plt.figure(figsize=(15, 8))
#creating a sub-dataframe with no extreme values / less than 500
sub=df[df.price < 500]
#using violinplot to showcase density and distribtuion of prices 
viz_2=sns.violinplot(data=sub, x='neighbourhood_group', y='price')
viz_2.set_title('Density and distribution of prices for each neighberhood_group')


# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x='neighbourhood_group', y='price', hue='room_type',data=df)


# In[ ]:


#word cloud
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in df.name)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=400).generate(text)
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['neighbourhood'])['minimum_nights', 'reviews_per_month'].count().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['neighbourhood'])['minimum_nights', 'number_of_reviews'].count().plot(ax=ax)


# In[ ]:





# In[ ]:




