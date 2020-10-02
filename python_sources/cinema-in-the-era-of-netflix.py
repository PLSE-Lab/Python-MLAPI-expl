#!/usr/bin/env python
# coding: utf-8

# I last had a Telivision way back in 2009.I cater to my needs through internet and youtube.Netflix has ccome into India recently.Through this dataset we will try to Learn More about NetFlix.This is a kernel in process.I will be updating the Kernel in Coming Days.If you Like my Work please do vote.

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


# **Importing Python Modules**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
#plt.style.use('ggplot')


# **Importing the Dataset**

# In[ ]:


df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
df.head()


# **Summary of Dataset**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n  :',df.columns.tolist())
print('\nMissing values :',df.isnull().values.sum())
print('\nUnique values  :  \n',df.nunique())


# In[ ]:


df.info()


# In[ ]:


df=df.dropna()


# **Changing the  to Date time format**

# In[ ]:


df["date_added"] = pd.to_datetime(df['date_added'])
df['day_added'] = df['date_added'].dt.day
df['year_added'] = df['date_added'].dt.year
df['month_added']=df['date_added'].dt.month
df['year_added'].astype(int);
df['day_added'].astype(int);
#df.year_added = df.year_added.astype(float)
#df.style.set_precision(0)
df.head()


# **What Type?**

# In[ ]:


print(df['type'].value_counts())


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['type'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Type of Movie')
ax[0].set_ylabel('Count')
sns.countplot('type',data=df,ax=ax[1],order=df['type'].value_counts().index)
ax[1].set_title('Count of Source')
plt.show()


# So 98 % items in the dataset are movies and remaining small percentage is TV show

# **Movie Rating?**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['rating'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Movie Rating')
ax[0].set_ylabel('Count')
sns.countplot('rating',data=df,ax=ax[1],order=df['rating'].value_counts().index)
ax[1].set_title('Count of Rating')
plt.show()


# 33% Fall in catogery TV-MA ("TV-MA" is a rating assigned by the TV Parental Guidelines to a television program that was designed for mature audiences only.)
# 
# 23% fall in catigery TV-14 (Programs rated TV-14 contains material that parents or adult guardians may find unsuitable for children under the age of 14.
# 
# 12.5 % fall in category TV-PG (TV-PG: Parental guidance suggested. This program contains material that parents may find unsuitable for younger children)

# **Movie count by Country**

# In[ ]:


group_country_movies=df.groupby('country')['show_id'].count().sort_values(ascending=False).head(10);
plt.subplots(figsize=(15,8));
group_country_movies.plot('bar',fontsize=12,color='blue');
plt.xlabel('Number of Movies',fontsize=12)
plt.ylabel('Country',fontsize=12)
plt.title('Movie count by Country',fontsize=12)
plt.ioff()


# So we can say most movies are from Hollywood,Bollywood and British Film industry

# **How many Movies Per Year?**

# In[ ]:


group_country_movies=df.groupby('year_added')['show_id'].count().sort_values(ascending=False).head(10);
plt.subplots(figsize=(15,8));
group_country_movies.plot('bar',fontsize=12,color='blue');
plt.xlabel('Number of Movies',fontsize=12)
plt.ylabel('Year',fontsize=12)
plt.title('Movie Count By Year',fontsize=12)
plt.ioff()


# Every Year the movie Count is increasing indicating that popularity of Netfilx is increasing every years.

# **Which Month has more movies Added?**

# In[ ]:


df['month_added'].value_counts();


# In[ ]:


ax=df.groupby('show_id')['month_added'].unique().value_counts().iloc[:-1].sort_index().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Number of Movies',fontsize=15)
ax.set_title('Number of Moves Based on Month',fontsize=15)
ax.set_xticklabels(('Jan','Feb','Mar','April','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'));
plt.show()


# We see more movies are added in the month of September followed by October and March.

# **Duration of Movie**

# In[ ]:



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)
df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)
#df.head()


# **Which are most popular words for Title?**

# In[ ]:


from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from PIL import Image
plt.style.use('seaborn')
wrds1 = df["title"].str.split("(").str[0].value_counts().keys()

wc1 = WordCloud(stopwords=STOPWORDS,scale=5,max_words=1000,colormap="rainbow",background_color="black").generate(" ".join(wrds1))
plt.figure(figsize=(20,14))
plt.imshow(wc1,interpolation="bilinear")
plt.axis("off")
plt.title("Key Words in Movie Titles",color='black',fontsize=20)
plt.show()


# Love,Man,Christmas etc are some of the most prominent words for movie Title 

# **Whats in the Description?**

# In[ ]:


#df['description'][1]
df['length']=df['description'].str.len()
df.dropna();


# In[ ]:


plt.figure(figsize=(12,5))

g = sns.distplot(df['length'])
g.set_title("Price Distribuition Filtered 300", fontsize=20)
g.set_xlabel("Prices(US)", fontsize=15)
g.set_ylabel("Frequency Distribuition", fontsize=15)


plt.show()


# We can see that the many reviews are in te ranged 140 to 150 words.

# In[ ]:


plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="rating", y="length", data=df,width=0.8,linewidth=3)
ax.set_xlabel('Rating',fontsize=30)
ax.set_ylabel('Length of Description',fontsize=30)
plt.title('Length of Description Vs Rating',fontsize=40)
ax.tick_params(axis='x',labelsize=20,rotation=90)
ax.tick_params(axis='y',labelsize=20,rotation=0)
plt.grid()
plt.ioff()


# Suprisingly the mean length of the description across all the ratings remain same.

# **Listed in?**

# In[ ]:


import plotly.graph_objects as go
from collections import Counter
col = "listed_in"
categories = ", ".join(df['listed_in']).split(", ")
counter_list = Counter(categories).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
plt.figure(figsize=(12,5))
sns.barplot(values[0:20],labels[0:20]);
plt.xlabel('Count',fontsize=10)
#plt.ylabel('',fontsize=20)
plt.title('Movie Listing',fontsize=20)
#ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# More of the listings are in the Catogery International TV Shows,TV Drama and Movies
