#!/usr/bin/env python
# coding: utf-8

# ## In which I start to explore the dataset, and this notebook is essentially a set of 'notes' on possible research questions which can be explored using the awesome Pitchfork reviews dataset
# ### Anything marked 'Do later' is a possible angle that I will be tackling sometime in the future

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3, datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

# Any results you write to the current directory are saved as output.


# In[ ]:


con = sqlite3.connect('../input/database.sqlite')

artists = pd.read_sql('SELECT * FROM artists', con)
content = pd.read_sql('SELECT * FROM content', con)
genres = pd.read_sql('SELECT * FROM genres', con)
labels = pd.read_sql('SELECT * FROM labels', con)
years = pd.read_sql('SELECT * FROM years', con)
reviews = pd.read_sql('SELECT * FROM reviews', con)

con.close()


# ### Combine everything into one dataframe

# In[ ]:


df1 = pd.merge(genres,reviews,on='reviewid')
df2 = pd.merge(labels,df1,on='reviewid')
df3 = pd.merge(content,df2,on='reviewid')
df_final = pd.merge(years,df3,on='reviewid')
df_final.head()


# In[ ]:


# Get rid of what we don't need right now
df_final.drop(['url','author_type','pub_date','pub_weekday','pub_day','pub_month'],axis=1,inplace=True)
df_final.dropna(inplace=True)
df_final['year'] = df_final['year'].astype(int)
df_final['pub_year'] = df_final['pub_year'].astype(int)
df_final.head()


# In[ ]:


df_final.drop_duplicates('reviewid',inplace=True)


# In[ ]:


df_final.shape


# ### This is a very basic first look at the data, so we don't impute missing values etc yet -- Do later

# ## How do average ratings correlate by genre, year?

# In[ ]:


df_final['genre'].value_counts() # 9 different genres, with an overwhelmingly large number of rock reviews


# In[ ]:


df_final.groupby('genre')['score'].mean().plot(kind='bar',rot=45,color='b',alpha=0.75,                                                figsize=(10,8),fontsize=14);
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Mean score',fontsize=16);


# ### So global music, experimental and jazz seem to have the highest average ratings, but is this just Pitchfork being niche or these 3 genres having a low number of reviews? Metal has a low number of reviews as well, but a lower overall score than the aforementioned genres. 

# In[ ]:


# Do overall score trends change over the years?
df_final.groupby('year')['score'].mean().plot(color='b',alpha=0.75,figsize=(10,8))
plt.xlabel('Year',fontsize=16)
plt.ylabel('Mean score',fontsize=16);


# ### So the reviews do seem to get harsher the newer the album release date is.. could be something to do with the fact that recently there have been several retrospective reviews of classic albums... but let's also look at how the years have treated the albums belonging to a particular genre

# In[ ]:


# Do overall score trends change by genre over the years?
df_final[df_final['genre']=='rock'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='b',alpha=0.75, label='Rock')
df_final[df_final['genre']=='electronic'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='r',alpha=0.75, label='Electronic')
df_final[df_final['genre']=='folk/country'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='g',alpha=0.75, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='y',alpha=0.75, label='Pop/RnB')
df_final[df_final['genre']=='jazz'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='c',alpha=0.75, label='Jazz')
df_final[df_final['genre']=='rap'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='m',alpha=0.75, label='Rap')
df_final[df_final['genre']=='experimental'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='k',alpha=0.75, label='Experimental')
df_final[df_final['genre']=='metal'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='gold',alpha=0.75, label='Metal')
df_final[df_final['genre']=='global'].groupby(['year'])['score'].mean().plot(figsize=(15,10),color='plum',alpha=0.75, label='Global')
plt.xlabel('Year',fontsize=16)
plt.ylabel('Mean score',fontsize=16);
plt.legend(fontsize=16);


# ### The same overall trend holds, but with some strange peaks and troughs. Rap seems to have peaked in 1995-96 (Tupac?) while there seems to have been a resurgence in critically acclaimed Folk/Country in early 2000, while Experimental completely nosedives around the year 2000. Will need to investigate further before identifying trends with any confidence. 
# 
# ### Let's also look at the boxplot of score trends by genre

# In[ ]:


plt.figure(figsize=(12,10))
df_final.boxplot(column='score',by='genre',figsize=(12,10))
plt.xlabel('Genre', fontsize=16)
plt.suptitle('')
plt.title('Score');


# ### Pitchfork really doesn't seem to like Metal, and Rock has a lot of low rated albums, which is again probably just a consequence of the sheer number of Rock album reviews.
# ### Let's check the distribution of poorly rated ($<5$) and highly rated albums ($>9$). 

# In[ ]:


df_high = df_final[df_final['score']>9]
df_low = df_final[df_final['score']<5]
df_high['genre'].value_counts()


# ### Metal has zero highly rated albums, I told you Pitchfork hates metal!!

# In[ ]:


df_low['genre'].value_counts()


# ### No real surprises here, what about just looking at extremely badly rated ($<3$) or perfectly rated ($10/10$) albums?

# In[ ]:


df_final[df_final['score']<3]['genre'].value_counts()


# In[ ]:


df_final[df_final['score']==10.0]['genre'].value_counts()


# ### A stacked bar chart would give more insights into this perhaps

# ### Let's look at a ratio of total number of albums reviewed to the low or high rated albums by genre

# In[ ]:


A = df_final.groupby('genre').size()
B = df_final[df_final['score']<5].groupby('genre').size()
C = df_final[df_final['score']>9].groupby('genre').size()


# In[ ]:


pd.concat([A,B],keys=["Total albums","Number of negatively reviewed albums (Score < 5)"],axis=1).plot(kind='bar',stacked=True,figsize=(12,10),rot=45)
plt.xlabel('Genre',fontsize=16);


# In[ ]:


pd.concat([A,C],keys=["Total albums","Number of positively reviewed albums (Score > 9)"],axis=1,sort=True).plot(kind='bar',stacked=True,figsize=(12,10),rot=45)
plt.xlabel('Genre',fontsize=16);


# ### No trend easily stands out. 

# In[ ]:


df_final[df_final['genre']=='rock']['score'].plot(kind='kde',figsize=(10,8),color='b',alpha=0.5, label='Rock')
df_final[df_final['genre']=='electronic']['score'].plot(kind='kde',figsize=(10,8),color='r',alpha=0.5, label='Electronic')
df_final[df_final['genre']=='folk/country']['score'].plot(kind='kde',figsize=(10,8),color='g',alpha=0.5, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b']['score'].plot(kind='kde',figsize=(10,8),color='y',alpha=0.5, label='Pop/R&B')
df_final[df_final['genre']=='jazz']['score'].plot(kind='kde',figsize=(10,8),color='c',alpha=0.5, label='Jazz')
df_final[df_final['genre']=='rap']['score'].plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='rap']['score'].plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='experimental']['score'].plot(kind='kde',figsize=(10,8),color='k',alpha=0.5, label='Experimental')
df_final[df_final['genre']=='metal']['score'].plot(kind='kde',figsize=(10,8),color='gold',alpha=0.5, label='Metal')
df_final[df_final['genre']=='global']['score'].plot(kind='kde',figsize=(10,8),color='plum',alpha=0.5, label='Global')
plt.xlabel('Score',fontsize=16)
plt.ylabel('Density',fontsize=16);
plt.legend(fontsize=16);
plt.ylim(0,)
plt.xlim(0,12);


# ### No apparent distinction in the distributions. Check via t-test, KS-test too perhaps? -- Do later

# #### Does it matter when the review was published? Are reviews over time getting more or less stringent? -- Do later

# In[ ]:


# Is the best new music biased towards one genre?
df_final[df_final['best_new_music']==1.0]['genre'].value_counts()
# Could just be a direct correlation to the number of albums reviewd per genre


# ### What about the 'best_new_music' albums... 

# In[ ]:


df_final[df_final['best_new_music']==1.0]['score'].plot(kind='kde',color='b',                                                        alpha=0.75,figsize=(10,8))
plt.xlabel('Score, Best New Music',fontsize=16)
plt.ylabel('Density',fontsize=16);
plt.ylim(0,);


# ### The best new music is predictably highly rated.. would be interesting to see if these vary by genre, or author -- Do later

# In[ ]:


df_final['best_new_music'].value_counts()


# In[ ]:


df_final[df_final['best_new_music']==1.0]['pub_year'].plot(kind='hist',color='b',                                                        alpha=0.75,figsize=(10,8),bins=50);


# ### 2009 onwards the number of albums with 'best_new_music' designations seems to be pretty steady; 

# In[ ]:


df_final.groupby('genre')['best_new_music'].sum().plot(kind='bar',color='b',alpha=0.75,
                                                      figsize=(10,8))
plt.xlabel('Genre',fontsize=16)
plt.ylabel('Number of albums designated "Best New Music"',fontsize=16);


# ### Rock does, predictably, have a high number of best_new_music designations. Quite a lot for Electronic as well. 

# In[ ]:


A = df_final.groupby('genre').size()
B = df_final[df_final['best_new_music']==1.0].groupby('genre').size()
pd.concat([A,B],keys=["Total albums","Number of albums designated 'Best New Music'"],axis=1,sort=True).plot(kind='bar',stacked=True,figsize=(12,10),rot=45);


# ### *But what does it all mean???*
# ## Well that's what I mean to explore in the subsequent kernels. Let me know your suggestions and logical deductions in the comments section!

# In[ ]:




