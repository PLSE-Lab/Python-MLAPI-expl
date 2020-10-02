#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df_anime=pd.read_csv('../input/anime.csv')
df_rate=pd.read_csv('../input/rating.csv')


# In[ ]:


df_anime.head()


# In[ ]:


df_anime.tail()


# In[ ]:


df_anime.describe()


# In[ ]:


df_anime.dtypes


# In[ ]:


df_anime.isnull().sum()


# In[ ]:


df_anime.dropna(subset=['rating','genre','type'],inplace=True)


# In[ ]:


df_anime.isnull().sum()


# In[ ]:


df_anime.dtypes


# In[ ]:


df_anime=df_anime[df_anime["episodes"]!='Unknown']


# In[ ]:


df_anime['episodes']=df_anime['episodes'].astype('int64')


# In[ ]:


df_anime.head()


# In[ ]:


df_anime.type.unique()


# # Exploring Movies of Anime

# In[ ]:


df_movies=df_anime[df_anime["type"]=='Movie']


# In[ ]:


df_movies.head()


# In[ ]:


df_movies.sort_values(['members','rating'],ascending=False)


# In[ ]:


plt.figure(figsize = (16,5))
df_check=df_movies.corr()
sns.heatmap(df_check, annot=True, fmt="g", cmap='viridis')
plt.show()


# In[ ]:


del df_movies['episodes']


# In[ ]:


del df_movies['anime_id']


# In[ ]:


del df_movies['type']


# In[ ]:


df_movies


# In[ ]:


plt.figure(figsize = (16,5))
df_check=df_movies.corr()
sns.heatmap(df_check, annot=True, fmt="g", cmap='viridis')
plt.show()


# In[ ]:


print(df_movies['members'].mean())
print(df_movies['rating'].mean())


# In[ ]:


def f(row):
    val=0
    if row['rating']>=9.5:
        val=1
    elif row['rating']>=9:
        val=2
    elif row['rating']>=8.5:
        val=3
    elif row['rating']>=8:
        val=4
    elif row['rating']>=7.5:
        val=5
    elif row['rating']>=7:
        val=6
    elif row['rating']>=6.5:
        val=7
    return val

df_movies['hexcode'] = df_movies.apply(f, axis=1)


# In[ ]:


df_movies.head()


# In[ ]:


df_considerable_movies=df_movies[(df_movies['members']>df_movies['members'].mean()) & (df_movies['rating']>df_movies['rating'].mean())]
df_considerable_movies.head()


# In[ ]:


colordict={1:'#30E6B1',2:'#A1F5AD',3:'#E4B06E',4:'#EA4052',5:'#42C6CF',6:'#C6D359',7:'#45B39C'}
plt.figure(figsize=(16, 8))
plt.scatter(df_considerable_movies['rating'],df_considerable_movies['members'],s=df_considerable_movies['rating']**2,alpha=0.8,c=df_considerable_movies['hexcode'])
plt.xlabel('Ratings')
plt.ylabel('Audience')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 8))
sns.lmplot('rating','members', data=df_considerable_movies, hue='hexcode',fit_reg=False,size=8)
plt.show()


# In[ ]:


df_rate.describe()


# In[ ]:


df_movies=df_anime[df_anime["type"]=='Movie']
df_movies_rec=pd.merge(df_movies,df_rate,on='anime_id')


# In[ ]:


mainmean=df_movies_rec['rating_x'].mean()
print(mainmean)


# In[ ]:


df_movies_rec.head()


# In[ ]:


df_movies_rec.describe()


# In[ ]:


df_movies_rec.dtypes
df_movies_rec['rating_x']=df_movies_rec['rating_x'].astype('float64')
df_movies_rec['rating_y']=df_movies_rec['rating_y'].astype('float64')


# In[ ]:


df_movies_rec.name.unique()
print(len(df_movies_rec))


# In[ ]:


count_node=pd.DataFrame(df_movies_rec.groupby('name')['rating_y'].mean())
count_node['num of ratings'] = pd.DataFrame(df_movies_rec.groupby('name')['rating_y'].count())


# In[ ]:


count_node.sort_values(['rating_y'],ascending=False)
count_node.head()


# In[ ]:


count_node.reset_index(inplace=True)


# In[ ]:


count_node.head()


# In[ ]:


count_node.describe()


# In[ ]:


plt.figure(figsize=(18,5))
count_node['num of ratings'].hist(bins=300,color='g',alpha=0.6)
plt.title('NUMBER OF RATING DISTRIBUTION OVER THE COUNT')
plt.xlim(0,1600)
plt.show()


# In[ ]:


plt.figure(figsize=(18,5))
count_node['rating_y'].hist(bins=30,color='y',alpha=0.4)
plt.title('RATING DISTRIBUTION OVER THE COUNT')
plt.show()


# In[ ]:


genre_sets=set()
for i in df_movies['genre']:
    k=i.split(',')
    for i in k:
        genre_sets.add(str(i))


# In[ ]:


plt.figure(figsize=(20,5))
sns.jointplot(x='rating_y',y='num of ratings',data=count_node,color='#2ecc71',size=10,kind='reg')
plt.show()


# In[ ]:


df_genre_mov =pd.DataFrame(df_movies[['name','genre']])


# In[ ]:


df_genre_mov.reset_index(inplace=True)
df_genre_mov.head()


# In[ ]:


df_genre_mov.set_index('name',inplace=True)


# In[ ]:


df_main=pd.merge(df_rate,df_movies,on='anime_id')
df_main.head(1)


# In[ ]:


del df_main['anime_id']
del df_main['rating_x']
del df_main['genre']
del df_main['members']
del df_main['type']
del df_main['episodes']


# In[ ]:


df_main.head(1)


# # Generating suggestion on mainmean rating and comparing mean

# In[ ]:


def rec_mov(name_):
    rating_name_=count_node[count_node['name']==name_]
    #print(rating_name_)
    k_genre_list=df_genre_mov.loc[name_]['genre'].split(',')
    for i in range(len(k_genre_list)):
        k_genre_list[i]=k_genre_list[i].strip()
    df_suggest=df_genre_mov[df_genre_mov.isin(k_genre_list)]
    df_suggest.dropna(subset=['genre'],inplace=True)
    df_suggest.reset_index(inplace=True)
    df_joint=pd.merge(df_suggest,count_node,on='name')
    df_joint['rating_y']=df_joint['rating_y'].astype('float64')
    check=float(rating_name_['rating_y'])
    df_joint_1=df_joint.copy()
    del df_joint_1['genre']
    del rating_name_['name']
    del df_joint_1['index']
   #print(df_joint_1)
    df=df_main.pivot_table(index='user_id',columns='name',values='rating_y')
    similar_df=df.corrwith(df[name_])
    corr_df = pd.DataFrame(similar_df, columns=['Correlation'])
    corr_df.dropna(inplace=True)
    corr_df.reset_index(inplace=True)
    corr_df=corr_df.merge(count_node,on='name')
   #print(corr_df.head(3))
    show_recommendation=corr_df[corr_df['num of ratings'] > 100].sort_values(by=['Correlation','num of ratings'], ascending=False).head(10)
    del show_recommendation['Correlation']
    del show_recommendation['num of ratings']
    del show_recommendation['rating_y']
    return show_recommendation.head(10)
    


# In[ ]:


def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed

set_pandas_options()
rec_mov('Kimi no Na wa.')


# In[ ]:


rec_mov('Koe no Katachi')


# In[ ]:


rec_mov('.hack//G.U. Trilogy')


# In[ ]:


rec_mov('Ookami Kodomo no Ame to Yuki')


# In[ ]:





# In[ ]:




