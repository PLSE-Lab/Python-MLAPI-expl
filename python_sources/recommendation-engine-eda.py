#!/usr/bin/env python
# coding: utf-8

# <h1>Anime Recommendation Analysis & Simple Content-based recommendation engine</h1>

# <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTuvG5vqsmLDMcWOZ-8Thyq1nrdAI5P1d32SdVLwDJRbJbSJAPDow&s'>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


anime=pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')
rating=pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')


# <h1>Dataset Exploration</h1>

# In[ ]:


def first_look(df):
    print('dataset shape: \n')
    print('number of rows: ',df.shape[0],' number of columns: ',df.shape[1])
    print('dataset column names: \n')
    print(df.columns)
    print('columns data-type')
    print(df.dtypes)
    print('missing data')
    c=df.isnull().sum()
    print(c[c>0])


# <h3>Anime dataset</h3>

# In[ ]:


first_look(anime)


# <h3>rating dataset</h3>

# In[ ]:


first_look(rating)


# <h1>Data Cleaning & Preprocessing </h1>

# <ul>
#     <li>first we are going to change the data-type of episodes column in anime dataset</li>
#     <li>next we are going fill the missing rows in rating columns in anime by using the median of rating column in rating dataset</li>
#     <li>drop the the missing rows that we can not replace it by rating</li>
#     <li>handling genre column in anime dataset</li>
#     <li>drop duplicate animes in dataset</li> 

# In[ ]:


anime['episodes']=anime['episodes'].replace('Unknown',np.nan)
anime['episodes']=anime['episodes'].astype(float)


# In[ ]:


shared_id=anime[anime['anime_id'].isin(rating['anime_id'])]
shared_id['rating'].isnull().sum()


# <p>this is the number of missing ratings in anime dataset, that have an anime_id in rating column</p>

# In[ ]:


for i,j in zip(shared_id[shared_id['rating'].isnull()].index,shared_id[shared_id['rating'].isnull()]['anime_id'].values):
    median_value=rating[rating['anime_id']==j]['rating'].median()
    print('median value: ',median_value)
    anime.loc[i,'rating']=median_value
    print('index {} done!'.format(str(i)))


# In[ ]:


anime.dropna(subset=['rating'],axis=0,inplace=True)


# In[ ]:


anime['genre']=anime['genre'].str.replace(', ',',')


# In[ ]:


anime=anime.drop_duplicates('name')


# <h1>EDA</h1>

# In[ ]:


anime['type'].value_counts().plot.pie(autopct='%.1f%%',labels=None,shadow=True,figsize=(8,8))
plt.title('type of Animes in dataset')
plt.ylabel('')
plt.legend(anime['type'].value_counts().index.tolist(),loc='upper right')
plt.show()


# <p>As I expect TV has the largest percentage(30.4%), and we are going to analyze TV anime type </p>

# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(x='type',y='rating',data=anime)
plt.title('anime-type VS rating')
plt.show()


# In[ ]:


for i in anime['type'].unique().tolist():
    print('mean of '+str(i)+' :\n')
    print(anime[anime['type']==i]['rating'].mean())


# <p>let's see the top 20 genres in TV-animes</p>

# In[ ]:


TV_anime=anime[anime['type']=='TV']
TV_anime['genre'].value_counts().sort_values(ascending=True).tail(20).plot.barh(figsize=(8,8))
plt.title('genres of TV-Animes')
plt.xlabel('frequency')
plt.ylabel('genres')
plt.show()


# In[ ]:


TV_anime.drop('anime_id',axis=1).describe()


# <p>let's see which TV anime has the maximum episodes, and which has the minimum</p>

# In[ ]:


TV_anime[TV_anime['episodes']==TV_anime['episodes'].max()]


# In[ ]:


TV_anime[TV_anime['episodes']==TV_anime['episodes'].min()]


# <p>let's see which TV anime has the maximum rating, and which has the minimum</p>

# In[ ]:


TV_anime[TV_anime['rating']==TV_anime['rating'].max()]


# In[ ]:


TV_anime[TV_anime['rating']==TV_anime['rating'].min()]


# <p>let's see which TV anime has the maximum members, and which has the minimum</p>

# In[ ]:


TV_anime[TV_anime['members']==TV_anime['members'].max()]


# In[ ]:


TV_anime[TV_anime['members']==TV_anime['members'].min()]


# <p>let's see the distribution plots of rating, members</p>

# In[ ]:


fig=plt.figure(figsize=(13,5))
for i,j in zip(TV_anime[['rating','members']].columns,range(3)):
    ax=fig.add_subplot(1,2,j+1)
    sns.distplot(TV_anime[i],ax=ax)
    plt.axvline(TV_anime[i].mean(),label='mean',color='blue')
    plt.axvline(TV_anime[i].median(),label='median',color='green')
    plt.axvline(TV_anime[i].std(),label='std',color='red')
    plt.title('{} distribtion'.format(i))
    plt.legend()
plt.show()


# <p>let's use boxplot to see the outliers in rating and members columns</p>

# In[ ]:


fig=plt.figure(figsize=(13,5))
for i,j in zip(TV_anime[['rating','members']].columns,range(3)):
    ax=fig.add_subplot(1,2,j+1)
    sns.boxplot(i,data=TV_anime,ax=ax)
    plt.title('{} distribtion'.format(i))
plt.show()


# <p>since we have alot of outliers in rating column we are going to figure out them</p> 

# In[ ]:


import json
stats=TV_anime.drop('anime_id',axis=1).describe()
def show_outliers(df,col): 
    outliers={}
    for j,k in zip(df[col].index,df[col].tolist()):
        iqr=stats.loc['75%',col]-stats.loc['25%',col]
        upper_bound=stats.loc['75%',col]+iqr*1.5
        lower_bound=stats.loc['25%',col]-iqr*1.5
        if k>upper_bound :
            outliers[k]=['upper',df.loc[j,'name'],df.loc[j,'genre']]
        elif k<lower_bound:
            outliers[k]=['lower',df.loc[j,'name'],df.loc[j,'genre']]
    outliers=json.dumps(outliers)        
    print(outliers)
for i in TV_anime[['rating']].columns:
    print(i)
    print('-'*10)
    show_outliers(TV_anime,i)


# In[ ]:


iqr=stats.loc['75%','episodes']-stats.loc['25%','episodes']
upper_bound=stats.loc['75%','episodes']+iqr*1.5
lower_bound=stats.loc['25%','episodes']-iqr*1.5
episodes_lst=[]
for i in TV_anime['episodes'].values:
    if i<lower_bound:
        episodes_lst.append('small')
    elif i>upper_bound:
        episodes_lst.append('large')
    elif (i>lower_bound) and (i<upper_bound):
        episodes_lst.append('in-between')
    else:
        episodes_lst.append('no info!')
TV_anime['episodes_classification']=episodes_lst


# In[ ]:


TV_anime.head()


# In[ ]:


mean_lst=[]
mean_lst.append(TV_anime[TV_anime['episodes_classification']=='in-between']['rating'].mean())
mean_lst.append(TV_anime[TV_anime['episodes_classification']=='small']['rating'].mean())
mean_lst.append(TV_anime[TV_anime['episodes_classification']=='large']['rating'].mean())
mean_lst.append(TV_anime[TV_anime['episodes_classification']=='no info!']['rating'].mean())
plt.bar(['in-between','small','large','no info!'],mean_lst)
plt.title('mean comparison based on episodes classification')
plt.xlabel('episodes_classification')
plt.ylabel('average-rating')
plt.show()


# In[ ]:


TV_anime.drop('episodes_classification',axis=1,inplace=True)


# <h1>Content-Based Recommendation Engine</h1>

# <p>Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous, and can be commonly seen in online stores, movies databases and job finders. In this notebook, we will explore Content-based recommendation systems and implement a simple version of one using Python and the Pandas library.</p>

# <h1>RE-Preprocessing</h1>

# In[ ]:


TV_animes_df=TV_anime.copy()
TV_animes_df['genre']=TV_animes_df['genre'].str.split(',')
TV_animes_df.head()


# In[ ]:


for index, lst in zip(TV_animes_df.index,TV_animes_df['genre'].values):
    for genre in lst:
        TV_animes_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre


# In[ ]:


TV_animes_df = TV_animes_df.fillna(0)
TV_animes_df.head()


# In[ ]:


user_input=pd.DataFrame([{'name':'Fullmetal Alchemist: Brotherhood','user_rating':8.6},
                        {'name':'Tokyo Ghoul','user_rating':8}])
user_input


# In[ ]:


inputId = TV_anime[TV_anime['name'].isin(user_input['name'].tolist())]
user_input = pd.merge(inputId, user_input)
user_input = user_input.drop('genre', 1).drop('rating', 1).drop('episodes',1).drop('type',1).drop('members',1)
user_input


# In[ ]:


user_anime = TV_animes_df[TV_animes_df['name'].isin(user_input['name'].tolist())]
user_anime=user_anime.drop('rating',1)
user_anime


# In[ ]:


user_anime = user_anime.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
user_genre_table = user_anime.drop('anime_id', 1).drop('name', 1).drop('genre', 1).drop('type', 1).drop('episodes',1).drop('members',1)
user_genre_table


# In[ ]:


userProfile = user_genre_table.transpose().dot(user_input['user_rating'])
userProfile


# In[ ]:


genre_table = TV_animes_df.set_index(TV_animes_df['anime_id'])
genre_table = genre_table.drop('anime_id', 1).drop('name', 1).drop('genre', 1).drop('episodes', 1).drop('members',1).drop('rating',1).drop('type',1)
genre_table.head()


# <p>With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted average of every movie based on the input profile and recommend the top twenty movies that most satisfy it.</p>

# In[ ]:


recommendation_table_df = ((genre_table*userProfile).sum(axis=1))/(userProfile.sum())
recommendation_table_df.head()


# In[ ]:


recommendation_table_df = recommendation_table_df.sort_values(ascending=False)
#Just a peek at the values
recommendation_table_df.head()


# <p>let's now see the first 10 recommended TV animes</p>

# In[ ]:


TV_anime.loc[TV_anime['anime_id'].isin(recommendation_table_df.head(10).keys())]

