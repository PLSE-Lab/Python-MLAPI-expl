#!/usr/bin/env python
# coding: utf-8

# ## Netflix and Chill

# Netflix is one of the biggest video streaming platform worldwide. In recent time the lingo *Netflix and chill* got really famous so I thought let's do some Exploratry Data Analysis(EDA) on Netlfix data and find some insights if possible. So I found some of very intresting insights let's check it out.

# First thing first,so let's begin with importing pandas.

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# So our dataset contains following features:
# * Show_id : Show Id
# * type : Movies/Series
# * title : Title 
# * director : Director of it
# * cast : Casts of movie/series
# * country : In which country it released
# * date_added : On which date movie added on Netflix
# * release_year : Year in which it released
# * rating : Rating of movies/ series
# * duration : Runtime of movie and series
# * listed_in : Genre
# * description : Abstract idea of movie/series
# 

# Right now we dont need null values it has void importance for us. So,Dropping Null values.

# In[ ]:


data.dropna(inplace=True)


# ## Visualization

# Importing Seaborn and Matplotlib for visualising data.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# ## EDA

# Exploratory Data Analysis on this netflix data to get some meaningfull insights. There are tons of factor we can conider but this blog contains some of basic EDA.

# Let's begin with EDA process with rating.

# In[ ]:


ratingsdf = data['rating'].value_counts()


# By doing data['ratings'].value_counts(), we are getting ratings and thier respective count in our entire dataset. This give us perfect data for ploting graph of their ratings.

# In[ ]:


ratingsdf.columns=['Rating','Count']
df = pd.DataFrame(ratingsdf)
df['Count'] = df['rating']
df['Rating'] = df.index
df.reset_index(inplace=True)
df.drop(['index','rating'],axis=1,inplace=True)
df


# Above shown are 14 rating criteria of Movies/Series. What we are doing here is counting there occurence and ploting them to get insight of how many movies/series belongs to particulary rating. Below is the overall representation of ratings.

# In[ ]:


plt.figure(figsize=(12,7))
sns.barplot(data=df,x='Rating',y='Count')


# In[ ]:


plt.figure(figsize=(12,7))
sns.countplot(data=data,x='rating')


# In overall, most of the content is of TV-MA rating followed by TV-14.

# Now, let's see how many movies belongs to each ratings.

# In[ ]:


plt.figure(figsize=(12,7))
sns.countplot(data=data[data['type']=='Movie'],x='rating')


# Let's do this for Series

# In[ ]:


plt.figure(figsize=(12,7))
sns.countplot(data=data[data['type']=='TV Show'],x='rating')


# In[ ]:


data[data['type']=='TV Show']['rating'].unique()


# So TV Show only have following ratings
# * TV-MA
# * TV-14
# * TV-G
# * TV-PG
# * TV-Y7
# * TV-Y
# * R

# Let's look into country wise analysis of ratings irrespective of its type.

# Let's make a common function where user can pass argument like category as in feature by which they want to extract insights and argument as value they are searching for. For example :
#     *Rate('country','India')* when Rate function will be called it will treat 'country' as feature and 'India' as specific value of whose data is required.

# In[ ]:


def Rate(cat,arg):
    plt.figure(figsize=(12,7))
    sns.countplot(data=data[data[cat]==arg],x='rating')


# In[ ]:


Rate('country','India')


# In[ ]:


Rate('duration','1 Season')


# Let's move forward with other features. Now will try to take some insights from listed_in feature.

# So, here what we are doing is, first we are seeing its distribution with grouped genre as given in dataset.

# In[ ]:


listeddf = data['listed_in'].value_counts()
listeddf.columns=['listed_in','Count']
df = pd.DataFrame(listeddf)
df['Count'] = df['listed_in']
df['Listed_in'] = df.index
df.reset_index(inplace=True)
df.drop(['index','listed_in'],axis=1,inplace=True)
df


# Above, this table shows Listed in and its respective count. Below is its visualization.

# In[ ]:


plt.figure(figsize=(15,100))
sns.barplot(data=df,y='Listed_in',x='Count')


# We can also do these plotting with single line of code. Shown below

# In[ ]:


plt.figure(figsize=(20,100))
sns.countplot(data=data,y='listed_in')


# By nature we have multiple category of same movie/series. What we can do is seperate every genre seperately and see how many movie/tv shows belong to which category.

# In[ ]:


lists = []
listed = data['listed_in'].str.split(',')
for li in listed:
    for l in li:
        lists.append(l)
df= pd.DataFrame(data=lists,columns=['Genre'])
df


# In[ ]:


plt.figure(figsize=(10,35))
sns.countplot(data=df,y='Genre')


# As we made a common funtion for ratings we can do the same for listed_in. With same idea and parameter.

# In[ ]:


def Genre(cat='all',arg='all'):
    if cat == 'all' or arg=='all':
        plt.figure(figsize=(10,35))
        sns.countplot(data=df,y='Genre')
    else:
        lists = []
        listed = data[data[cat]==arg]['listed_in'].str.split(',')
        for li in listed:
            for l in li:
                lists.append(l)
        df= pd.DataFrame(data=lists,columns=['Genre'])
        df
        plt.figure(figsize=(35,20))
        sns.countplot(data=df,y='Genre')
Genre('duration','1 Season')


# In[ ]:


Genre('country','India')


# Now let's move forward with release_year. This feature basically tells us about when does this content was released in. We will follow same process as we did above.

# In[ ]:


df = data['release_year'].value_counts()
df = pd.DataFrame(df)
df['Release_year'] = df.index
df['Count'] = df['release_year']
df.reset_index(inplace=True)
df.drop(['index','release_year'],inplace=True,axis=1)
df


# This table represent Release_year with its number of releases.

# In[ ]:


plt.figure(figsize=(12,15))
sns.countplot(data=data,y='release_year')


# 2017 has the highest number of releases followd by 2016 and 2018

# Let's make same common function for it.

# In[ ]:


def Year(cat='all',arg='all'):
    if cat =='all' or arg=='all':
        plt.figure(figsize=(12,15))
        sns.countplot(data=data,y='release_year')
    else:
        df = data[data[cat]==arg]['release_year'].value_counts()
        df = pd.DataFrame(df)
        df['Release_year'] = df.index
        df['Count'] = df['release_year']
        df.reset_index(inplace=True)
        df.drop(['index','release_year'],inplace=True,axis=1)
        df
        plt.figure(figsize=(30,15))
        sns.barplot(data=df,x='Release_year',y='Count')


# In[ ]:


Year('country','India')


# Let's move forward to country wise EDA

# In[ ]:


country = data['country']


# In[ ]:


count = country.str.split(',')


# In[ ]:


country = []
for countie in count:
    for c in countie:
        country.append(c)


# In[ ]:


df = pd.DataFrame(data=country,columns=['Country'])


# In[ ]:


df = df['Country'].value_counts()
df = pd.DataFrame(df)
df['Count'] = df['Country']
df['Country'] = df.index
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
df


# In[ ]:


df[df['Country']=='India']


# In[ ]:


data['country'].values


# In[ ]:


plt.figure(figsize=(15,25))
# plt.xlim(2000)
sns.barplot(data=df[df['Count']>50],x='Count',y='Country')


# United States, India and United Kingdom are countries with highest movies/Tv shows released on. Above chart was for countries having Movies/TV shows more than 50 count. And below graph represents Movies/TV shows with less than 50 count of items.

# In[ ]:


plt.figure(figsize=(15,25))
# plt.xlim(2000)
sns.barplot(data=df[df['Count']<50],x='Count',y='Country')


# Lets check how many movies and TV shows are there in total.

# In[ ]:


sns.countplot(data=data,x='type')


# Netflix have more Movies than TV shows. Lets check rating and type distribution.

# In[ ]:


plt.figure(figsize=(12,7))
sns.countplot(data=data,x='rating',hue='type')


# In[ ]:


def Type(cat='all',arg='all'):
    if cat =='all' or arg =='all':
        sns.countplot(data=data,x='type')
    else:
        sns.countplot(data=data[data[cat]==arg],x='type')
Type('country','India')


# In[ ]:


plt.figure(figsize=(10,25))
sns.countplot(data=data,y='release_year',hue='type')


# In[ ]:



sns.countplot(data=data[data['country']=='India'],x='country',hue='type')


# Let's see which director has more movies on netflix

# In[ ]:


df = data['director'].value_counts()
df.columns=['Director','count']
df = pd.DataFrame(df)
df['Count'] = df['director']
df['Director']=df.index
df.reset_index(inplace=True)
df.drop(['index','director'],axis=1,inplace=True)
df


# In[ ]:


plt.figure(figsize=(20,7))
sns.barplot(data=df[:10],x='Director',y='Count')


# Common function for same to do analysis based on arguments we passed.

# In[ ]:


data['duration'].nunique()


# In[ ]:


data.duration


# In[ ]:


plt.figure(figsize=(15,35))
sns.countplot(data=data,y='duration')


# In[ ]:


tv=data[data['type']=='TV Show']


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(data=tv,x='duration')


# In[ ]:


data.info()


# In[ ]:


data['cast'].nunique()


# In[ ]:


casts = data['cast'].str.split(',')


# In[ ]:


actors = []
for cast in casts:
    for actor in cast:
        actors.append(actor)


# In[ ]:


df = pd.DataFrame(actors,columns=['Actor'])
df = df['Actor'].value_counts()
df = pd.DataFrame(df)
df['Count'] = df['Actor']
df['Actor'] = df.index
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
df


# In[ ]:


plt.figure(figsize=(20,7))
sns.barplot(data=df[:10],x='Actor',y='Count')


# uhm, Seems like Bollywood actors have more movies on netflix than other film industry.

# ## End Note:
# 
# This was some basic over view of EDA. We got some insights which are amazing. Please do share your valuable comments. I know there is little less explanation of some stuff. And, please do tell in comments if you want some details in this and part 2 of EDA in the same dataset. If you enjoy reading this please upvote it. Thank you :)

# In[ ]:




