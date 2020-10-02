#!/usr/bin/env python
# coding: utf-8

# # Importing the important packages

# In[ ]:


import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


# # Reading the data

# In[ ]:


df=pd.read_csv('../input/online-streaming/Mo.csv',index_col=False)
df1=pd.read_csv('../input/online-streaming/Mo.csv',index_col=False)
df.head()


# # Checking the size of the dataframe.

# In[ ]:


df.shape


# # Droping the duplicate values from the dataframe.

# In[ ]:


df.drop_duplicates(inplace=True)


# # Knowing the data.

# In[ ]:


df.info()


# # Checking the null values in the dataframe.

# In[ ]:


df.isnull().sum()


# ## Dropping type column because it  only contains 0 values which are not at all considerable.

# In[ ]:


df.drop('Type',axis=1,inplace=True)


# # Getting the summary of our data from the describe method.

# In[ ]:


df.describe()


# ## Splitting the genres so that we can find unique genres.

# In[ ]:


s = df['Genres'].str.split(',').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Genres'
del df['Genres']
df_Genre = df.join(s)


# ## Splitting the languages so we can find unique languages.

# In[ ]:


d= df1['Language'].str.split(',').apply(Series, 1).stack()
d.index = d.index.droplevel(-1)
d.name = 'Language'
del df1['Language']
df1 = df1.join(d)


# In[ ]:


df1.head()


# ## We can se the change in the dataset after splitting the values.

# In[ ]:


df_Genre.head()


# In[ ]:


Netflix=df['Netflix'].sum()
print("Number of movies on Netflix:",Netflix)


# In[ ]:


Hulu=df['Hulu'].sum()
print("Number of movies on Hulu:",Hulu)


# In[ ]:


Prime_Video=df['Prime Video'].sum()
print("Number of movies on Amazon Prime Video:",Prime_Video)


# In[ ]:


Disney=df['Disney+'].sum()
print("Number of movies on Disney+:",Disney)


# In[ ]:


Total=Netflix+Hulu+Prime_Video+Disney
print("The total number of movies on these online platform:",Total)


# In[ ]:


num_platform = (Netflix,Hulu,Prime_Video,Disney)
col_names = ('Netflix','Hulu','Prime Video','Disney+')
PlatformList = list(zip(col_names,num_platform))
PlatformCounts = pd.DataFrame(data=PlatformList,columns=['Platform','Number of Movie'])
PlatformCounts


# In[ ]:


print("The movies available in these platforms are from the year:",df['Year'].min(),"to:",df['Year'].max())


# In[ ]:


print("The genre of the movies are:",df_Genre['Genres'].unique())
print("The number of unique genres are:",df_Genre['Genres'].nunique())


# In[ ]:


print("Average run time of movies on these platforms:",df['Runtime'].mean())


# # Plotted Histogram to see how the variables are distributed.

# In[ ]:


df.hist(color='brown',figsize=(10,10))


# > ## Changing the string age into int age.

# In[ ]:


df['Age'].replace("13+",13,inplace=True)
df['Age'].replace("18+",18,inplace=True)
df['Age'].replace("7+",7,inplace=True)
df['Age'].replace("all",0,inplace=True)
df['Age'].replace("16+",16,inplace=True)


# # No. of movies  released per year

# In[ ]:


df_year = pd.DataFrame(df.groupby(df['Year']).Title.nunique())
df_year.head()


# # Max movies produced in the year

# In[ ]:


df_year.nlargest(5,'Title')


# ## Line plot of the movies produced in a year

# In[ ]:


df_year.plot(title='Movies made per year',color='red',kind='line')


# ## Horizontal bar plot of the movies sorted by genres.

# In[ ]:


df_Genre['Genres'].value_counts().plot(kind='barh',figsize=(10,10))


# ## Horizontal bar plot of the movies grouped by age.

# In[ ]:


df['Age'].value_counts().plot(kind='barh')


# In[ ]:


print("The unique languages",df1['Language'].unique())
print("The number of unique languages:",df1['Language'].nunique())


# # To sort the movies according to the IMDb rating

# In[ ]:


@interact
def show_articles_more_than( x=3.1):
    return df.loc[df['IMDb'] >= x]
#interact will help you in toggling the age through the toggle button, it will only work in jupyter notebook


# # To sort movies according to the year

# In[ ]:


@interact
def show_articles_more_than(x=1000):
    print("The number of movies in the year:",x,"are:",df.loc[df['Year']==x].shape[0])
    return df.loc[df['Year'] ==x]


# ## To sort movies according to age.

# In[ ]:


@interact
def show_articles_more_than(x=10):
    return df.loc[df['Age']<=x]
#age=0 means every individual can see that movie


# ## Changing the data type of the rotten tomatoes rating from string to float.

# In[ ]:


df['Rotten Tomatoes'] = df['Rotten Tomatoes'].str.rstrip('%').astype('float') 


# ## Calculating the average rating of Rotten Tomatoes.

# In[ ]:


disney_avg_rt= round(df.loc[df['Disney+']==1]['Rotten Tomatoes'].mean(),1)
hulu_avg_rt=round(df.loc[df['Hulu']==1]['Rotten Tomatoes'].mean(),1)
netflix_avg_rt=round(df.loc[df['Netflix']==1]['Rotten Tomatoes'].mean(),1)
prime_avg_rt=round(df.loc[df['Prime Video']==1]['Rotten Tomatoes'].mean(),1)


# ## Calculating the average rating of IMDb on various platforms.

# In[ ]:


disney_avg_imdb= round(df.loc[df['Disney+']==1]['IMDb'].mean(),1)
hulu_avg_imdb=round(df.loc[df['Hulu']==1]['IMDb'].mean(),1)
netflix_avg_imdb=round(df.loc[df['Netflix']==1]['IMDb'].mean(),1)
prime_avg_imdb=round(df.loc[df['Prime Video']==1]['IMDb'].mean(),1)


# ## Movies available on Netflix.

# In[ ]:


Net=df.loc[df['Netflix']==1]
Net.head()


# ## Movies available on Hulu.

# In[ ]:


Hu=df.loc[df['Hulu']==1]
Hu.head()


# ## Movies available on Prime Video.

# In[ ]:


Pr=df.loc[df['Prime Video']==1]
Pr.head()


# ## Creating a dataframe that contains the number of movies, average imdb rating and the rotten tomatoes rating of the movies on various platforms.

# In[ ]:


# create dataframe:
no_platform = (Netflix,Hulu,Prime_Video,Disney)
col_names = ('Netflix','Hulu','Prime Video','Disney+')
avg_imdb = (netflix_avg_imdb,hulu_avg_imdb,prime_avg_imdb,disney_avg_imdb)
avg_roto = (netflix_avg_rt,hulu_avg_rt,prime_avg_rt,disney_avg_rt)
List = list(zip(col_names,no_platform,avg_imdb,avg_roto))
Counts =  pd.DataFrame(data=List,columns=['Platform','Number of Movie','Average IMDb rate','Average % Rotten Tomattoes rate'])
Counts


# ## A bar graph to show the movie counts on the platforms.

# In[ ]:


sns.barplot(x='Platform',y='Number of Movie',data=PlatformCounts)


# ## Normalizing the rating of rotten tomatoes in 10 so that it can be easily understood while plotting the scatter plots.

# In[ ]:


df['Rotten_t']=df['Rotten Tomatoes']/10


# ## IMDb Rating vs Rotten Tomatoes Rating Scatter Plot.

# In[ ]:


df.plot.scatter(x='IMDb', y='Rotten_t',title='Profit vs Vote Avg',color='DarkBlue',figsize=(6,5));

