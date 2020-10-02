#!/usr/bin/env python
# coding: utf-8

# # Getting started

# In[ ]:


import numpy as np 
import pandas as pd 
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')


# In[ ]:


df.head(10)


# In[ ]:


df.shape


# Let's take a look at the datatype of each column

# In[ ]:


df.dtypes


# Check for NA's in each column

# In[ ]:


df.isnull().sum()


# In[ ]:


df.rating[df['rating']=='UR'].value_counts()


# We have 10 instances where the rating of a show or movie is missing. We will include these instances as 'Unrated' (UR). 
# 
# This will take the total of UR rating to 17.

# In[ ]:


df['rating'] = df['rating'].fillna('UR')


# We will now divide the dataset into two groups: one containing movies on Netflix and the other containing only the TV shows.
# 
# We will then clean the subsets one by one.

# In[ ]:


movies_df = df[df['type'] == 'Movie']
tv_df = df[df['type'] == 'TV Show']


# # Working with the Movies subset

# In[ ]:


movies_df.shape


# Checking for the NA values
# 

# In[ ]:


movies_df.isna().sum()


# We will fill the NA values in the 'country' column with 'Undefined'.
# 
# We will also fill the empty instances in 'date_added' column with the year as 2020.

# In[ ]:


movies_df['country'] = movies_df['country'].fillna('Undefined')


# In[ ]:


movies_df['date_added'] = movies_df['date_added'].fillna('January 1, 2020')


# Adding two more columns to the movies subset: one containing the month the content was added and other containing the year it was added in

# In[ ]:


movies_df['month_added'] = pd.to_datetime(movies_df['date_added']).dt.month_name()
movies_df['year_added'] = pd.to_datetime(movies_df['date_added']).dt.year


# In[ ]:


movies_df.head(10)


# In order to visualize the country and genres of the movies, we first need to clean the columns and bring them in an uniform format.
# 
# By taking a closer look at the dataset we see that the comma spacing in some instances is different than others.

# In[ ]:


movies_df['country'] = movies_df.country.str.replace(", | ,", ",")
movies_df['listed_in'] = movies_df.listed_in.str.replace(", | ,", ",")


# Now once we have the column in an uniform format, we split the column and create a new dataframe containg the show_id and the country names where a movie was released

# In[ ]:


movies_country_df = pd.DataFrame(movies_df.country.str.split(',').tolist(), index=movies_df.show_id).stack()
movies_country_df = movies_country_df.reset_index([0, 'show_id'])
movies_country_df.columns = ['show_id', 'country']


# # Which are the countries where most number of movies are available?

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

#for count plot
ax = sns.countplot(x="country", data=movies_country_df, palette='Blues_d', order = movies_country_df.country.value_counts().iloc[:20].index)

#aesthetics of the plot
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Countries with most movies available', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Countries', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))


#setting count values on each bar
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# We can see that United States has the most number of movies available on Netflix. India comes a distant second with less than half the movies available than in the US.

# In[ ]:


movies_genre_df = pd.DataFrame(movies_df.listed_in.str.split(',').tolist(), index=movies_df.show_id).stack()
movies_genre_df = movies_genre_df.reset_index([0, 'show_id'])
movies_genre_df.columns = ['show_id', 'genre']


# # Which movie genres are the most frequent on Netflix?

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

#for count plot
ax = sns.countplot(x="genre", data=movies_genre_df, palette="PuBu_d", order = movies_genre_df.genre.value_counts().iloc[:20].index)

#aesthetics of the plot
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Most frequent movie genres', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Genres', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))


#setting count values on each bar
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# # Which year saw the most number of movies getting added on Netflix?

# In[ ]:


#count of content added every year


plt.figure(figsize=(20,10))
sns.set(style="white")

#for count plot
ax = sns.countplot(x="year_added", data=movies_df, palette='PuBu_d', order = movies_df.year_added.value_counts().index)

#aesthetics of the plot
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which year saw the most movies being added?', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Year', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))


#setting count values on each bar
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# We see that the content on this platform has increased every year. We also see a steep increase in the number of movies getting added in 2017 as compared to 2016. The number of movies saw a 477% increase from 2016 to 2019.

# # Which months saw the most number of movies getting added on Netflix? (All years inclusive)

# In[ ]:


#count of content added every month


plt.figure(figsize=(20,10))
sns.set(style="white")

#for count plot
ax = sns.countplot(x="month_added", data=movies_df, palette='PuBu_d', order = movies_df.month_added.value_counts().index)

#aesthetics of the plot
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which months saw the most movies being added?', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Month', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))


#setting count values on each bar
for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# # Heatmap to drill down into the month-year relationship

# In[ ]:


plt.figure(figsize=(20,10))

movies_heatmap_df = movies_df.groupby(['year_added', 'month_added']).size().reset_index(name='count')

month_ordered = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August','September','October','November','December']

movies_heatmap_df['month_added'] = movies_heatmap_df['month_added'].astype(pd.api.types.CategoricalDtype(month_ordered))

plt.figure(figsize=(40,14))
sns.set(font_scale=2.25)

ax = pd.pivot_table(movies_heatmap_df, values= 'count', index=['month_added'], columns='year_added')
sns.heatmap(ax, cmap="PuBu", annot = True, fmt = ".1f", annot_kws={'size':22})


# # A genrewise look at how the movies got added over the years  

# In[ ]:


movies_genrewise_df = movies_df[['show_id', 'year_added']]
movies_genrewise_df = pd.merge(movies_genre_df, movies_genrewise_df, on='show_id')


# In[ ]:


movies_genrewise_df = movies_genrewise_df.groupby(['year_added', 'genre']).size().reset_index(name='count')


# In[ ]:


genre_list = ['International Movies','Dramas','Comedies','Documentaries','Action & Adventure', 'Independent Movies', 'Thrillers','Children & Family Movies', 'Romantic Movies', 'Stand-Up Comedy' ]
movies_genrewise_df = movies_genrewise_df[movies_genrewise_df['genre'].isin(genre_list)]


# In[ ]:


g = sns.FacetGrid(movies_genrewise_df, col= 'genre', hue='genre', col_wrap=5, height = 4.5, aspect = 1.0, sharex=False, sharey = False)
g = g.map(plt.plot, 'year_added', 'count')
g = g.map(plt.fill_between, 'year_added', 'count', alpha=0.2)
g = g.set_titles("{col_name}")
g = g.set(yticks = np.arange(0,800,100))


plt.subplots_adjust(top=0.85)
g.fig.suptitle('Yearly addition of most popular Movie genres on Netflix', fontsize = 20.5) 


# We see that all the categories have seen a rise in their numbers in 2019. We see that the 'International Movies' genre saw its highest numbers in 2018 rather than in 2019 like the other genres. 
# 
# 
# 'Dramas', 'Independent Movies' and 'Thrillers' have their been on Netflix since 2008 while genres like 'Romantics Movies', 'Stand-Up Comedy' and 'Action & Adventure' appear to be fairly new additions on Netflix.

# # Duration between a movie's release and its addition on Netflix

# In[ ]:


movies_df['duration'] = movies_df['year_added'] - movies_df['release_year']


# In[ ]:


duration_df = pd.Series(pd.cut(movies_df['duration'], np.arange(0,80,5)).value_counts()).reset_index()


# In[ ]:


duration_df = duration_df.rename(columns= {'index': 'bins', 'duration':'counts'})


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.barplot(x = duration_df['bins'], y = duration_df['counts'] ,data=duration_df, palette='PuBu_d')

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title("Duration (in years) between a movie's release and its addition on Netflix", fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Years', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# We observe an interesting fact about 13 movies being 70 to 75 years old when they were added on Netflix. 
# 
# Let's take a look at those 13 movies

# # Oldest movies on Netflix

# In[ ]:


movies_df[movies_df.release_year.isin(movies_df['release_year'].nsmallest(13))]


# # Who are the directors with most movies on Netflix (Worldwide)?

# In[ ]:


movies_director_df = movies_df[['show_id', 'director']]
movies_director_df = movies_director_df.dropna()


# In[ ]:


movies_director_df['director'] = movies_director_df.director.str.replace(", | ,", ",")


# In[ ]:


movies_director_df = pd.DataFrame(movies_director_df.director.str.split(',').tolist(), index=movies_director_df.show_id).stack()
movies_director_df = movies_director_df.reset_index([0, 'show_id'])
movies_director_df.columns = ['show_id', 'director']


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="director", data=movies_director_df, palette='PuBu_d', order = movies_director_df.director.value_counts().iloc[:20].index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which directors have the most movies on Netflix?', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Directors', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+0.5), fontsize = 12.5)


# # A look at the directors in the countries with highest amount of movies

# In[ ]:


movies_director_df = pd.merge(movies_country_df, movies_director_df, on='show_id', how = 'inner')
movies_director_df = movies_director_df.groupby(['director','country']).size().reset_index(name='count')


# In[ ]:


country_list = ['United States', 'India', 'United Kingdom', 'France', 'Canada', 'Spain' ]
movies_director_df = movies_director_df[movies_director_df['country'].isin(country_list)]


# In[ ]:


movies_director_df = movies_director_df.groupby(['country']).apply(lambda x: x.sort_values(['count'],ascending = False)).reset_index(drop = True)
movies_director_df = movies_director_df.groupby(['country']).head(10)


# In[ ]:


f, axes = plt.subplots(2, 3, figsize=(25, 12), sharex=False)

ax1 = sns.barplot(x = 'count', y = 'director', data = movies_director_df[movies_director_df['country'] == 'United States'], palette="PuBu_d", ax=axes[0, 0]).set_title("United States")

ax2 = sns.barplot(x = 'count', y = 'director', data = movies_director_df[movies_director_df['country'] == 'India'], palette="PuBu_d", ax=axes[0, 1]).set_title("India")

ax3 = sns.barplot(x = 'count', y = 'director', data = movies_director_df[movies_director_df['country'] == 'United Kingdom'], palette="PuBu_d", ax=axes[0, 2]).set_title("United Kingdom")

ax4 = sns.barplot(x = 'count', y = 'director', data = movies_director_df[movies_director_df['country'] == 'France'], palette="PuBu_d", ax=axes[1, 0]).set_title("France")

ax5 = sns.barplot(x = 'count', y = 'director', data = movies_director_df[movies_director_df['country'] == 'Canada'], palette="PuBu_d", ax=axes[1, 1]).set_title("Canada")

ax6 = sns.barplot(x = 'count', y = 'director', data = movies_director_df[movies_director_df['country'] == 'Spain'], palette="PuBu_d", ax=axes[1, 2]).set_title("Spain")


plt.setp(axes, xticks=np.arange(0,16,2))
plt.tight_layout()


# # Artists with most content on Netflix (Worldwide)

# In[ ]:


movies_cast_df = movies_df[['show_id', 'cast']]
movies_cast_df = movies_cast_df.dropna()


# In[ ]:


movies_cast_df['cast'] = movies_cast_df.cast.str.replace(", | ,", ",")


# In[ ]:


movies_cast_df = pd.DataFrame(movies_cast_df.cast.str.split(',').tolist(), index=movies_cast_df.show_id).stack()
movies_cast_df = movies_cast_df.reset_index([0, 'show_id'])
movies_cast_df.columns = ['show_id', 'cast']


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

#for count plot
ax = sns.countplot(x="cast", data=movies_cast_df, palette='PuBu_d', order = movies_cast_df.cast.value_counts().iloc[:20].index)

#aesthetics of the plot
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which artists have the most movies on Netflix?', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Actors', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))


#setting count values on each bar
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+0.5), fontsize = 12.5)


# # A look at the artists in the countries with highest amount of movies

# In[ ]:


movies_cast_df = pd.merge(movies_country_df, movies_cast_df, on='show_id', how = 'inner')
movies_cast_df = movies_cast_df.groupby(['cast','country']).size().reset_index(name='count')


# In[ ]:


country_list = ['United States', 'India', 'United Kingdom', 'France', 'Canada', 'Spain']
movies_cast_df = movies_cast_df[movies_cast_df['country'].isin(country_list)]


# In[ ]:


movies_cast_df = movies_cast_df.groupby(['country']).apply(lambda x: x.sort_values(['count'],ascending = False)).reset_index(drop = True)
movies_cast_df = movies_cast_df.groupby(['country']).head(10)


# In[ ]:


f, ax = plt.subplots(2, 3, figsize=(25, 12), sharex=False)

ax1 = sns.barplot(x = 'count', y = 'cast', data = movies_cast_df[movies_cast_df['country'] == 'United States'], palette="PuBu_d", ax=ax[0, 0]).set_title("United States")

ax2 = sns.barplot(x = 'count', y = 'cast', data = movies_cast_df[movies_cast_df['country'] == 'India'], palette="PuBu_d", ax=ax[0, 1]).set_title("India")

ax3 = sns.barplot(x = 'count', y = 'cast', data = movies_cast_df[movies_cast_df['country'] == 'United Kingdom'], palette="PuBu_d", ax=ax[0, 2]).set_title("United Kingdom")

ax4 = sns.barplot(x = 'count', y = 'cast', data = movies_cast_df[movies_cast_df['country'] == 'France'], palette="PuBu_d", ax=ax[1, 0]).set_title("France")

ax5 = sns.barplot(x = 'count', y = 'cast', data = movies_cast_df[movies_cast_df['country'] == 'Canada'], palette="PuBu_d", ax=ax[1, 1]).set_title("Canada")

ax6 = sns.barplot(x = 'count', y = 'cast', data = movies_cast_df[movies_cast_df['country'] == 'Spain'], palette="PuBu_d", ax=ax[1, 2]).set_title("Spain")


plt.setp(ax, xticks=np.arange(0,32,2))
plt.tight_layout()


# # How the movies are divided based on their audience ratings

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="rating", data=movies_df, palette='PuBu_d', order = movies_df.rating.value_counts().index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('How the TV Shows are divided based on their audience ratings', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Year', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# # Working with the TV Shows subset

# In[ ]:


tv_df.head(10)


# In[ ]:


tv_df.shape


# Checking for the NA values
# 

# In[ ]:


tv_df.isna().sum()


# We will fill the NA values in the 'country' column with 'Undefined'.
# 
# We will also fill the empty instances in 'date_added' column with the year as 2020.

# In[ ]:


tv_df['date_added'] = tv_df['date_added'].fillna('January 1, 2020')


# In[ ]:


tv_df['country'] = tv_df['country'].fillna('Undefined')


# Adding two more columns to the tv show subset: one containing the month the content was added and other containing the year it was added in

# In[ ]:


tv_df['month_added'] = pd.to_datetime(tv_df['date_added']).dt.month_name()
tv_df['year_added'] = pd.to_datetime(tv_df['date_added']).dt.year


# In order to visualize the country and genres of the tv shows, we first need to clean the columns and bring them in an uniform format.
# 
# By taking a closer look at the dataset we see that the comma spacing in some instances is different than others.

# In[ ]:


tv_df['listed_in'] = tv_df.listed_in.str.replace(", | ,", ",")
tv_df['country'] = tv_df.country.str.replace(", | ,", ",")


# Now once we have the column in an uniform format, we split the column and create a new dataframe containg the show_id and the country names where the tv show was released

# # Which are the countries where most number of TV Shows are available?

# In[ ]:


tv_country_df = pd.DataFrame(tv_df.country.str.split(',').tolist(), index=tv_df.show_id).stack()
tv_country_df = tv_country_df.reset_index([0, 'show_id'])
tv_country_df.columns = ['show_id', 'country']


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="country", data=tv_country_df, palette='YlGn_d', order = tv_country_df.country.value_counts().iloc[:20].index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Countries with most TV shows available', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Countries', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# We again observe that United States has the most number of TV Shows available. A concern we have here is that there are many instances where the countries in which the content is available are missing and that category is second highest in terms of its occurrence.

# # Which TV Show genres are the most frequent on Netflix?

# In[ ]:


tv_genre_df = pd.DataFrame(tv_df.listed_in.str.split(',').tolist(), index=tv_df.show_id).stack()
tv_genre_df = tv_genre_df.reset_index([0, 'show_id'])
tv_genre_df.columns = ['show_id', 'genre']


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="genre", data=tv_genre_df, palette="YlGn_d", order = tv_genre_df.genre.value_counts().iloc[:20].index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Most frequent TV show genres', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Genres', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# # Which year saw the most number of TV Shows getting added on Netflix?

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="year_added", data=tv_df, palette='YlGn_d', order = tv_df.year_added.value_counts().index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which year saw the most TV shows being added?', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Year', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# # Which months saw the most number of TV Shows getting added on Netflix? (All years inclusive)

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="month_added", data=tv_df, palette='YlGn_d', order = tv_df.month_added.value_counts().index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which months saw the most TV shows being added?', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Month', fontsize = 17.5)
plt.ylabel('')

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)


# # Heatmap to drill down into the month-year relationship

# In[ ]:


plt.figure(figsize=(20,10))

tv_heatmap_df = tv_df.groupby(['year_added', 'month_added']).size().reset_index(name='count')

month_ordered = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August','September','October','November','December']

tv_heatmap_df['month_added'] = tv_heatmap_df['month_added'].astype(pd.api.types.CategoricalDtype(month_ordered))

plt.figure(figsize=(40,14))
sns.set(font_scale=2.25)

ax = pd.pivot_table(tv_heatmap_df, values= 'count', index=['month_added'], columns='year_added')
sns.heatmap(ax, cmap="YlGn", annot = True, annot_kws = {'size':22})


# # A genrewise look at how the TV Shows got added over the years  

# In[ ]:


tv_genrewise_df = tv_df[['show_id', 'year_added']]
tv_genrewise_df = pd.merge(tv_genre_df, tv_genrewise_df, on='show_id')


# In[ ]:


tv_genrewise_df = tv_genrewise_df.groupby(['year_added', 'genre']).size().reset_index(name='count')


# In[ ]:


genre_list = ['International TV Shows','TV Dramas','TV Comedies','Crime TV Shows','Docuseries', 'Romantic TV Shows','British TV Shows',"Kids' TV", 'Reality TV', 'Korean TV Shows' ]
tv_genrewise_df = tv_genrewise_df[tv_genrewise_df['genre'].isin(genre_list)]


# In[ ]:



g = sns.FacetGrid(tv_genrewise_df, col= 'genre', hue='genre', col_wrap=5, height = 4.5, aspect = 1.0, sharex = False, sharey = False)
g = g.map(plt.plot, 'year_added', 'count')
g = g.map(plt.fill_between, 'year_added', 'count', alpha=0.2).set_titles("{col_name} country")
g = g.set_titles("{col_name}")
g = g.set(xticks=np.arange(2012, 2021, 1), yticks = np.arange(0,450,50))
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Yearly addition of most popular TV Show genres on Netflix', fontsize = 20.5) 


# # Artists with the most TV content on Netflix (Worldwide)

# In[ ]:


tv_cast_df = tv_df[['show_id', 'cast']]
tv_cast_df = tv_cast_df.dropna()


# In[ ]:


tv_cast_df['cast'] = tv_cast_df.cast.str.replace(", | ,", ",")


# In[ ]:


tv_cast_df = pd.DataFrame(tv_cast_df.cast.str.split(',').tolist(), index=tv_cast_df.show_id).stack()
tv_cast_df = tv_cast_df.reset_index([0, 'show_id'])
tv_cast_df.columns = ['show_id', 'cast']


# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

#for count plot
ax = sns.countplot(x="cast", data=tv_cast_df, palette='YlGn_d', order = tv_cast_df.cast.value_counts().iloc[:20].index)

#aesthetics of the plot
ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('Which artists have the most TV Shows on Netflix??', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Cast', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))


#setting count values on each bar
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+0.5), fontsize = 12.5)


# # A look at the artists in the countries with highest amount of TV Shows

# In[ ]:


tv_cast_df = pd.merge(tv_country_df, tv_cast_df, on='show_id', how = 'inner')
tv_cast_df = tv_cast_df.groupby(['cast','country']).size().reset_index(name='count')


# In[ ]:


country_list = ['United States', 'United Kingdom', 'Japan', 'South Korea', 'Canada', 'Taiwan']
tv_cast_df = tv_cast_df[tv_cast_df['country'].isin(country_list)]


# In[ ]:


tv_cast_df = tv_cast_df.groupby(['country']).apply(lambda x: x.sort_values(['count'],ascending = False)).reset_index(drop = True)
tv_cast_df = tv_cast_df.groupby(['country']).head(10)


# In[ ]:


f, axes = plt.subplots(2, 3, figsize=(25, 12), sharex=False)

ax1 = sns.barplot(x = 'count', y = 'cast', data = tv_cast_df[tv_cast_df['country'] == 'United States'], palette="YlGn_d", ax=axes[0, 0]).set_title("United States")

ax2 = sns.barplot(x = 'count', y = 'cast', data = tv_cast_df[tv_cast_df['country'] == 'United Kingdom'], palette="YlGn_d", ax=axes[0, 1]).set_title("United Kingdom")

ax3 = sns.barplot(x = 'count', y = 'cast', data = tv_cast_df[tv_cast_df['country'] == 'Japan'], palette="YlGn_d", ax=axes[0, 2]).set_title("Japan")

ax4 = sns.barplot(x = 'count', y = 'cast', data = tv_cast_df[tv_cast_df['country'] == 'South Korea'], palette="YlGn_d", ax=axes[1, 0]).set_title("South Korea")

ax5 = sns.barplot(x = 'count', y = 'cast', data = tv_cast_df[tv_cast_df['country'] == 'Canada'], palette="YlGn_d", ax=axes[1, 1]).set_title("Canada")

ax6 = sns.barplot(x = 'count', y = 'cast', data = tv_cast_df[tv_cast_df['country'] == 'Taiwan'], palette="YlGn_d", ax=axes[1, 2]).set_title("Taiwan")




plt.setp(axes, xticks=np.arange(0,20,2))
plt.tight_layout()


# # How the TV Shows are divided based on their audience ratings

# In[ ]:


plt.figure(figsize=(20,10))
sns.set(style="white")

ax = sns.countplot(x="rating", data=tv_df, palette='YlGn_d', order = tv_df.rating.value_counts().index)

ax.grid(False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90, fontsize = 13)
ax.set_yticklabels(ax.get_yticks(), size = 10)
sns.despine(bottom = True, left = True)
plt.title('How the TV Shows are divided based on their audience ratings', fontsize = 20.5, fontweight = 'bold')
plt.xlabel('Ratings', fontsize = 17.5)
plt.ylabel('')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.2, p.get_height()+5), fontsize = 12.5)

