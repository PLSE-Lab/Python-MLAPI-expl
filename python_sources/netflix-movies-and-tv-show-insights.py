#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## **Introduction**
# This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third-party Netflix search engine.
# The dataset contains 6234 titles (TV Series and Movies) available on Netflix.
# There are 12 columns related to each title:
# 1. show_id - unique id of the title
# 2. type - type of the title [TV Show, Movie]
# 3. title - title name
# 4. director - director of the movie
# 5. cast - Leads of the title
# 6. country - Country of origin
# 7. date_added - date on which the title was added to Netflix
# 8. release_year - year of release.
# 9. rating - TV-rating
# 10. duration - Runtime of the title (in minutes or seasons)
# 11. listed_in - Genre
# 12. description - short summary of the title
# 
# Let's gather some insights from this data

# In[ ]:


# Loading the csv
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df.head()


# In[ ]:


# columns in the dataset
print(df.columns)
# shape of dataset
print(df.shape)


# ## Year of release
# Let's look at the distribution of titles based on the year they were released in. We can use 'release_year' to extract counts of titles released each year. We'll plot data of last 15 years.

# In[ ]:


plt.figure(figsize = (15, 5))
sns.countplot(df['release_year'][df.release_year>2004], palette = 'PuBuGn')
df.release_year.value_counts()[:15]


# We can also drill down to the type of content released each year. Let's have a look at how the content count of TV shows and movies changed over the years.

# In[ ]:


# Grouping the titles by their release year and then by the type of content(ie. TV Show or Movies)
counts = df.groupby(['release_year', 'type'])['show_id'].count().to_frame().reset_index()
#splitting the type column into two columns (TV Show and Movie) which will contain the count of TV Shows and Movies releasedrespectively 
counts = pd.pivot_table(counts, values='show_id', index=['release_year'], columns=['type']).fillna(0).reset_index()

plt.figure(figsize = (20,5))
ax = sns.lineplot(x='release_year', y='Movie', data=counts, label = 'Movie')
ax2 = sns.lineplot(x='release_year', y='TV Show', data=counts, label = 'TV Show').set_title('Content count from each year')
plt.legend()
counts


# Let's zoom in a bit and have a look at last 40 years' data.

# In[ ]:


plt.figure(figsize = (15,5))
ax = sns.lineplot(x='release_year', y='Movie', data=counts[40:])
ax2 = sns.lineplot(x='release_year', y='TV Show', data=counts[40:]).set_title('Content from last 40 years')


# The plots shows that Netflix has consistently increased the title count from each starting 2011. Netflix added most content from year 2018 with 1063 titles. Year 2019 was an exception as it saw a decrease in the number of titles added from that year. It could be the outcome of Network companies starting their own streaming services and not giving streaming rights to Netflix. Also, many of the titles may not yet be available for streaming (because they are in theates or other contractual reasons). 

# ## Content Ratings
# About 1/3rd of Netflix's content is rated for Mature Audience and about 900 titles are PG rated(TV-PG and PG). Netflix's target audience are young adults, so it makes perfect sense that half of their content is aiming towards that market. TV-14 content also finds more than 1600 titles aimed for younger audience. 

# In[ ]:


plt.figure(figsize = (15,5))
sns.countplot(df.rating, palette = 'plasma')


# ## Netflix content release patterns
# We can also look at the patterns of when and how many titles were added to Netflix each year.
# Let's first look at the total titles added to Netflix each year.

# In[ ]:


df['year_added'] = df['date_added'].fillna(df['release_year'])
df['year_added'] = df['year_added'].astype(str).apply(lambda x: x[-4:])

plt.figure(figsize = (15,5))
sns.countplot(df.year_added, palette = "ch:2.5,-.2,dark=.3")


# The pattern is as expected, with Netflix adding more and more content each year. Let's now split the plots to show the patterns of TV Shows and Movies separately.

# In[ ]:


plt.figure(figsize = (15,5))
sns.countplot(x='year_added', hue='type',data=df)


# In[ ]:


h = df['listed_in'].unique()
pd.set_option('display.max_rows', 500)
h[0].split(',')


# ## Popular Genres
# There are 72 genres in which the titles are categorized. We will split the 'listed_in' column and melt back into multiple rows for each title. The plot below shows the distribution of titles across genres.
# 

# In[ ]:


# splitting the 'listed_in' column.
genres = df['listed_in'].str.split(',', 4, expand=True)
# pd.set_option('display.max_rows', genres.shape[0] + 1)

# adding title column and reordering the columns
genres['title'] = df['title']
genres = genres[['title', 0, 1, 2]]

#melting the genres columns into a single column.
genres = pd.melt(genres, id_vars=['title'], value_vars=[0, 1, 2])
genres = genres.dropna()
genres = genres.drop('variable', axis = 1)

# plotting the countplot for genres.
plt.figure(figsize=(25,10))
plot = sns.countplot(x = 'value', data = genres)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
plt.xlabel('Genres')
plt.ylabel('Number of Titles')

