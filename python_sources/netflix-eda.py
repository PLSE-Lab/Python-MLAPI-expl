#!/usr/bin/env python
# coding: utf-8

# ## Netflix Movies and TV Shows - EDA
# 
# * Source: https://www.kaggle.com/shivamb/netflix-shows

# In[ ]:


# open libraries
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


#to get the kaggle path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# open df
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv', encoding = 'latin1', engine = 'python', delimiter = ',')


# In[ ]:


# print 4 first rows 
df.head()


# In[ ]:


# print info about 
df.info()


# #### show_id      
# 
# Unique ID for every Movie / Tv Show
# 
# 6234 non-null

# In[ ]:


# check if there's repetead values
## as expected, there aren't repetead values.

df.show_id.value_counts()


# #### type     
# 
# Identifier - A Movie or TV Show
# 
# 6234 non-null   object

# In[ ]:


df.type.value_counts()/len(df)*100


# In[ ]:


## a new dataset with only movies.
df_netflix = df[(df.type == 'Movie')]


# #### title        
# 
# Title of the Movie / Tv Show
# 
# 4265 non-null   object

# In[ ]:


df_netflix.title.describe()


# In[ ]:


df_netflix[(df_netflix.title == "The Silence")]
#there's one movie repetead, thus I'll drop one row.


# In[ ]:


df_netflix = df_netflix[(df_netflix.show_id != '80238292')]


# #### director   
# 
# Director of the Movie
# 
# 4137 non-null object

# In[ ]:


df_netflix.director.describe()


# In[ ]:


#print the movies directed by Raúl Campos and Jan Suter together.
df_netflix[(df_netflix.director == "Raúl Campos, Jan Suter")]


# In[ ]:


#create a df that split the director column
df_director = df_netflix.assign(var1 = df_netflix.director.str.split(',')).explode('var1').reset_index(drop = True)

#To remove white space at the beginning of string:
df_director['var1'] = df_director.var1.str.lstrip()


# In[ ]:


sns.countplot(y = df_director.var1, order=df_director.var1.value_counts().iloc[:15].index, palette = 'colorblind')
plt.title('Top 15 Directors')
plt.xlabel('')
plt.ylabel('')


# #### cast          
# 
# Actors involved in the movie / show
# 
# 3905 non-null object

# In[ ]:


df_netflix.cast.describe()


# In[ ]:


#create a df that split the cast column
df_cast = df_netflix.assign(var1=df_netflix.cast.str.split(',')).explode('var1').reset_index(drop=True)

#To remove white space at the beginning of string:
df_cast['var1'] = df_cast.var1.str.lstrip()


# In[ ]:


#plot the top 15 actors with the most movies.
sns.countplot(y = df_cast.var1, order=df_cast.var1.value_counts().iloc[:15].index, palette = 'colorblind')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.title('Top 15 Actors')


# In the plot above, we've plotted the number of movies that each actor worked. Most of them have Indian names.

# ####  country     
# 
# Country where the movie / show was produced
# 
# 4070 non-null object

# In[ ]:


df_netflix.country.describe()


# In[ ]:


#create a df that split the country column
df_country = df_netflix.assign(var1=df_netflix.country.str.split(',')).explode('var1').reset_index(drop=True)

#To remove white space at the beginning of string:
df_country['var1'] = df_country.var1.str.lstrip()


# In[ ]:


df_country.var1.value_counts()/len(df_country)*100


# In[ ]:


#plot the top 15 countries with the most movies produced.
sns.countplot(y = df_country.var1, order=df_country.var1.value_counts().iloc[:15].index, palette = 'colorblind')
plt.xlabel('Number of Movies')
plt.ylabel('Country')
plt.title('Top 15 Countries')


# The United States is the country with the most number of movies, followed by India and the United Kingdom.

# #### date_added    
# 
# Date it was added on Netflix
# 
# 4264 non-null datetime64

# In[ ]:


## convert to date
df_netflix['date_added']= df_netflix.date_added.astype('datetime64') 

## get only the year
df_netflix['date_added_year'] = df_netflix.date_added.dt.year

## get only the month
df_netflix['date_added_month'] = df_netflix.date_added.dt.month


# In[ ]:


## plot a density plot
sns.distplot(df_netflix.date_added_year)
plt.title('Added Year')
plt.xlabel("Year")


# Netflix has addded movies since 2008, from 2016 there's an expressive growth trend.

# In[ ]:


## plot a density plot
sns.distplot(df_netflix.date_added_month)
plt.title('Added Month')
plt.xlabel("Month")


# In[ ]:


# plot the number of movies added grouped by month 
#set order by the number of movies descending
sns.countplot(y=df_netflix.date_added_month, palette = 'colorblind', order = df_netflix.date_added_month.value_counts().index)


# Netflix adds more movies in December, January, October and November. Thus, in the end of the year.

# ### release_year
# 
# Actual Release year of the move / show
# 
# 4265 non-null

# In[ ]:


sns.distplot(df_netflix.release_year)
plt.title('Release Year')
plt.xlabel('Year')


# Most of the movies were released from 2017.

# #### rating        
# 
# TV Rating of the movie / show
# 
# 4257 non-null

# In[ ]:


df_netflix.rating.unique()


# In[ ]:


sns.countplot(y = df_netflix.rating, palette = 'colorblind', order = df_netflix.rating.value_counts().index)
plt.title('Rating')
plt.xlabel('')
plt.ylabel('')


# In[ ]:


df_netflix['duration'] = df_netflix.duration.str.rstrip('min').astype(float)

#df_disney.runtime.str.rstrip('min')


# In[ ]:


# duration density plot
sns.distplot(df_netflix.duration)
plt.title('Duration Density Plot')
plt.xlabel('Duration')


# #### listed_in         
# 
# Genres
# 
# 4265 non-null   object   

# In[ ]:


df_netflix.listed_in.describe()


# In[ ]:


#create a df that split the listed column
df_list = df_netflix.assign(var1 = df_netflix.listed_in.str.split(',')).explode('var1').reset_index(drop = True)

#To remove white space at the beginning of string:
df_list['var1'] = df_list.var1.str.lstrip()


# In[ ]:


df_list.var1.value_counts()/len(df_list)*100


# 21.16% of the movies are International Movies, 17.82% are Dramas and 12.22% are Comedies.

# In[ ]:


sns.countplot(y = df_list.var1, palette = 'colorblind', order = df_list.var1.value_counts().index)
plt.title('Movies Genres')
plt.ylabel('')


# #### description       
#  
#  The summary description
#  
#  4265 non-null   object      

# In[ ]:


# plot the most frequent words in the movies

# combine multiple rows into one object
text = df_netflix['description'].str.cat(sep='/ ')

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["one", "two", "three", "four", "five"])

# Create and generate a word cloud image:
wc= WordCloud(stopwords = stopwords, background_color="white").generate(text)

# Display the generated image:
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

#wc.to_file("netflix.png")
#adapted code from: https://www.datacamp.com/community/tutorials/wordcloud-python

