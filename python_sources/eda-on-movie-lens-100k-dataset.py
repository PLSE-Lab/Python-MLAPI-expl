#!/usr/bin/env python
# coding: utf-8

# Hi guys, Exploratory Data Analysis(EDA) is all about getting insight from datasets, and if this process is followed properly then we get some really good understanding of the features and distributions we have. In depth EDA certainly results in outstanding Feature Engineering which eventually leaves heavy impact on model performance.
# 
# This is a random exercise that I have performed. Here I've used 100K movie rating's older dataset provided by Movie Lens (https://grouplens.org/datasets/movielens/) but If you want, you can use 1M dataset as well.

# ### Importing Libraries and reading datasets

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud
from uszipcode import SearchEngine


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


dateparse = lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S')

rating_df = pd.read_csv('../input/u.data', sep='\t', 
                        names=['user_id', 'movie_id', 'rating', 'timestamp'], 
                        parse_dates=['timestamp'], 
                        date_parser=dateparse)

movie_df = pd.read_csv('../input/u.item', sep='|', encoding='latin-1',
                    names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 
                           'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 
                           'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'])

user_df = pd.read_csv('../input/u.user', sep='|', encoding='latin-1',
                     names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])


# In[ ]:


movie_df.sample(10)


# ### Exploring 'item_df' - 

# In[ ]:


movie_df.info()


# In[ ]:


movie_df.sample(6)


# In[ ]:


movie_df.describe()


# Observations:
#     1. video_release_date seems containing lots of NaN values.
#     2. release_date needs to be parsed to datetime.
#     3. imdb_url contains external links which is not usefull here anyways.

# In[ ]:


movie_df.release_date = pd.to_datetime(movie_df.release_date)


# ### Checkout NaN values percent in each column

# In[ ]:


def check_nan_percent(df):
    columns_null_count = df.isnull().sum()
    null_percent_df = pd.DataFrame({'null_percent': (columns_null_count/df.shape[0])*100, 'total_null': columns_null_count})
    return null_percent_df[null_percent_df.null_percent>0]


# In[ ]:


check_nan_percent(movie_df)


# Clearly video_release_date contains nothing and imdb_url is of no use, lets remove them. In release_date we have only null value, if we remove entire row, then it will cost us one missing movie_id which is not good. So lets fill it with mode value of same column, since this approach is not gonna harm much.

# In[ ]:


movie_df.drop(columns=['video_release_date', 'imdb_url'], inplace=True)
movie_df.release_date.fillna(value=movie_df.release_date.mode()[0], inplace=True)

check_nan_percent(movie_df)


# Cool

# ### Exploring rating_df

# In[ ]:


rating_df.info()


# In[ ]:


rating_df.sample(6)


# In[ ]:


rating_df.describe()


# Observations:
#     1. Everything seems all ok.

# #### Check missing values

# In[ ]:


check_nan_percent(rating_df)


# perfect, we don't have any missing value.

# ### Exploring user_df

# In[ ]:


user_df.info()


# In[ ]:


user_df.sample(6)


# Zip codes can be converted to city names, let's do that, and later we can use it for some additional analysis.

# In[ ]:


search = SearchEngine(simple_zipcode=True)
user_df['city'] = user_df.zip_code.apply(lambda zip: search.by_zipcode(zip).major_city)


# In[ ]:


user_df.describe()


# #### Check missing values

# In[ ]:


check_nan_percent(user_df)


# 37 zip codes can not be converted to city names. Lets check these records

# In[ ]:


user_df[user_df.city.isnull()]


# In all these cases zip codes seems invalid. Again removing these records completely will bring inconsistency, so better we fill these value with mode of the column.

# In[ ]:


user_df.city.fillna(value=user_df.city.mode()[0], inplace=True)


# ## Finding Insight - 

# I believe the best way to do that is by asking usefull questions from dataset, and not moving forward till you get the answers.

# __QA. From movie_df -__
#    1. [What are common genere of movies?](#QA1)
#    2. [How many movies got released each year and month?](#QA2)
#    3. [What is the prefferable week of month to release movies?](#QA3)
# 
# __QB. From movie_df -__
#    1. [Who watches more movies Men/Women?](#QB1)
#    2. [What age group watches more movies?](#QB2)
#    3. [Which kind of occupant watches more movies?](#QB3)
#     
# __QC. From movie_df + user_df -__
#    1. [What gender likes which kind of genere](#QC1)
#    2. [What age group watches which kind of movies?](#QC2)
#    3. [Are movie lover's increasing over time](#QC3)
#       1. [Overall](#QC3a)
#       2. [Gender wise](#QC3b)
#       3. [Age group wise](#QC3c)
# 
# __QD. From movie_df + user_df + rating_df__
#    1. [How much rating people give mostly.](#QD1)
#    2. [Most Rated Movies](#QD2)
#       1. [during all years](#QD2a)
#       3. [gender wise](#QD2c)
#       4. [age group wise](#QD2d)
#    3. [Most Loved Movies](#QD3)
#       1. [during all years](#QD3a)
#       3. [gender wise](#QD3c)
#       4. [age group wise](#QD3d)
#    4. [Worst movie as per user rating.](QD4)

# If you notice at various places, we have used word age group, which is not already there, so lets go ahead and create one additional categorical feature in user_df called __age_group__. Minimum age is 7 and maximum age is 73 so we should have following divisions:
# 
# * 5-12   -  Gradeschooler
# * 13-19  -  Teenager
# * 20-35  -  Young
# * 35-55  -  Midlife
# * above 55    -  Old

# In[ ]:


user_df['age_group'] = user_df.age.apply(lambda age: 'Gradeschooler' if 5<=age<=12 else ('Teenager' if 13<=age<=19 else ('Young' if 20<=age<=35 else ('Midlife' if 35<=age<=55 else 'Old'))))
user_df.sample(5)


# Lets also create joined DataFrames, they will be helpful later.

# In[ ]:


rating_user_df = rating_df.join(other=user_df, how='inner', on='user_id', lsuffix='_R')
rating_user_movie_df = rating_user_df.join(other=movie_df, how='inner', on='movie_id', rsuffix='_M')
rating_movie_df = rating_df.join(other=movie_df, how='inner', on='movie_id', rsuffix='_M')


# ### <a id='QA1'>What are common genere of movies?</a>

# In[ ]:


generes = ['unknown', 'action',
       'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary',
       'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery',
       'romance', 'sci_fi', 'thriller', 'war', 'western']

plt.figure(figsize=(12,7))
genere_counts = movie_df.loc[:,generes].sum().sort_values(ascending=False)
sns.barplot(x=genere_counts.index, y=genere_counts.values)
plt.xticks(rotation=60);


# drama and comedy are the most common genere type. We must also note that it can not be a clear indication of people's preference, because One movie can have more than one genere and drama is the most commor genere type.

# ### <a id="QA2">How many movies got released each year and month?</a>

# Yearly release

# In[ ]:


plt.figure(figsize=(12,7))
yearly_release_counts = movie_df.groupby(movie_df.release_date.dt.year).size().sort_values(ascending=False)
sns.lineplot(yearly_release_counts.index, yearly_release_counts.values);
plt.xlabel('Release Year');


# #release significantly increased after 80s, however it doesn't exactly depicts the actual number of release, but the number of rated release, actual number of release must be greate than what we have. Anyway, one thing is clear that people mostly watched movies released in 90s. There is a sudden fall in 1998's record, that might be because of incomplete record.

# Since there is not much record available for movies before 90s, so I will consider release count for months in 90s only.`

# In[ ]:


plt.figure(figsize=(12,7))
monthly_release_counts = movie_df[movie_df.release_date.dt.year > 1990].groupby(movie_df.release_date.dt.month).size()
sns.barplot(['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec'], monthly_release_counts.values);
plt.xlabel('Release Month');


# this is weird, rated movie rlease count in Jan month is too high as cmpare to other months. Dataset creators must have considered Jan 1st as the default release month and date.

# ### <a id="QA3">What is the prefferable week of month to release movies?</a>

# In[ ]:


plt.figure(figsize=(12,7))
weekday_release_counts = movie_df[movie_df.release_date.dt.year > 1990].groupby(movie_df.release_date.dt.dayofweek).size()
sns.barplot(['mon', 'tue', 'wed', 'thus', 'fri', 'sat', 'sun'], weekday_release_counts.values);
plt.xlabel('Release Day of Week');


# This trend seems fine. Mostly rated movies got released on Friday, and then on weekends.

# ### <a id='QB1'>Who watches more movies Men/Women?</a>

# In[ ]:


plt.figure(figsize=(9,6))
sns.barplot(user_df.groupby('gender').size().index, user_df.groupby('gender').size().values)
plt.title('Male/Female movie rating ratio');


# Male, of course ;)
# or may be mostly Male takes interest in rating movies.

# ### <a id='QB2'>What age group watches more movies?</a>

# In[ ]:


plt.figure(figsize=(9,6))
sns.barplot(user_df.groupby('age_group').size().index, user_df.groupby('age_group').size().values)
plt.title('movie watchers age_group wise');


# Obviously Gradeschoolers and Teenagers don't have that much of time and old people don't have much interestest. Youngesters are the ones who watches movies mostly and prefers rating them. So overall people from age 20 - 55 watches more movies.

# ### <a id='QB3'>Which kind of occupant watches more movies?</a>

# In[ ]:


plt.figure(figsize=(12,7))
movie_watcher_occupants = user_df.groupby('occupation').size().sort_values(ascending=False)
sns.barplot(movie_watcher_occupants.index, movie_watcher_occupants.values)
plt.title('movie watchers age_group wise')
plt.xticks(rotation=50);


# It appears that Students watches more movies, may be the ones who are above 20 means Young students. Irony is that people who are in entertainment don't watch or may rate movies. Lets explore it bit more.

# In[ ]:


pd.DataFrame(user_df.groupby(['occupation', 'age_group']).size().sort_values(ascending=False))


# From above dataframe it is clear that Young students are more interested in movies. After that Midelife people in any profession seems interested too.

# ### <a id='QC1'>What gender likes which kind of genere</a>

# In[ ]:


temp_df = rating_user_movie_df.groupby('gender').sum().loc[:,generes]
temp_df = temp_df.transpose()
temp_df


# Stacked Bar Chart-

# In[ ]:


plt.figure(figsize=(12, 6))

temp_df.M.sort_values(ascending=False).plot(kind='bar', color='teal', label="Male")
temp_df.F.sort_values(ascending=False).plot(kind='bar', color='black', label="Fe-Male")
plt.legend()
plt.xticks(rotation=60)
plt.show()


# Multi Car Chart - 

# In[ ]:


plt.figure(figsize=(12, 6))
m_temp_df = temp_df.M.sort_values(ascending=False)
f_temp_df = temp_df.F.sort_values(ascending=False)

plt.bar(x=m_temp_df.index, height=m_temp_df.values, label="Male", align="edge", width=0.3, color='teal')
plt.bar(x=f_temp_df.index, height=f_temp_df.values, label="Female", width=0.3, color='black')
plt.legend()
plt.xticks(rotation=60)
plt.show()


# Looks cool.. isn't it.

# ### <a id='QC2'>What age group watches which kind of movies?</a>

# ### <a id='QC3'>Are movie lover's increasing over time</a>

# ####      <a id='QC3a'>Overall</a>

# In[ ]:


rating_df.groupby(rating_df.timestamp.dt.year).size()


# We have rating record for only two years. Which is not enought for this observation.

# ####      <a id='QC3b'>Gender wise</a>

# In[ ]:


rating_user_df.groupby([rating_user_df.timestamp.dt.year, 'gender']).size()


# We don't have enough record in ratings dataset :/

# ####      <a id='QC3c'>Age group wise</a>

# In[ ]:


rating_user_df.groupby([rating_user_df.timestamp.dt.year, 'age_group']).size()


# We don't have enough record in ratings dataset :/

# ### <a id='QD1'>How much rating people give mostly.</a>

# In[ ]:


temp_df = rating_user_df.groupby(['gender', 'rating']).size()
plt.figure(figsize=(10, 5))
m_temp_df = temp_df.M.sort_values(ascending=False)
f_temp_df = temp_df.F.sort_values(ascending=False)

plt.bar(x=m_temp_df.index, height=m_temp_df.values, label="Male", align="edge", width=0.3, color='teal')
plt.bar(x=f_temp_df.index, height=f_temp_df.values, label="Female", width=0.3, color='black')
plt.title('Ratings given by Male/Female Viewers')
plt.legend()
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()


# Viewers gave mostly 4 start rating then 3 and then 5

# ### <a id='QD2'>Most Rated Movies</a>

# ####      <a id='QD2a'>during all years</a>

# In[ ]:


def draw_horizontal_movie_bar(movie_titles, ratings_count, title=''):
    plt.figure(figsize=(12, 7))
    sns.barplot(y=movie_titles, x=ratings_count, orient='h')
    plt.title(title)
    plt.ylabel('Movies')
    plt.xlabel('Count')
    plt.show()


# In[ ]:


top_ten_rated_movies = rating_movie_df.groupby('movie_id').size().sort_values(ascending=False)[:10]
top_ten_movie_titles = movie_df.iloc[top_ten_rated_movies.index].movie_title

draw_horizontal_movie_bar(top_ten_movie_titles.values, top_ten_rated_movies.values, 'Top 10 watched movies')


# ####      <a id='QD2c'>gender wise</a>

# In[ ]:


top_rated_movies_gender_wise = rating_user_movie_df.groupby(['gender','movie_id']).size()

for index_label in top_rated_movies_gender_wise.index.get_level_values(0).unique():

    top_10_userkind_rated_movies = top_rated_movies_gender_wise[index_label].sort_values(ascending=False)[:10]
    top_10_userkind_rated_movie_titles = movie_df.iloc[top_10_userkind_rated_movies.index].movie_title
    draw_horizontal_movie_bar(top_10_userkind_rated_movie_titles.values, top_10_userkind_rated_movies.values, f'Top 10 {index_label} watched movies')


# ####      <a id='QD2d'>age group wise</a>

# In[ ]:


top_rated_movies_age_group_wise = rating_user_movie_df.groupby(['age_group','movie_id']).size()

for index_label in top_rated_movies_age_group_wise.index.get_level_values(0).unique():
    top_10_userkind_rated_movies = top_rated_movies_age_group_wise[index_label].sort_values(ascending=False)[:10]
    top_10_userkind_rated_movie_titles = movie_df.iloc[top_10_userkind_rated_movies.index].movie_title
    draw_horizontal_movie_bar(top_10_userkind_rated_movie_titles.values, top_10_userkind_rated_movies.values, f'Top 10 {index_label} watched movies')


# Mostly movies are same for all user kinds, may be because of limited record we have in our ratings dataset

# ### <a id='QD3'>Most Loved Movies</a>

# ####      <a id='QD3a'>during all years</a>

# In[ ]:


top_ten_most_loved_movies = rating_movie_df.groupby('movie_id').sum().rating.sort_values(ascending=False)[:10]
top_ten_most_loved_movie_titles = movie_df.iloc[top_ten_most_loved_movies.index].movie_title

draw_horizontal_movie_bar(top_ten_most_loved_movie_titles.values, top_ten_most_loved_movies.values, 'Top 10 most loved movies')


# ####      <a id='QD3c'>gender wise</a>

# In[ ]:


most_loved_movies_gender_wise = rating_user_movie_df.groupby(['gender','movie_id']).sum().rating

for index_label in most_loved_movies_gender_wise.index.get_level_values(0).unique():

    top_10_userkind_loved_movies = most_loved_movies_gender_wise[index_label].sort_values(ascending=False)[:10]
    top_10_userkind_loved_movie_titles = movie_df.iloc[top_10_userkind_loved_movies.index].movie_title
    draw_horizontal_movie_bar(top_10_userkind_loved_movie_titles.values, top_10_userkind_loved_movies.values, f'Top 10 {index_label} loved movies')


# ####      <a id='QD3d'>age group wise</a>

# In[ ]:


most_loved_movies_age_group_wise = rating_user_movie_df.groupby(['age_group','movie_id']).sum().rating

for index_label in most_loved_movies_age_group_wise.index.get_level_values(0).unique():
    top_10_userkind_loved_movies = top_rated_movies_age_group_wise[index_label].sort_values(ascending=False)[:10]
    top_10_userkind_loved_movie_titles = movie_df.iloc[top_10_userkind_loved_movies.index].movie_title
    draw_horizontal_movie_bar(top_10_userkind_loved_movie_titles.values, top_10_userkind_loved_movies.values, f'Top 10 {index_label} loved movies')


# To be honest, I haven't watched most of these movies so not quite sure what movie contains what kind of storyline :>

# ### <a id='QD4'>Worst movies as per user rating.</a>

# In[ ]:


movies_ratings_sum = rating_user_movie_df.groupby('movie_id').sum().rating.sort_values()
movies_ratings_sum.index = movie_df.iloc[movies_ratings_sum.index].movie_title
# Will show movies with 0 < total_rating<= 10
lowest_rated_movies = movies_ratings_sum[movies_ratings_sum <= 10]


wordcloud = WordCloud(min_font_size=7, width=800, height=500, random_state=21, max_font_size=50, relative_scaling=0.5, colormap='Dark2')
# Substracted lowest_rated_movies from 11 so that we can have greater font size of least rated movies.
wordcloud.generate_from_frequencies(frequencies=(11-lowest_rated_movies).to_dict())
plt.figure(figsize=(16,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Cool! Hedd Wyn, Salut cousin, The Crude Oasis, The Man from Down Under and To Cross the Rubicon are few lowest rated movies.

# So this is it in this EDA, I have covered most of the questions that could be asked from this dataset except the ones related to user locations(zip codes), we can generate some insight from those as well, like 
# - Is there any pattern in location and movie choices?
# - What kind of occupats are residing in which place?
# - Whether place has any impact on users being soft rater and hard rater etc.
# 
# Will do that later ;)
