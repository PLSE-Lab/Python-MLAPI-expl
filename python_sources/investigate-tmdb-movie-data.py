#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset as a part of Nano Degree Program
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#qus">Questions</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#ref">References</a></li>
# </ul>

# <a id='intro'></a>
# 
# ## Introduction
# 
# >Investigation finds secret information by digging data. The TMDb movie data is used that contians information about 10000 movies. A logical method of investigation fetches out relevant information. This project shows the practical application of methods we should follow. So, it start with questions and end with the conclusions. Every part uses the relevant technologies and output accordingly. The outputs are belong tentative solution. However, I tzried to find the best one to show the meaningful insight of data.  
# 
# <a id='qus'></a>
# ### Questions
# 
# <ul>
# <li>What genre of movies are got popularity year to year?</li>
# <li>What kind of properties made movies earn highest revenues?</li>
# <li>Budget vs Revenue (Moving Average)</li>
# <li>Top 10 actors' gross revenue </li>
# </ul>

# In[ ]:


# Necessay Libraries are included
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[ ]:


# load TMdb Movie data in data frame
df_tmdb = pd.read_csv("/kaggle/input/tmdb-movies.csv")
df_tmdb.head(5)


# In[ ]:


# Find shape(row, col) of the data set
df_tmdb.shape


# In[ ]:


# Overall information of the data type and presence and missing of value of column 
df_tmdb.info()


# In[ ]:


df_tmdb.isnull().sum()


# In[ ]:


# Summary of Statistical presence of data over the columns that have numerial data type. 
df_tmdb.describe()


# ### Data Cleaning (Replace this with more specific notes!)
# > Cleaning means formating data what ever its content and content type. Formating null, special character.

# In[ ]:


# The output(in above) of functions shows that there are few columns of missing rows or values. Now fill null with
# emplty string 
df_tmdb.genres.fillna('', inplace=True)
df_tmdb.cast.fillna('', inplace=True)


# In[ ]:


df_tmdb.isnull().sum()


# In[ ]:


# Split the given sting by pipline character 
def explode_string(data):
    return data.split('|')


# In[ ]:


df_tmdb.genres = df_tmdb.genres.apply(explode_string)
df_tmdb.cast = df_tmdb.cast.apply(explode_string)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### 1. What genre of movies are got popularity year to year?
# The popularity of movies can be found by the popularity or vote average column. The statistical summary shows that 75% of movies have popularity bellow 0.713817 and vote average bellow 6.6. So, I have thought only to take the movies that popularity has > 0.70. There are few findings 
# <ul>
#     <li> Adventure, Drama, Action and Comedy genres are always hit</li>
#     <li> Since 1985 thriller movies has been getting audience</li>
#     <li> Romance movies were more popular in the decade 60</li>
#     <li> War movies are popular in the decade 60 and 70</li>    
# </ul>
# 

# In[ ]:


popu= 0.7
def movies_based_on_popularity(movies):
    """
    filter function to threshold off popularity column 
    """
    return movies[movies['popularity'] > popu]


# In[ ]:


df_popular = df_tmdb.groupby('release_year').apply(movies_based_on_popularity)
df_popular.describe()


# In[ ]:


def movies_genres_year(movies):
    """
        This function takes a Data and returns a dictionary having
        release_year and genre(array) as key and count of genre as value.
    """
    movies_in_year_by_genre = {} # declare of dictionary
    for (release_year,pos), genres in movies.items():
        for genre in genres:
            if release_year in movies_in_year_by_genre:  #check year is exist
                if genre in movies_in_year_by_genre[release_year]: # check  genre is exist
                    movies_in_year_by_genre[release_year][genre] += 1 # increment genre if it is exist
                else:
                    movies_in_year_by_genre[release_year][genre] = 1  # new genre entry 
            else:
                movies_in_year_by_genre[release_year] = {} # declare dictionary within dictionary
                movies_in_year_by_genre[release_year][genre] = 1 # new genre and year entry
                
    return movies_in_year_by_genre


# In[ ]:


popular_movies = movies_genres_year(df_popular.genres)
#popular_movies


# In[ ]:


# draw graph of every five years
for year,genre_dic in popular_movies.items():
    if year%5 == 0:
        pd.DataFrame(popular_movies[year], index=[year]).plot(kind='bar', title="Popular Movie by Genre", figsize=(20, 6))


# ### 2. What kind of properties made movies earn highest revenues?
# >The statistical summary shows that mean of adjustable revenue is 1.6 billion. I am considering the highest revenue earned movies are those have adjustable revenue >= 1600000000. The output of data frame 'higest_earned_movies' shows that the highest earned revenue movies have few significant properties those are given bellow
# > 1. The value of the 'popularity' column is greater than the Q3 quartile. This is the signature of earning a huge amount.
# > 2. The vote is one of the most countable action to understand a product's quality. And quality begs to earn in the normal sense.
# > Hence, both columns 'vote_count' and 'vote_averate' contain value higher than Q3 quartile
# > 3. The movie genre plays an important role to make the big shot. 5 out of 7 movies are in the Adventure genre. 3s are in Action.

# In[ ]:


higest_earned_movies = df_tmdb[df_tmdb['revenue_adj']>=1600000000].sort_values(by='revenue_adj',ascending=False)
higest_earned_movies.shape


# ### 2.1 Popularity 

# In[ ]:


pd.DataFrame(list(higest_earned_movies.popularity), index=list(higest_earned_movies.original_title)).plot(kind='bar',
title='Popularity of Highest earned movies; Q3 = 0.713817',legend= False,figsize=(20, 10))
plt.show()


# ### 2.2 Vote Count and Vote average

# In[ ]:


pd.DataFrame(list(higest_earned_movies.vote_count), index=list(higest_earned_movies.original_title)).plot(kind='bar',
title='Vote Count of Highest earned movies; Q3 = 145',legend= False,figsize=(20, 10))
plt.show()


# In[ ]:


pd.DataFrame(list(higest_earned_movies.vote_average), index=list(higest_earned_movies.original_title)).plot(kind='bar',
title='Vote Average of Highest earned movies; Q3=6.6 ',legend= False,figsize=(20, 10))
plt.show()


# ### 2.3 Movie Genres

# In[ ]:


def movies_highest_genre_count(data):
    """
        This function takes a Data and returns a dictionary having
        genre as key and count of genre as value.
    """
    movies_highest_by_genre = {} # declare of dictionary
    for genres in data:
        for genre in genres:
            if genre in movies_highest_by_genre:  #check genre is exist
                movies_highest_by_genre[genre] += 1 # increment genre if it is exist
            else:
                movies_highest_by_genre[genre] = 1  # new genre entry 
           
                
    return movies_highest_by_genre


# In[ ]:


highest_revenue_genres = movies_highest_genre_count(higest_earned_movies.genres)
#highest_revenue_genres


# In[ ]:


pd.DataFrame(highest_revenue_genres, index=['Genres']).plot(kind='bar',
title='Frequency of Genres of Highest earned movies',figsize=(20, 10))
plt.show()


# ### 3. Budget vs Revenue (Moving Average)
# > Simply calculate the moving averate of investment and return form 1960 to 2015. 
# And it shows that on an average movies are able to take profit.

# In[ ]:


# create dataframe contains only revenue adj >0
df_revenue = df_tmdb[df_tmdb.revenue_adj > 0].sort_values(by='release_year',ascending=True)
index_list = list(range(0, df_revenue.shape[0]))
df_revenue.index = index_list # rearrange the index 
#df_revenue.head(50)


# In[ ]:


ma_df_revenue=[]
ma_df_revenue_year=[]
t_revenue = 0

for i in range(df_revenue.shape[0]-31):
    for j in range(31):
        t_revenue += df_revenue['revenue_adj'][i+j] 
    
    ma_df_revenue.append(round(t_revenue/30,2))
    ma_df_revenue_year.append(df_revenue['release_year'][i+30])


# In[ ]:


# create dataframe contains only budget adj >0
df_budget = df_tmdb[df_tmdb.budget_adj > 0].sort_values(by='release_year',ascending=True)
index_list = list(range(0, df_budget.shape[0])) # same size to revenue
df_budget.index = index_list # rearrange the index 


# In[ ]:


ma_df_budget =[]
t_budget  = 0

for i in range(df_revenue.shape[0]-31):
    for j in range(30):
        t_budget += df_budget['revenue_adj'][i+j] 
    
    ma_df_budget.append(round(t_budget/30,2))


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(ma_df_revenue_year,ma_df_budget,color='orange')
plt.plot(ma_df_revenue_year,ma_df_revenue,color='green')
plt.xlabel('Years')
plt.ylabel('Amount in $')
plt.title('Budget vs Revenue 1960 to 2015 ')
plt.legend(['Budget','Revenue'])
plt.show()


# ### 4. Top 10 actors' gross revenue 
# > The question ask to know who are in the leaderboard( top 10). The calculation is made based on the revenue the movie earned.
# Result will be shown in bar and pie graph. Harrison Ford and Tom Cruise first and second respectively.

# In[ ]:


def calculate_actors_gross_revenue(movies):   
    """
        This function takes data, forming dictionary having actors as key and sum of earned revenue as value. 
        return this formed dictionary to the caller.
    """
    actors_gross_revenue ={}
    for id,row in movies.iterrows():
        #print(row)
        for key in row.cast:
            if key in actors_gross_revenue:
                #print(actors_gross_revenue[key])
                actors_gross_revenue[key] +=  row.revenue_adj 
            else:
                actors_gross_revenue[key] = row.revenue_adj 
    
    return actors_gross_revenue;


# In[ ]:


top_actors_gross_revenue = sorted(calculate_actors_gross_revenue(df_revenue).items(), key=lambda item: item[1], reverse=True)[:10]


# In[ ]:


def list_to_dict(data, label):
    """
        This function uses data as value and label as key for froming a dictionary. At the end return this dictionary.
    """
    top_actors = {label: []}
    index = []
    
    for item in data:
        top_actors[label].append(item[1])
        index.append(item[0])
        
    return top_actors, index

top10_actors_revenue, top10_actors_name = list_to_dict(top_actors_gross_revenue, 'actors')


# In[ ]:


pd.DataFrame(top10_actors_revenue, index=top10_actors_name).plot(kind='bar',title='Top 10 actors gross revenue all time',figsize=(20, 15))
plt.show()


# In[ ]:


pd.DataFrame(top10_actors_revenue, index=top10_actors_name).plot(kind='pie',y='actors',title='Top 10 actors gross revenue all time',figsize=(20, 15))
plt.show()


# <a id='conclusions'></a>
# ## Conclusions
# 
# > Overall, I got a few significant and meaningful findings and correlation of data. It seems that the movie genre has impacts on most of the findings and observations. It is related to the highest revenue, highest vote, vote count, and popularity. The next one is that the budget and revenue proportion to each other.

# <a id='ref'></a>
# ## References
# 1. <a href='https://pandas.pydata.org/'>Pandas</a>
# 2. <a href='https://matplotlib.org/'>Matplotlib</a>
# 3. <a href='https://stackoverflow.com/'>Stack Overflow</a>
