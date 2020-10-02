#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Import the numpy and pandas packages

import numpy as np
import pandas as pd
import csv               #import the csv package


# ## Task 1: Reading and Inspection
# 
# -  ### Subtask 1.1: Import and read
# 
# Import and read the movie database. Store it in a variable called `movies`.

# In[ ]:


movies =pd.read_csv("../input/MovieAssignmentData.csv")     #importing the csv file
movies


# -  ### Subtask 1.2: Inspect the dataframe
# 
# Inspect the dataframe's columns, shapes, variable types etc.

# In[ ]:


movies.shape     #inspecting the number of rows and columns (rows, columns)


# In[ ]:


movies.info()    #inspecting the meta-data


# In[ ]:


movies.describe()      #statistical summary of the DataFrame


# ## Task 2: Cleaning the Data
# 
# -  ### Subtask 2.1: Inspect Null values
# 
# Find out the number of Null values in all the columns and rows. Also, find the percentage of Null values in each column. Round off the percentages upto two decimal places.

# In[ ]:


movies.isnull().sum()     #column-wise null count


# In[ ]:


movies.isnull().sum(axis=1)      #row-wise null count


# In[ ]:


round(100*(movies.isnull().sum()/len(movies.index)),2)       #column-wise null percentages


# -  ### Subtask 2.2: Drop unecessary columns
# 
# For this assignment, you will mostly be analyzing the movies with respect to the ratings, gross collection, popularity of movies, etc. So many of the columns in this dataframe are not required. So it is advised to drop the following columns.
# -  color
# -  director_facebook_likes
# -  actor_1_facebook_likes
# -  actor_2_facebook_likes
# -  actor_3_facebook_likes
# -  actor_2_name
# -  cast_total_facebook_likes
# -  actor_3_name
# -  duration
# -  facenumber_in_poster
# -  content_rating
# -  country
# -  movie_imdb_link
# -  aspect_ratio
# -  plot_keywords

# In[ ]:


movies= movies.drop('color', axis=1)       #code for dropping the column 'color'
movies


# In[ ]:


movies= movies.drop('director_facebook_likes', axis=1)        #code for dropping the column 'director_facebook_likes'
movies


# In[ ]:


movies= movies.drop('actor_1_facebook_likes', axis=1)         #code for dropping the column 'actor_1_facebook_likes'
movies


# In[ ]:


movies= movies.drop('actor_2_facebook_likes', axis=1)         #code for dropping the column 'actor_2_facebook_likes'
movies


# In[ ]:


movies= movies.drop('actor_3_facebook_likes', axis=1)         #code for dropping the column 'actor_3_facebook_likes'
movies


# In[ ]:


movies= movies.drop('actor_2_name', axis=1)                   #code for dropping the column 'actor_2_name'
movies


# In[ ]:


movies= movies.drop('cast_total_facebook_likes', axis=1)      #code for dropping the column 'cast_total_facebook_likes'
movies


# In[ ]:


movies= movies.drop('actor_3_name', axis=1)                   #code for dropping the column 'actor_3_name'
movies


# In[ ]:


movies= movies.drop('duration', axis=1)                       #code for dropping the column 'duration'
movies


# In[ ]:


movies= movies.drop('facenumber_in_poster', axis=1)           #code for dropping the column 'facenumber_in_poster'
movies


# In[ ]:


movies= movies.drop('content_rating', axis=1)                 #code for dropping the column 'content_rating'
movies


# In[ ]:


movies= movies.drop('country', axis=1)                        #code for dropping the column 'country'
movies


# In[ ]:


movies= movies.drop('movie_imdb_link', axis=1)                #code for dropping the column 'movie_imdb_link'
movies


# In[ ]:


movies= movies.drop('aspect_ratio', axis=1)                  #code for dropping the column 'aspect_ratio'
movies


# In[ ]:


movies= movies.drop('plot_keywords', axis=1)                 #code for dropping the column 'plot_keywords'
movies


# -  ### Subtask 2.3: Drop unecessary rows using columns with high Null percentages
# 
# Now, on inspection you might notice that some columns have large percentage (greater than 5%) of Null values. Drop all the rows which have Null values for such columns.

# In[ ]:


movies=movies[pd.notnull(movies['gross'])]      #code for dropping the rows which have null values for the column 'gross'


# In[ ]:


movies=movies[pd.notnull(movies['budget'])]     #code for dropping the rows which have null values for the column 'budget'


# In[ ]:


round(100*(movies.isnull().sum()/len(movies.index)),2)         #inspecting the column-wise null percentages


# -  ### Subtask 2.4: Drop unecessary rows
# 
# Some of the rows might have greater than five NaN values. Such rows aren't of much use for the analysis and hence, should be removed.

# In[ ]:


movies[movies.isnull().sum(axis=1)>5]         #code for dropping the rows having greater than five NaN values


# -  ### Subtask 2.5: Fill NaN values
# 
# You might notice that the `language` column has some NaN values. Here, on inspection, you will see that it is safe to replace all the missing values with `'English'`.

# In[ ]:


movies.loc[pd.isnull(movies['language']), ['language']] = 'English'    #code for filling the NaN values in the 'language' column as English


# -  ### Subtask 2.6: Check the number of retained rows
# 
# You might notice that two of the columns viz. `num_critic_for_reviews` and `actor_1_name` have small percentages of NaN values left. You can let these columns as it is for now. Check the number and percentage of the rows retained after completing all the tasks above.

# In[ ]:


movies.shape    #code for checking number of retained rows


# In[ ]:


#code for checking the percentage of the rows retained


# **Checkpoint 1:** You might have noticed that we still have around `77%` of the rows!

# ## Task 3: Data Analysis
# 
# -  ### Subtask 3.1: Change the unit of columns
# 
# Convert the unit of the `budget` and `gross` columns from `$` to `million $`.

# In[ ]:


movies['budget']=round(movies.budget/1000000,2)        #code for unit conversion from $ to million $ in the column 'budget'


# In[ ]:


movies['gross']=round(movies.gross/1000000,2)          #code for unit conversion from $ to million $ in the column 'gross'


# In[ ]:


movies.rename(columns={'gross':'gross_in_millions'}, inplace=True)      #code for changing the column name
movies.rename(columns={'budget':'budget_in_millions'}, inplace=True)


# -  ### Subtask 3.2: Find the movies with highest profit
# 
#     1. Create a new column called `profit` which contains the difference of the two columns: `gross` and `budget`.
#     2. Sort the dataframe using the `profit` column as reference.
#     3. Extract the top ten profiting movies in descending order and store them in a new dataframe - `top10`

# In[ ]:


movies['profit_in_millions'] = movies['gross_in_millions'] - movies['budget_in_millions']   #code for creating the profit column


# In[ ]:


movies.sort_values(by='profit_in_millions',ascending=False, inplace=True)   #code for sorting the dataframe with 'profit_in_millions' column as the reference


# In[ ]:


top10 = movies.nlargest(10, 'profit_in_millions')     #code to get the top 10 profiting movies
top10


# -  ### Subtask 3.3: Drop duplicate values
# 
# After you found out the top 10 profiting movies, you might have notice a duplicate value. So, it seems like the dataframe has duplicate values as well. Drop the duplicate values from the dataframe and repeat `Subtask 3.2`.

# In[ ]:


top10.drop_duplicates(subset=None, keep='first', inplace=True)        #code for dropping duplicate values


# In[ ]:


movies.drop_duplicates(subset=None, keep='first', inplace=True)        #code for repeating subtask 2 after dropping duplicate values
movies.sort_values(by='profit_in_millions',ascending=False, inplace=True)
top10 = movies.nlargest(10, 'profit_in_millions')
top10


# **Checkpoint 2:** You might spot two movies directed by `James Cameron` in the list.

# -  ### Subtask 3.4: Find IMDb Top 250
# 
#     1. Create a new dataframe `IMDb_Top_250` and store the top 250 movies with the highest IMDb Rating (corresponding to the column: `imdb_score`). Also make sure that for all of these movies, the `num_voted_users` is greater than 25,000.
# Also add a `Rank` column containing the values 1 to 250 indicating the ranks of the corresponding films.
#     2. Extract all the movies in the `IMDb_Top_250` dataframe which are not in the English language and store them in a new dataframe named `Top_Foreign_Lang_Film`.

# In[ ]:


voted_users=movies.loc[movies.num_voted_users>25000, :]      #code for making sure 'num_voted_users' is greater than 25,000
voted_users.sort_values(by='imdb_score',ascending=False, inplace=True)
IMDb_Top_250 = voted_users.nlargest(250, 'imdb_score')      #code for extracting the top 250 movies as per the IMDb score 


# In[ ]:


initial_value=1          #code for creating the 'Rank' column
IMDb_Top_250['Rank'] = range(initial_value, len(IMDb_Top_250) +initial_value)
IMDb_Top_250 = IMDb_Top_250.set_index('Rank')
IMDb_Top_250


# In[ ]:


Top_Foreign_Lang_Film = IMDb_Top_250.loc[IMDb_Top_250.language!='English', :]        #code to extract top foreign language films from 'IMDb_Top_250'
Top_Foreign_Lang_Film


# **Checkpoint 3:** Can you spot `Veer-Zaara` in the dataframe?

# - ### Subtask 3.5: Find the best directors
# 
#     1. Group the dataframe using the `director_name` column.
#     2. Find out the top 10 directors for whom the mean of `imdb_score` is the highest and store them in a new dataframe `top10director`. 

# In[ ]:


arrangebymean = movies.groupby('director_name', as_index=False)['imdb_score'].mean()
arrangebymean.sort_values(by='imdb_score',ascending=False, inplace=True)
top10directors = arrangebymean.nlargest(10, 'imdb_score')      #code for extracting the top 10 directors
top10directors


# **Checkpoint 4:** No surprises that `Damien Chazelle` (director of Whiplash and La La Land) is in this list.

# -  ### Subtask 3.6: Find popular genres
# 
# You might have noticed the `genres` column in the dataframe with all the genres of the movies seperated by a pipe (`|`). Out of all the movie genres, the first two are most significant for any film.
# 
# 1. Extract the first two genres from the `genres` column and store them in two new columns: `genre_1` and `genre_2`. Some of the movies might have only one genre. In such cases, extract the single genre into both the columns, i.e. for such movies the `genre_2` will be the same as `genre_1`.
# 2. Group the dataframe using `genre_1` as the primary column and `genre_2` as the secondary column.
# 3. Find out the 5 most popular combo of genres by finding the mean of the gross values using the `gross` column and store them in a new dataframe named `PopGenre`.

# In[ ]:


movies['genre_1'] = movies['genres'].str.split('|').str[0]      #code for extracting the first two genres of each movie
movies['genre_2'] = movies['genres'].str.split('|').str[1]
movies['genre_2'] = movies['genre_2'].fillna(movies['genre_1'])
movies


# In[ ]:


movies_by_segment = movies.groupby(['genre_1','genre_2'], as_index=False)['gross_in_millions'].mean()    #code for grouping the dataframe
movies_by_segment.sort_values(by='gross_in_millions',ascending=False, inplace=True)
movies_by_segment


# In[ ]:


PopGenre = movies_by_segment.nlargest(5, 'gross_in_millions')        #code for getting the 5 most popular combo of genres
PopGenre


# **Checkpoint 5:** Well, as it turns out. `Family + Sci-Fi` is the most popular combo of genres out there!

# -  ### Subtask 3.7: Find the critic-favorite and audience-favorite actors
# 
#     1. Create three new dataframes namely, `Meryl_Streep`, `Leo_Caprio`, and `Brad_Pitt` which contain the movies in which the actors: 'Meryl Streep', 'Leonardo DiCaprio', and 'Brad Pitt' are the lead actors. Use only the `actor_1_name` column for extraction. Also, make sure that you use the names 'Meryl Streep', 'Leonardo DiCaprio', and 'Brad Pitt' for the said extraction.
#     2. Append the rows of all these dataframes and store them in a new dataframe named `Combined`.
#     3. Group the combined dataframe using the `actor_1_name` column.
#     4. Find the mean of the `num_critic_for_reviews` and `num_user_for_review` and identify the actors which have the highest mean.

# In[ ]:


Meryl_Streep = movies.loc[movies['actor_1_name'] == 'Meryl Streep', :]  #including all movies in which 'Meryl Streep' is the lead


# In[ ]:


Leo_Caprio = movies.loc[movies['actor_1_name'] == 'Leonardo DiCaprio', :]    #including all movies in which 'Leonardo DiCaprio' is the lead


# In[ ]:


Brad_Pitt = movies.loc[movies['actor_1_name'] == 'Brad Pitt', :]    #including all movies in which 'Brad Pitt' is the lead


# In[ ]:


temp=Meryl_Streep.append(Leo_Caprio, ignore_index=True)  #code for combining the three dataframes
Combined = temp.append(Brad_Pitt, ignore_index=True)
Combined


# In[ ]:


pregrouped = Combined.groupby(['actor_1_name'])     #code for grouping the combined dataframe
Grouped = pregrouped['num_critic_for_reviews','num_user_for_reviews'].mean()
Grouped


# In[ ]:


Final = Grouped.sort_values(by=['num_critic_for_reviews','num_user_for_reviews'], ascending=False) #code for finding the mean of critic reviews and audience reviews
Final


# **Checkpoint 6:** `Leonardo` has aced both the lists!

# In[ ]:




