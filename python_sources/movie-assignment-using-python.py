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


# ## Task 1: Reading and Inspection
# 
# -  ### Subtask 1.1: Import and read
# 
# Import and read the movie database. Store it in a variable called `movies`.

# In[ ]:


movies =pd.read_csv('../input/movie_dataset.csv') # Write your code for importing the csv file here
movies


# -  ### Subtask 1.2: Inspect the dataframe
# 
# Inspect the dataframe's columns, shapes, variable types etc.

# In[ ]:


# Write your code for inspection here
print(movies.shape)
print(type(movies.info()))


# ## Task 2: Cleaning the Data
# 
# -  ### Subtask 2.1: Inspect Null values
# 
# Find out the number of Null values in all the columns and rows. Also, find the percentage of Null values in each column. Round off the percentages upto two decimal places.

# In[ ]:


# Write your code for column-wise null count here
movies.isnull().sum(axis=0)


# In[ ]:


# Write your code for row-wise null count here
movies.isnull().sum(axis=1)


# In[ ]:


# Write your code for column-wise null percentages here
round(100*(movies.isnull().sum()/len(movies.index)),2)


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


# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 
movies=movies.drop('color', axis=1)
movies=movies.drop('director_facebook_likes', axis=1)
movies=movies.drop('actor_1_facebook_likes', axis=1)
movies=movies.drop('actor_2_facebook_likes', axis=1)
movies=movies.drop('actor_3_facebook_likes', axis=1)
movies=movies.drop('actor_2_name', axis=1)
movies=movies.drop('cast_total_facebook_likes', axis=1)
movies=movies.drop('actor_3_name', axis=1)
movies=movies.drop('duration', axis=1)
movies=movies.drop('facenumber_in_poster', axis=1)
movies=movies.drop('content_rating', axis=1)
movies=movies.drop('country', axis=1)
movies=movies.drop('movie_imdb_link', axis=1)
movies=movies.drop('aspect_ratio', axis=1)
movies=movies.drop('plot_keywords', axis=1)
round(100*(movies.isnull().sum()/len(movies.index)),2)


# -  ### Subtask 2.3: Drop unecessary rows using columns with high Null percentages
# 
# Now, on inspection you might notice that some columns have large percentage (greater than 5%) of Null values. Drop all the rows which have Null values for such columns.

# In[ ]:


# Write your code for dropping the rows here
movies = movies[~np.isnan(movies['gross'])]
movies = movies[~np.isnan(movies['budget'])]
round(100*(movies.isnull().sum()/len(movies.index)),2)


# -  ### Subtask 2.4: Drop unecessary rows
# 
# Some of the rows might have greater than five NaN values. Such rows aren't of much use for the analysis and hence, should be removed.

# In[ ]:


# Write your code for dropping the rows here
movies = movies[movies.isnull().sum(axis=1) <= 5]
round(100*(movies.isnull().sum()/len(movies.index)),2)


# -  ### Subtask 2.5: Fill NaN values
# 
# You might notice that the `language` column has some NaN values. Here, on inspection, you will see that it is safe to replace all the missing values with `'English'`.

# In[ ]:


# Write your code for filling the NaN values in the 'language' column here
movies.loc[pd.isnull(movies['language']), ['language']] = 'English'
round(100*(movies.isnull().sum()/len(movies.index)), 2)


# -  ### Subtask 2.6: Check the number of retained rows
# 
# You might notice that two of the columns viz. `num_critic_for_reviews` and `actor_1_name` have small percentages of NaN values left. You can let these columns as it is for now. Check the number and percentage of the rows retained after completing all the tasks above.

# In[ ]:


# Write your code for checking number of retained rows here
print(movies.index)
round(100*(len(movies.index)/5043),2)


# **Checkpoint 1:** You might have noticed that we still have around `77%` of the rows!

# ## Task 3: Data Analysis
# 
# -  ### Subtask 3.1: Change the unit of columns
# 
# Convert the unit of the `budget` and `gross` columns from `$` to `million $`.

# In[ ]:


# Write your code for unit conversion here
movies[['gross','budget']].apply(lambda x:x/1000000)


# -  ### Subtask 3.2: Find the movies with highest profit
# 
#     1. Create a new column called `profit` which contains the difference of the two columns: `gross` and `budget`.
#     2. Sort the dataframe using the `profit` column as reference.
#     3. Extract the top ten profiting movies in descending order and store them in a new dataframe - `top10`

# In[ ]:


# Write your code for creating the profit column here
movies['profit']=movies['gross'] - movies['budget']
movies


# In[ ]:


# Write your code for sorting the dataframe here
movies.sort_values(by='profit',ascending=True)


# In[ ]:


top10=movies.sort_values(by=['profit'],ascending=False).head(10) # Write your code to get the top 10 profiting movies here
top10


# -  ### Subtask 3.3: Drop duplicate values
# 
# After you found out the top 10 profiting movies, you might have notice a duplicate value. So, it seems like the dataframe has duplicate values as well. Drop the duplicate values from the dataframe and repeat `Subtask 3.2`.

# In[ ]:


# Write your code for dropping duplicate values here
movies.drop_duplicates()


# In[ ]:


# Write code for repeating subtask 2 here
top10=movies.sort_values(by=['profit'],ascending=False).head(10)
top10


# **Checkpoint 2:** You might spot two movies directed by `James Cameron` in the list.

# -  ### Subtask 3.4: Find IMDb Top 250
# 
#     1. Create a new dataframe `IMDb_Top_250` and store the top 250 movies with the highest IMDb Rating (corresponding to the column: `imdb_score`). Also make sure that for all of these movies, the `num_voted_users` is greater than 25,000.
# Also add a `Rank` column containing the values 1 to 250 indicating the ranks of the corresponding films.
#     2. Extract all the movies in the `IMDb_Top_250` dataframe which are not in the English language and store them in a new dataframe named `Top_Foreign_Lang_Film`.

# In[ ]:


# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'
IMDb_Top_250= movies.sort_values(by=['imdb_score'],ascending=False).head(250) # stored top 250 movies with highest imdb rating in new dataframe IMDb_top_250
IMDb_Top_250.loc[(IMDb_Top_250.num_voted_users>25),:]   # checking for all these movies,num_voted_users is greater than 25000
IMDb_Top_250['Rank']=range(1,251)      # Add a new column 'Rank' ranging from 1 to 251 indicate the ranks of the corresponding films  
IMDb_Top_250


# In[ ]:


Top_Foreign_Lang_Film =IMDb_Top_250.loc[(IMDb_Top_250.language != 'English'),:]       # Write your code to extract top foreign language films from 'IMDb_Top_250' here
Top_Foreign_Lang_Film 


# **Checkpoint 3:** Can you spot `Veer-Zaara` in the dataframe?

# - ### Subtask 3.5: Find the best directors
# 
#     1. Group the dataframe using the `director_name` column.
#     2. Find out the top 10 directors for whom the mean of `imdb_score` is the highest and store them in a new dataframe `top10director`. 

# In[ ]:


# Write your code for extracting the top 10 directors here
top10director=movies.groupby('director_name')['imdb_score'].mean().sort_values(ascending=False).head(10)
top10director


# **Checkpoint 4:** No surprises that `Damien Chazelle` (director of Whiplash and La La Land) is in this list.

# -  ### Subtask 3.6: Find popular genres
# 
# You might have noticed the `genres` column in the dataframe with all the genres of the movies seperated by a pipe (`|`). Out of all the movie genres, the first two are most significant for any film.
# 
# 1. Extract the first two genres from the `genres` column and store them in two new columns: `genre_1` and `genre_2`. Some of the movies might have only one genre. In such cases, extract the single genre into both the columns, i.e. for such movies the `genre_2` will be the same as `genre_1`.
# 2. Group the dataframe using `genre_1` as the primary column and `genre_2` as the secondary column.
# 3. Find out the 5 most popular combo of genres by finding the mean of the gross values using the `gross` column and store them in a new dataframe named `PopGenre`.

# In[ ]:


# Write your code for extracting the first two genres of each movie here
first=movies['genres'].apply(lambda x: pd.Series(x.split('|')))
movies['genre_1']=first[0]
movies['genre_2']=first[1]
movies.loc[pd.isnull(movies['genre_2']), ['genre_2']] = movies['genre_1']
print(movies.genre_1)
print(movies.genre_2)


# In[ ]:


movies_by_segment =movies.groupby(['genre_1','genre_2']) # Write your code for grouping the dataframe here
movies_by_segment


# In[ ]:


PopGenre =movies_by_segment['gross'].mean().sort_values(ascending=False).head(5) # Write your code for getting the 5 most popular combo of genres here
PopGenre


# **Checkpoint 5:** Well, as it turns out. `Family + Sci-Fi` is the most popular combo of genres out there!

# -  ### Subtask 3.7: Find the critic-favorite and audience-favorite actors
# 
#     1. Create three new dataframes namely, `Meryl_Streep`, `Leo_Caprio`, and `Brad_Pitt` which contain the movies in which the actors: 'Meryl Streep', 'Leonardo DiCaprio', and 'Brad Pitt' are the lead actors. Use only the `actor_1_name` column for extraction. Also, make sure that you use the names 'Meryl Streep', 'Leonardo DiCaprio', and 'Brad Pitt' for the said extraction.
#     2. Append the rows of all these dataframes and store them in a new dataframe named `Combined`.
#     3. Group the combined dataframe using the `actor_1_name` column.
#     4. Find the mean of the `num_critic_for_reviews` and `num_user_for_review` and identify the actors which have the highest mean.

# In[ ]:


# Write your code for creating three new dataframes here
Meryl_Streep=movies.loc[(movies.actor_1_name=='Meryl Streep'),:].head(3891)  # Include all movies in which Meryl_Streep is the lead
Meryl_Streep


# In[ ]:


Leo_Caprio=movies.loc[(movies.actor_1_name=='Leonardo DiCaprio'),:].head(3891) # Include all movies in which Leo_Caprio is the lead
Leo_Caprio


# In[ ]:


Brad_Pitt=movies.loc[(movies.actor_1_name=='Brad Pitt'),:].head(3891)  # Include all movies in which Brad_Pitt is the lead
Brad_Pitt


# In[ ]:


# Write your code for combining the three dataframes here
Combined=Meryl_Streep.append(Leo_Caprio).append(Brad_Pitt)
Combined


# In[ ]:


# Write your code for grouping the combined dataframe here
actor_name=Combined.groupby('actor_1_name')   # grouping the Combined dataframe using actor_1_name and store it in new dataframe actor_name
actor_name


# In[ ]:


# Write the code for finding the mean of critic reviews and audience reviews here
critic_reviews=actor_name['num_critic_for_reviews'].mean().sort_values(ascending=False).head(49)
print(critic_reviews)
audience_reviews=actor_name['num_user_for_reviews'].mean().sort_values(ascending=False).head(49)
print(audience_reviews)


# **Checkpoint 6:** `Leonardo` has aced both the lists!
