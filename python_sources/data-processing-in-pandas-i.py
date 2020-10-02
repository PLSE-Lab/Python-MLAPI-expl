#!/usr/bin/env python
# coding: utf-8

# # Data Processing in Pandas - Notes  
# 
# *This is first part of a two part series : Part 1, [Part 2](https://www.kaggle.com/ctxplorer/data-processing-in-pandas-ii)*
# 
# #### Content:
# 1. Creating a DataFrame
# 2. Loading and saving CSVs
# 3. Inspecting a DataFrame
# 4. Selecting columns
# 5. Selecting rows 
# 6. Selecting rows with logical conditions
# 7. Resetting indices 

# In[ ]:


import pandas as pd


# ## 1. Creating a DataFrame

# #### Add data using Dictonary

# In[ ]:


df1 = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Joe Schmo'],
    'address': ['123 Main St.', '456 Maple Ave.', '789 Broadway'],
    'age': [34, 28, 51]
})
print(df1)


# #####  The columns will appear in alphabetical order because dictionaries don't have any inherent order for columns

# #### Add data using List

# In[ ]:


df2 = pd.DataFrame([
    ['John Smith', '123 Main St.', 34],
    ['Jane Doe', '456 Maple Ave.', 28],
    ['Joe Schmo', '789 Broadway', 51]
    ],
    columns=['name', 'address', 'age'])
print(df2)


# ## 2. Loading and saving CSVs

# In[ ]:


# save data to a CSV
df1.to_csv('new-csv-file.csv')

# load CSV file into a DataFrame in Pandas
df3 = pd.read_csv('../input/sample-csv-file/sample.csv')

print(df3)


# ## 3. Inspecting a DataFrame

# In[ ]:


df4 = pd.read_csv('../input/imdb-data/IMDB-Movie-Data.csv')

# print first 3 rows of DataFrame (Default 5)
print(df4.head(3))

# print statistics for each columns
print(df4.info())


# ## 4. Selecting columns

# #### Using name of column
# ##### used only if name of columns follows all the rules of variable naming

# In[ ]:


# Select column 'Title'
imdb_title = df4.Title
print(imdb_title.head())


# #### Using key value

# In[ ]:


# Select column 'Runtime (Minutes)'
imdb_runtime_minutes = df4['Runtime (Minutes)']
print(imdb_runtime_minutes.head())


# #### Selecting multiple columns

# In[ ]:


imdb_data = df4[['Title', 'Runtime (Minutes)']]
print(imdb_data.head())


# ## 5. Selecting rows

# In[ ]:


# select fourth row
sing_movie = imdb_data.iloc[3]
print(sing_movie)


# #### Selecting multiple rows

# In[ ]:


# select last third row
last_three_movies = imdb_data.iloc[-3:]
print(last_three_movies)


# ## 6. Selecting rows with logical conditions

# In[ ]:


# select rows with runtime less than 75
short_movies = imdb_data[imdb_data['Runtime (Minutes)'] < 75]
print(short_movies)


# #### Selecting rows with multiple logical conditions
# ##### Use paranthesis when combining multiple logical condition

# In[ ]:


# select rows with runtime between 60 and 80
medium_length_movies = imdb_data[(imdb_data['Runtime (Minutes)'] > 60) &
                                 (imdb_data['Runtime (Minutes)'] < 80)]
print(medium_length_movies)


# #### Selecting rows with specific values

# In[ ]:


# select rows with title in the list
fav_movies = imdb_data[imdb_data.Title.isin([
    'Wolves at the Door', 'Guardians of the Galaxy'
])]
print(fav_movies)


# ## 7. Resetting indices  
# ##### When we select a subset of a DataFrame using logic, we end up with non-consecutive indices. We can fix this using the method **.reset_index()**. 

# In[ ]:


# reset indices without changing the source DF
fav_movies = fav_movies.reset_index(drop=True)
print(fav_movies)

# reset indices in the source DF
medium_length_movies.reset_index(drop=True, inplace=True)
print(medium_length_movies)


# ### That is all for now. Hope it helped you!
# #### Check out [Part 2](https://www.kaggle.com/ctxplorer/data-processing-in-pandas-ii) of the series.
