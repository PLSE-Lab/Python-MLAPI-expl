#!/usr/bin/env python
# coding: utf-8

# # Investigating A Movie DataSet
# 
# ## Table Of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#analysis">Exploratory Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue. 
# 
# > Questions To Explore
# >>1. Which genres are most popular from year to year? <br>
# >>2. Whether the popularity of the movie is dependent on the movie's budget?<br>
# 
# ><br><br> Import the packaegs and read the "tmdb-movies.csv" into a dataframe

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

tmdb_df = pd.read_csv("../input/tmdbcsv/tmdb-movies.csv")
# Checking the data types and total number of data points before starting the analysis. 
tmdb_df.info()


# >Describe the Data

# In[ ]:


tmdb_df.describe()


# <a id='analysis'></a>
# ## Exploratory Analysis
# 
# > Data wrangling phase :As it is evident from the above table, we don't have revenue and budget information for all the movies. We have revenue information only for 4850 movies and budget information for only 5170 movies out of 10866 movie details avaliable. I am planning to use Panda function dropna to handle this null values when I am plotting a visualization<br><br>
# >As first part of the analysis we are going to separate the genre details along with years to analyze the data. The DataFrame provides information about the number of movies in each genre for every year.
# 

# In[ ]:


# Obtaining a list of genres
genre_details = list(map(str,(tmdb_df['genres'])))
genre = []
for i in genre_details:
	split_genre = list(map(str, i.split('|')))
	for j in split_genre:
		if j not in genre:
			genre.append(j)
# printing list of seperated genres.
print(genre)


# >There are 20 different genres and there are movies with no genre information and they are noted as nan

# > Next I obtain the range of years in the dataset

# In[ ]:


# minimum range value
min_year = tmdb_df['release_year'].min()
# maximum range value
max_year = tmdb_df['release_year'].max()
# print the range
print(min_year, max_year)


# >Next a dataframe is created to create a table structure where the rows correspond to each genre and the columns correspond to the range of years.

# In[ ]:


# Creating a dataframe with genre as index and years as columns
genre_df = pd.DataFrame(index = genre, columns = range(min_year, max_year + 1))
# to fill not assigned values to zero
genre_df = genre_df.fillna(value = 0)
print (genre_df.head())


# >Here I create a array of years to access the various movies in each year and split the genres and increment the value of genre in the correspoding position in the dataframe.

# In[ ]:


# list of years of each movie
year = np.array(tmdb_df['release_year'])
# index to access year value
z = 0
for i in genre_details:
    split_genre = list(map(str,i.split('|')))
    for j in split_genre:
        genre_df.loc[j, year[z]] = genre_df.loc[j, year[z]] + 1
    z+=1
genre_df


# > The sum of all the values in the DataFrame is greater than 10866 since multiple genres can be assigned to the same movie.

# >Next a pie chart is created which depicts the count of movies released in each genre over the years according to the given dataset.
# >The top 10 genres are displayed and the remaining genres are displayed under the label others.

# In[ ]:


# number of movies in each genre so far.
genre_count = {}
genre = []
for i in genre_details:
	split_genre = list(map(str,i.split('|')))
	for j in split_genre:
		if j in genre:
			genre_count[j] = genre_count[j] + 1
		else:
			genre.append(j)
			genre_count[j] = 1
gen_series = pd.Series(genre_count)
# pi chart
gen_series = gen_series.sort_values(ascending = False)
label = list(map(str,gen_series[0:10].keys()))
label.append('Others')
gen = gen_series[0:10]
sum = 0
for i in gen_series[10:]:
    sum += i
gen['sum'] = sum
fig1, ax1 = plt.subplots()
ax1.pie(gen,labels = label, autopct = '%1.1f%%', startangle = 90)
ax1.axis('equal')
plt.title("Percentage of movies in each genre between 1960 and 2015")
plt.show()


# >The above chart the number of movies in each genre over the years. We can interpret that most of movies released were in Drama followed by Comedy and so on. Therefore the number of movies in the genre drama is higher than the others compared. The plot depicts the top ten genres and others depicts the count of movies in the remaining genres.

# >Next a seperate table is created  as a dataframe in a similar manner as above to hold the popularity value of the movies according to their genres for every year.

# In[ ]:


# Creating a dataframe with genre as index and years as columns to get a count of popularity
popularity_df = pd.DataFrame(index = genre, columns = range(min_year, max_year + 1))
# to fill not assigned values to zero
popularity_df = popularity_df.fillna(value = 0.0)
print(popularity_df.head())


# >An array is created which holds all the popularity details and it is checked for null values to see whether there is any unavalability of popularity in any of the datarows in the given dataset.

# In[ ]:


# list of popularity levels of each movie
popularity = np.array(tmdb_df['popularity'])
# to check whether any popularity is zero.
print (len(popularity[popularity==0]))
# index to access year value
z = 0
for i in genre_details:
    split_genre = list(map(str,i.split('|')))
    for j in split_genre:
            popularity_df.loc[j, year[z]] = popularity_df.loc[j, year[z]] + popularity[z]
    z+=1
popularity_df


# >In the above output the 0 indicates all the datarows are provided with popularity values and there is no discreapancy in the data.

# >I use a function to standardize the data of values in the popularity dataframe so that the data does not contain any discrepancies and is depicted as the number of standard deviations it is away from the mean. Positive value indicates the movie is popular and the negative values indicates the movie is unpopular or less popular comparatively. The mean value acts as the line of seperation for identifying the popular movies.

# In[ ]:


# function to standardize the popularity of values in dataframe.
def standardize(p):
    p_std = (p - p.mean()) / p.std(ddof = 0)
    return p_std


# In[ ]:


popularity_std = standardize(popularity_df)
popularity_std


# >Next I create a series to hold the most popular genre for every year so i create a series with the range of years as the index.

# In[ ]:


# Creating a series to hold the popular genre for every year.
pop_genre = pd.Series(index = range(min_year, max_year + 1))
pop_genre.head()


# > Finally from the dataframe which contains the table of standardized popularity values i identify the maximum value for each column and it is identified as the most popular genre for that particular year and it is added to the Series.

# In[ ]:


# to identify the genre with maximum standardized popularity value
for i in range(min_year, max_year + 1):
    pop_genre[i] = popularity_std[i].argmax()
pop_genre


# >From the identify we can plainly see that the Drama is the most popular genre for most of the years. And in the graph below the changes in the popularity levels for the "Drama" genre is depicted over the years.

# In[ ]:


# to plot a histogram of genre 'Drama'.
plt.plot(popularity_std.loc['Drama'])
plt.xlabel('year')
plt.ylabel('popularity levels')
plt.title('Distribution of popularity for the genre Drama over the years')
plt.axis([1960, 2015, 0, 3.5])
plt.show()


# >From the above graph we can see the rise and fall of the genre Drama over the years. There are steep rises and falls in the popularity levels of the genre. The mean of the distribution above lies between 1980 t0 1990 since there steep rises and falls on both the ends so the mean almost has to be in the middle. From the graph we can also infer that the standarnd deviation is almost 0.5.

# ><b>The next question is posed is whether popularity of a movie is dependent upon the budget allocated for that movie</b>

# >Next I standardize the popularity data available in the array and load it in a new Series with the id of the dataset as the datas index.

# In[ ]:


# Standardizing popularity data
std_pop = pd.Series(((popularity - popularity.mean()) / popularity.std(ddof = 0)), index = tmdb_df['id'])
std_pop.head()


# >Next i load the information about the budget data into the new series.

# In[ ]:


# Obtaining budget data for positive values in standardized popularity.
budget = pd.Series(np.array(tmdb_df['budget']), index = tmdb_df['id'])
budget.head()


# > In the above budget array there are budget values which are zero. A movie cannot have been created with zero budget. Hence removing the data which is incomplete. I remove the data from the standardized popularity data and the budget data available by using a seprate boolean array which denotes if the budget is zero or not.

# In[ ]:


# to remove incomplete data from the dataset.
boolean = budget != 0
std_pop = std_pop[boolean]
budget = budget[boolean] 
budget.head()


# In[ ]:


print (budget.head(), std_pop.head())


# In[ ]:


print (len(std_pop), len(budget))


# > Only 5170 datarows out of the 10866 datarows in the dataset were complete with the budget details provided.

# >Next the budget data loaded in the series is being standardized.

# In[ ]:


# Standardizing the budget values using the function standardize defined above
std_budget = standardize(budget)
std_budget.head()


# >Next the co relation coefficient is calculated to determine details about the relationship between a movie's budget and its corresponding popularity

# In[ ]:


# co relation coefficient(Pearson's value)
(std_pop * std_budget).mean()


# >The Pearson's coefficient value is a positive value which denotes that there is a strong relatinship between the movie popularity and budget proportionaly. Therefore budget is one of the factors deciding the popularity of the movie.

# <a id='conclusions'></a>
# # Conclusion
# >Thus the most popular genre in most of the years is <b> Drama. </b>
# >The table gives a list of the most popular genres from 1960 to 2015.
# >The graphs shows the popularity distribution of the genre drama over the years in the given dataset and the distribution of genres in the movie set.<br>
# >The movie's budget plays a crucial role in the popularity of the film which is proved from the Pearson's coefficient which is calculated form the cleaned budget and popularity dataset.
# >But since there was lack of data regarding budget we cannot entirely rely on the data remaining after cleaning to determine whether the above stated relationship is true. From the cleaned dataset we have arrived at the relationship. Almost half the dataset was removed cause of unavailability of budget details. Therefore the above stated relationship is paritally true.
# >

# In[ ]:




