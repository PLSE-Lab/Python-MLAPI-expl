#!/usr/bin/env python
# coding: utf-8

# ## Sai Pranav Uppati
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# Below code imports NumPy and Pandas libraries, indicates the directory structure and lists the CSV files from the chosen dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id='intro'></a>
# # Introduction
# 
# For this project, I have chosen a Kaggle dataset consisting of metadata on ~5000 movies, which was sourced from The Movie Database (TMDb), a community built movie and TV database. Though the original option for this movie metadata dataset was the IMDb one, it was since removed from Kaggle due to a DMCA takedown request. Kaggle has decided to compensate for the loss of the IMDb dataset with this one from TMDb. Please read more about this dataset and the change from the original at https://www.kaggle.com/tmdb/tmdb-movie-metadata. I hope this replacement dataset suffices for demonstration of my data exploration and analysis capabilities.
# 
# Personally, I quite enjoy watching movies. One of the genres I really enjoy is science fiction. I'm a big fan of several of the major releases under the Marvel Cinematic Universe banner such as Iron Man, Captain America, Dr. Strange, The Avengers series etc. I have a general sense that this sub-genre of 'superhero' movies are high-budget and typically high-grossing. But I haven't had an opportunity to explore a quantitative dataset like this to better understand what makes movies commercial successes. I look forward to exploring this dataset, learning and presenting my findings.
# 
# Let's create the pandas dataframes from the CSV files and take a look at the available data...

# In[ ]:


# read in CSV data
movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

# preview movies df
movies.head()


# In[ ]:


# preview credits df
credits.head()


# Based on the previews of the two data files we have read into dataframes, the primary dataframe will be the `movies` for our exploratory data analysis. The `credits` dataframe provides the exhaustive list of cast and crew, which is not immediately relevant unless we plan to encode the presence/absence of certain cast/crew members into well organized dataset for training a predictive machine learning model. For the scope of this assigment, it's sufficient to focus on more immediately available quantitative variables.
# 
# The `movies` dataframe provides various parameters for each movie, including movie budget, runtime, list of associated movie genres, metrics on how well movie was received e.g. popularity and vote average/vote count, release date and revenue among other potentially useful metadata. Based on provided categories of data, we may ask the following questions that we can attempt to answer with the dataset.
# 
# ### Questions:
# 1. Are high-budget movies more or less profitable than low-budget movies?
# 2. Does release date affect how well a movie does, i.e. are more profitable movies typically released in a specific time of year?
# 3. Are longer movies more or less popular than shorter movies?
# 4. Are certain genres of movies more popular than others? Do certain genres generate more profit?
# 5. Based on my personal interest, are Marvel movies more or less popular and profitable than DC movies? Are Marvel movies more popular than the average movie? Are they more profitable than the average movie?

# In[ ]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import matplotlib.pyplot as plt

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# # Data Wrangling
# 
# ## General Properties
# 
# The data has already been loaded into dataframes in the section above, so we will start with high level data inspection for cleanliness and usability for exploratory data analysis.

# In[ ]:


# Perform operations to inspect data types and look for instances of missing or possibly errant data
movies.info()


# Below are a few checks on unique values in certain columns. Let's first check how many original language english movies there are, because we want to focus on these to simplify our analysis and any conclusions we may for the questions posed.

# In[ ]:


# english language movies check
movies.original_language.value_counts()


# Because 4505 out of the 4803 movies in the dataset are original english language movies, it's not a big loss to drop the remaining movies from our analysis. We will do this in the next section. 
# 
# Let's check now on whether two columns for movie titles are needed.

# In[ ]:


# Do we need to keep both original_title and title columns, let's see how many mismatches there are
(movies.original_title != movies.title).sum()


# Out of almost 5000 movies, only 261 mismatches are found between original_title and title columns, so we can say they are essentially the same data being communicated in the two columns. We can therefore drop the original_title column, which will do so in the next section.
# 
# Let's check now on the status column.

# In[ ]:


# let's check the values in status column
movies.status.value_counts()


# For the questions we are trying to answer, we need to focus on released movies anyway, so we will drop the other types (i.e. Rumored and Post Production) in the section below.

# ## Data Cleaning: Removing irrelevant columns and dropping missing value entries
# 
# It also looks there are a few columns with missing data. However, the first step is to drop columns that are not going to be useful for analysis. Right off the bat, columns like homepage and id are obviously not useful for our analysis. Columns with a lot of unique text for each movie like overview and tagline can be removed because we're not planning to use the plot summary in any text analytics and they are not necessary for the questions we're trying to answer. The columns production_countries and spoken_languages are not needed as well. Hence these aforementioned columns as well as the ones mentioned in the previous section can be removed first.

# In[ ]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.

# drop unnecessary columns
cols_to_drop = ['original_title', 'homepage', 'id', 'overview', 'tagline', 'production_countries', 'spoken_languages']
movies.drop(cols_to_drop, axis=1, inplace=True)


# Now let's drop the movies that are not originally english language movies. Then, we can drop the original_language column as well, because it provides no additional information.

# In[ ]:


# drop rows that are not originally english language movies
non_en_movies_ind = movies[movies.original_language != 'en'].index
movies.drop(non_en_movies_ind, inplace=True)
movies.original_language.value_counts()


# We can see now that we have successfully dropped the non-english language movies. Before dropping the original_language column, let's also drop movies that do have the 'Released' status, so that we can remove the status column at the same time.

# In[ ]:


# drop rows that don't have status 'Released'
unreleased_movies_ind = movies[movies.status != 'Released'].index
movies.drop(unreleased_movies_ind, inplace=True)
movies.status.value_counts()


# Now we have also removed the movies that didn't have the 'Released' status. We can now drop both the original_language and status columns.

# In[ ]:


# drop original_language and status columns
movies.drop(['original_language', 'status'], axis=1, inplace=True)
movies.info()


# As we can see above, all the columns we have removed are gone from the dataframe. The row count has also come down closer to 4500. It looks like we have missing values in the release_date and runtime columns (1 each), which we will also drop as it's only atmost two rows.

# In[ ]:


# drop missing value rows
movies.dropna(inplace=True)
movies.info()


# Let's also make sure that there aren't any zero values in columns like budget and runtime.

# In[ ]:


# check how many movies have budget equal to 0
zero_budget = movies.budget == 0
print("{} movies with 0 budget in dataset".format(zero_budget.sum()))
movies[zero_budget]


# Looking at the preview of some of these movies with 0 reported for budget in this dataset, e.g. Alvin and the Chipmunks: The Road Chip, it's clear that the budget was possibly converted to 0 when actual budget data wasn't provided during the data entry. Based on a Google search, this movie has a budget of 90 million USD. Therefore, we will not drop these zero budget rows, but will exclude them from the analysis when profitability questions are being investigated. Similarly, a revenue of 0 does not make much sense unless perhaps a movie was released directly to DVD/streaming and the revenue being considered here is box-office revenue. Either way, for profitability questions, it makes sense to exclude zero revenue movies as well.

# In[ ]:


# check how many movies have runtime equal to 0
(movies.runtime == 0).sum()


# Because this is such a small portion of the dataset, we will drop these rows using the code below.

# In[ ]:


# drop rows that don't have 0 runtime
nonsense_runtime_ind = movies[movies.runtime == 0].index
movies.drop(nonsense_runtime_ind, inplace=True)


# ## Data Cleaning: Simplify data structures/representations
# 
# Now, we have a dataframe that is relatively clean with no missing values, consisting of 4496 english language movies to analyze. However, before moving onto our exploratory data analysis phase, let's take a look at the genres and keywords columns and see if there is something that can be done to simplify the representation of data. 

# In[ ]:


# movie genres columns, let's take a look at the first entry
movies.genres.iloc[0]


# As you can see, the genre classification for the movies is provided as list of dictionaries, in which each dictionary has both an 'id' and 'name' associated with the genre type. For our analysis purposes, it's much simpler to just have a list of strings, where each string is one of the genres under which the movie falls. So we prefer to convert each list of dictionaries to a list of strings. But first, it should be noted that there are single quotes around the list of dictionaries (as you can see in the output above), which means this is a string representation of a list of dictionaries. Therefore, we will first need to convert this string to a list (using the `ast` module), before extracting the strings associated with the genre types.

# In[ ]:


# generate list from entries in genres column
movie_genre_list = movies.genres.tolist()

# import module ast and convert string representation of list to actual list for each list of genres
import ast
movie_genre_list = [ast.literal_eval(lst) for lst in movie_genre_list]

# for each list of dictionaries, extract the genre strings and make list of strings
new_genre_list = []
for lst in movie_genre_list:
    movie_genres = [dct['name'] for dct in lst]
    new_genre_list.append(movie_genres)

# Overwrite 'genres' columns in dataframe with list of lists
movies['genres'] = new_genre_list

# verify the list of genres for first movie in dataframe
movies.genres.iloc[0]


# Now we have genres for each movies in a simple list from which we can search when needed. Let's take a look at the keywords column now.

# In[ ]:


# get the first entry of keywords column
movies.keywords.iloc[0]


# As we can see, the keywords column has the same format as the genres format had originally. We will follow the same procedure to simplify the data structure here.

# In[ ]:


# generate list from entries in keywords column
movie_keyword_list = movies.keywords.tolist()

# use module ast to convert string representation of list to actual list for each list of keywords
movie_keyword_list = [ast.literal_eval(lst) for lst in movie_keyword_list]

# for each list of dictionaries, extract the keyword strings and make list of strings
new_keyword_list = []
for lst in movie_keyword_list:
    movie_keyword = [dct['name'] for dct in lst]
    new_keyword_list.append(movie_keyword)

# Overwrite 'keywords' columns in dataframe with list of lists
movies['keywords'] = new_keyword_list

# verify the list of keywords for first movie in dataframe
movies.keywords.iloc[0]


# Now that we have successfully cleaned and simplified the data in the dataframe, we are ready to begin the exploratory data analysis (EDA) to answer the questions we have posed.

# <a id='eda'></a>
# # Exploratory Data Analysis
# 
# First, let's look at overall summary statistics of data.

# In[ ]:


# summary stats
movies.describe()


# As we can see, budget and revenue both have mins of 0, which is not sensible in commercial movie production. However, because we don't know the reason behind the zero values (e.g. error in data entry), we will ignore these rows when considering profit-related questions. The remaining numerical variables seem to have sensible values. We don't know what the difference is between popularity and vote_average and whether they correlate well, but we will investigate it during our analysis. We should also note that for profit-related questions, there is no profit column in the data. So as part of question 1 we will create this column as a profit margin (% of budget obtained as profit) variable for those rows that have sensible budget/revenue data.
# 
# # Question 1: Are high budget movies more or less profitable than low budget movies?

# Let's first create a new dataframe for movies containing sensible budget/revenue data and create a profit column with % profit.

# In[ ]:


# create new movie dataframe with revenue/budget > 0
commercial_movie_filter = (movies.revenue > 0) & (movies.budget > 0)
commercial_movies = movies[commercial_movie_filter].copy()

# calculate profit
commercial_movies['profit'] = (commercial_movies.revenue - commercial_movies.budget) / commercial_movies.budget * 100

# view summary
commercial_movies.describe()


# The min value for profit of almost -100% is possible when a movie doesn't make any signifcant amount of money relative to the budget. The max value is not necessarily bounded by some set value, because a very profitable movie can make many multiples of movie budget. However, the 25th percentile indicates 3% profit margin whereas the 75th percenile indicates 335% profit, which defines the interquartile range (IQR) for this distribution of profit values. The max value though is many orders of magnitude higher the 75th percentile, which suggests there are at least outliers on the higher end (perhaps due to input data errors). Let's look at a boxplot of the profit column values to visually identify these outliers.

# In[ ]:


# distribution of profit values in a box-plot form
import seaborn as sns
sns.boxplot(x=commercial_movies.profit);
# commercial_movies.plot(kind='box')


# As we can see from the boxplot distribution, the box is not even visible because the outlier values extend the scale many orders of magnitude beyond the upper hinge (75th percenile). To remove the outliers, we will use the IQR method. This means that a value is an outlier if:
# * it is more than 1.5 times IQR below the lower hinge (25th percentile)
# * it is more than 1.5 times IQR above the upper hinge (75th percentile)

# In[ ]:


# define upper hinge, lower hinge and IQR values
profit_lower_hinge = commercial_movies.profit.quantile(0.25)
profit_upper_hinge = commercial_movies.profit.quantile(0.75)
profit_IQR = profit_upper_hinge - profit_lower_hinge

# filter outliers
profit_outlier_filter = (commercial_movies.profit >= profit_lower_hinge - 1.5 * profit_IQR) & (commercial_movies.profit <= profit_upper_hinge + 1.5 * profit_IQR)
cm_filtered = commercial_movies[profit_outlier_filter].copy()

# see filtered values distribution
print(cm_filtered.describe())
sns.boxplot(x=cm_filtered.profit);


# Now, after we've filtered out the outliers based on the profit margin column and using the IQR approach, we see that the box in the boxplot distribution is visible. Though there are still outlier values (more than 1.5 times the new IQR above the new upper hinge) in this filtered distribution, the scale is much more sensible. The max profit margin is about 825% which is reasonable for very successful movie and much more sensible than something in the order of 1e8 as observed originally. Now let's look at the correlation between profit margin and budget.

# In[ ]:


# scatter plot of profit v budget
ax = cm_filtered.plot(x='budget', y='profit', kind='scatter', figsize=(16,8))
ax.set_xlabel('Movie Budget, USD')
ax.set_ylabel('Profit Margin (%)')
ax.set_title('Relationship between movie profitability and budget');


# Though there is a wide spread of profit margins for low budget movies, a general negative correlation can be observed in the scatter plot indicating that higher budget movies tend to have lower profit margins than lower budget movies. We can calculate mean profit margin for each class, where the 'lower budget' class is defined by a movie budget less than or equal to the mean budget and the 'higher budget' class is defined by a movie budget higher than the mean, to get a more concrete sense.

# In[ ]:


# define mean profit margin for each class
lower_budget_profit_margin = cm_filtered.query('budget <= @cm_filtered.budget.mean()').profit.mean()
higher_budget_profit_margin = cm_filtered.query('budget > @cm_filtered.budget.mean()').profit.mean()
print('Lower budget movies have a mean profit margin of {}%.'.format(lower_budget_profit_margin))
print('Higher budget movies have a mean profit margin of {}%'.format(higher_budget_profit_margin))
fig, ax = plt.subplots()
ax.bar(['Lower', 'Higher'], [lower_budget_profit_margin, higher_budget_profit_margin])
ax.set_xlabel('Movie Budget Class')
ax.set_ylabel('Mean Profit Margin (%)');


# Though the difference in mean profit margin is not high, it still indicates that the *tendency that higher budget movies have lower profit margins*. The difference is likely going to be more significant when comparing profit margins of movies with budgets below the 25th percentile and above the 75th percentile. **The potential reason for this observation is that higher budget movies require very high revenues proportional to their budget level to have high profit margins. Given the large number of movies that any one movie is competing with and a high budget is not necessarily a guarantee of commercial success, the higher budget class of movies have a tougher time attaining and/or beating the profit margin levels of lower budget movies.**

# # Question 2: Are more profitable movies released in a specific time of year?

# For this question, we can just use the filtered dataframe `cm_filtered` to exclude the effect of extreme profit outliers. Let's first create a `release_month` column from values in `release date`. If we take a look at the release date format...

# In[ ]:


# release date values
cm_filtered.release_date


# The release date is given in YYYY-MM-DD string format. We can use the datetime module to extract parts of this date (year, month, day) as needed to create our required column.

# In[ ]:


# extract month from release_date
from datetime import datetime as dt
release_month = list(map(lambda x: dt.strptime(x, "%Y-%m-%d").month, cm_filtered.release_date))

# create release_month column in dataframe
cm_filtered['release_month'] = release_month


# Now let's look at the spread of profit margin in relation to release month of the movie using box plot distributions by month.

# In[ ]:


# distributions of profit margin by release month
fig, ax = plt.subplots(figsize=(16,8))
# cm_filtered.boxplot(column=['profit'], by=['release_month'], ax=ax)
sns.boxplot(x='release_month', y='profit', data=cm_filtered, ax=ax)
ax.set_xlabel('Movie Release Month')
ax.set_xticks(range(13))
ax.set_ylabel('Profit Margin (%)')
ax.set_title('Analysis of profit margin in relation to movie release month');


# Though there's not a marked distinction in profit margin in any month relative to the rest of the months, *the months of May, June and December seem to have the highest profit margins on average*. **This phenomenon is potentially explained by the fact that May/June movie releases coincide with the start of summer and December movie releases coincide with the start of winter. During both of these periods, schools (grade schools and universities) go on break, which means students along with their friends and families are more likely to go to theatres and watch movies, especially new releases at the time, than at other times of the year.**

# # Question 3: Are longer movies more or less popular than shorter movies?
# 
# To answer this question, we may considering using one of the two variables in the dataset that reflect the popular reception of a movie, i.e. `popularity` and `vote_average`. Let's take a look at the distribution of these scoring scales and the relationship between them.

# In[ ]:


movies.popularity.hist();


# In[ ]:


movies.vote_average.hist();


# In[ ]:


movies.plot(x='vote_average', y='popularity', kind='scatter', figsize=(16,8));


# Based on the results above, I choose to use vote average as a better measure of how well received a movie is, because it resembles the 0-10 score range of the well-known IMDb movie ratings scale. Moreover, we don't have to deal with the very right skewed distribution of the popularity variable. In general, there seems to be a positive correlation between popularity and vote average ratings, but we do see that the popularity value can be 0 or close to 0 even for highly rated movies based on vote average. In addition, vote average seems to be calculated based on a user provided rating between 0-10, according to the TMDb website. For all these, vote average makes more sense to use. Now let's see the relationship between runtime and vote average.

# In[ ]:


# scatter plot of vote average and runtime
ax = movies.plot(x='runtime', y='vote_average', kind='scatter', figsize=(16,8))
ax.set_xlabel('Movie Runtime (min)')
ax.set_ylabel('Average Rating (10-point scale)')
ax.set_title('Relationship between movie runtime and rating');


# Though it's not a strong correlation, there seems to a general positive trend between movie runtime and vote average. So, *longer movies tend to receive higher ratings.* According to a study by Rotten Tomatoes (https://screenrant.com/longer-movies-fresh-ratings-rotten-tomatoes/), longer movies are better rated, potentially because movie studios only decide to produce longer movies if they deem the quality of the movie well worth it. So, longer movies scripts that are of lower quality are weeded out before they ever get produced, which helps keep up the average rating of longer runtime movies. Other possible reasons are also discussed at the provided link.

# # Question 4: Are certain genres of movies more popular than others? Do certain genres generate more profit?
# 
# ## Movie Rating
# 
# For this question, we will use the `cm_filtered` dataset because we are going to analyze profit margin as well movie rating. For the first part of this question, we will segment the vote average of movies into four categories:
# * Very poor (<=2.5)
# * Poor (2.5<x<=5)
# * OK (5<x<=7.5)
# * Good (>7.5)
# 
# Then we will tally the top 3 genres associated with each category of movie rating, which will give us an indication of which genres of movies are more liked by people.

# In[ ]:


# create rating category column
rating = ['Very poor' for i in range(len(cm_filtered))]
cm_filtered['rating'] = rating
cm_filtered.loc[((cm_filtered.vote_average > 2.5) & (cm_filtered.vote_average <= 5)), 'rating'] = 'Poor'
cm_filtered.loc[((cm_filtered.vote_average > 5) & (cm_filtered.vote_average <= 7.5)), 'rating'] = 'OK'
cm_filtered.loc[(cm_filtered.vote_average > 7.5), 'rating'] = 'Good'

# create a dictionary (4 elements) of dictionaries to keep track of counts of genres associated with each movie in the 4 rating categories
genres = {'Very poor': {}, 'Poor': {}, 'OK': {}, 'Good': {}}

# loop through and populate dictionaries of counts
for i in range(len(cm_filtered)):
    genre_list = cm_filtered.genres.iloc[i]
    rating = cm_filtered.rating.iloc[i]
    for genre in genre_list:
        if genre not in genres[rating]:
            genres[rating][genre] = 1
        else:
            genres[rating][genre] += 1


# Now that we have the tally counts, let's make a plot of the top 3 genres for the four rating categories of movies.

# In[ ]:


# sort to be able to get the top 3 genres in each rating category
genres_bar = {}
for key in genres:
    genres[key] = {k: v for k, v in sorted(genres[key].items(), key=lambda item: item[1], reverse=True)}

# visualization setup
n = 5
fig, axes = plt.subplots(2,2, figsize=(15,10))
sns.barplot(x=list(genres['Very poor'].keys())[:n], y=list(genres['Very poor'].values())[:n], ax=axes[0,0])
axes[0,0].set_yticks([0,1,2])
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Rating: Very poor')
sns.barplot(x=list(genres['Poor'].keys())[:n], y=list(genres['Poor'].values())[:n], ax=axes[0,1])
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Rating: Poor')
sns.barplot(x=list(genres['OK'].keys())[:n], y=list(genres['OK'].values())[:n], ax=axes[1,0])
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Rating: OK')
sns.barplot(x=list(genres['Good'].keys())[:n], y=list(genres['Good'].values())[:n], ax=axes[1,1])
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Rating: Good')
fig.suptitle('Top {} Genres in Each Movie Rating Category'.format(n));


# Seeing the top 5 genres in both highly-rated and poorly-rated movies, there doesn't seem to be many genre associations that set apart good movies from bad ones. Drama, Thriller, Comedy, Action and Adventure are genre associations present in good and bad movies. The only genre that is unique in the 'Good' category is Crime, however it's hard to draw any strong conclusions from these results.
# 
# ## Profit Margin
# 
# Given the result from the previous analysis, we don't expect to see any revelations from the profit angle as well. However, we will follow a similar approach to categorize profit margins and count genre associations in each category. The following will be the profit margin categories:
# * Poor RoI: <=10% profit margin
# * Acceptable RoI: 10<x<=20% profit margin
# * Good RoI: 20<x<=50% profit margin
# * Excellent RoI: >50% profit margin

# In[ ]:


# create profit margin category column
roi = ['Poor' for i in range(len(cm_filtered))]
cm_filtered['roi'] = roi
cm_filtered.loc[((cm_filtered.profit > 10) & (cm_filtered.profit <= 20)), 'roi'] = 'Acceptable'
cm_filtered.loc[((cm_filtered.profit > 20) & (cm_filtered.vote_average <= 50)), 'roi'] = 'Good'
cm_filtered.loc[(cm_filtered.profit > 50), 'roi'] = 'Excellent'

# create a dictionary (4 elements) of dictionaries to keep track of counts of genres associated with each movie in the 4 RoI categories
genres2 = {'Poor': {}, 'Acceptable': {}, 'Good': {}, 'Excellent': {}}

# loop through and populate dictionaries of counts
for i in range(len(cm_filtered)):
    genre_list = cm_filtered.genres.iloc[i]
    roi = cm_filtered.roi.iloc[i]
    for genre in genre_list:
        if genre not in genres2[roi]:
            genres2[roi][genre] = 1
        else:
            genres2[roi][genre] += 1

# sort to be able to get the top 3 genres in each RoI category
genres_bar = {}
for key in genres2:
    genres2[key] = {k: v for k, v in sorted(genres2[key].items(), key=lambda item: item[1], reverse=True)}

# visualization setup
n = 5
fig, axes = plt.subplots(2,2, figsize=(15,10))
sns.barplot(x=list(genres2['Poor'].keys())[:n], y=list(genres2['Poor'].values())[:n], ax=axes[0,0])
axes[0,0].set_yticks([0,1,2])
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('RoI: Poor')
sns.barplot(x=list(genres2['Acceptable'].keys())[:n], y=list(genres2['Acceptable'].values())[:n], ax=axes[0,1])
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('RoI: Acceptable')
sns.barplot(x=list(genres2['Good'].keys())[:n], y=list(genres2['Good'].values())[:n], ax=axes[1,0])
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('RoI: Good')
sns.barplot(x=list(genres2['Excellent'].keys())[:n], y=list(genres2['Excellent'].values())[:n], ax=axes[1,1])
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('RoI: Excellent')
fig.suptitle('Top {} Genres in Each RoI Category'.format(n));


# Unsurpisingly, we don't see any distinguising genre associations with movies that were commercial successes compared to those that were failures. Hence, genre doesn't seem to be a good indicator of whether a movie will be generally well received or if it will be a commercial success.

# # Question 5: Are Marvel movies more or less popular and profitable than DC movies? Are Marvel movies more popular than the average movie? Are they more profitable than the average movie?
# 
# For this question, we will use the `cm_filtered` dataframe because we are answering questions regarding profit. We will first need to develop a new column to identify whether a movie is a Marvel, DC or Other movie.

# In[ ]:


# See how Marvel and DC movies are coded in keywords column
marvel_count = 0
dc_count = 0
for lst in cm_filtered.keywords:
    marvel = False
    dc = False
    for item in lst:
        if 'marvel' in item:
            print(item)
            if not marvel:
                marvel_count += 1
                marvel = True
        elif 'dc' in item:
            print(item)
            if not dc:
                dc_count += 1
                dc = True
print("Marvel count: ", marvel_count)
print("DC count: ", dc_count)


# There are total of 33 Marvel movies and 19 DC comics movies, excluding the results that contained 'dc' as part of a keyword from the total count.

# In[ ]:


# create comic category column
comic = ['Other' for i in range(len(cm_filtered))]
cm_filtered['comic'] = comic
for i in range(len(cm_filtered)):
    if ('marvel comic' in cm_filtered.keywords.iloc[i]) or ('marvel cinematic universe' in cm_filtered.keywords.iloc[i]):
        cm_filtered.comic.iloc[i] = 'Marvel'
    elif ('dc comics' in cm_filtered.keywords.iloc[i]) or ('dc extended universe' in cm_filtered.keywords.iloc[i]):
        cm_filtered.comic.iloc[i] = 'DC'

# check how many of each type
cm_filtered.comic.value_counts()


# Now we can compare Marvel movies to DC movies on profitability and rating. We can also compare Marvel movies to the average movie as well.

# In[ ]:


# Marvel and DC movies rating and profit comparison
marvel_rating = cm_filtered[cm_filtered.comic == 'Marvel'].vote_average.mean()
marvel_profit = cm_filtered[cm_filtered.comic == 'Marvel'].profit.mean()
dc_rating = cm_filtered[cm_filtered.comic == 'DC'].vote_average.mean()
dc_profit = cm_filtered[cm_filtered.comic == 'DC'].profit.mean()

# Plot
labels = ['Marvel', 'DC']
ratings = [marvel_rating, dc_rating]
profits = [marvel_profit, dc_profit]
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
sns.barplot(labels, ratings, ax=ax1)
ax1.set_ylabel('Mean Rating')
sns.barplot(labels, profits, ax=ax2)
ax2.set_ylabel('Mean Profit Margin (%)')
fig.suptitle('Performance of Marvel vs DC Movies');


# In[ ]:


# Marvel and average movie rating and profit comparison
avg_rating = cm_filtered.vote_average.mean()
avg_profit = cm_filtered.profit.mean()

# Plot
labels = ['Marvel', 'Average']
ratings = [marvel_rating, avg_rating]
profits = [marvel_profit, avg_profit]
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
sns.barplot(labels, ratings, ax=ax1)
ax1.set_ylabel('Mean Rating')
sns.barplot(labels, profits, ax=ax2)
ax2.set_ylabel('Mean Profit Margin (%)')
fig.suptitle('Performance of Marvel vs Average Movie');


# Based on the rating and profit margin comparisons, Marvel comics movies are on average rated about half a point higher than DC comics movies and have much higher average profit margin at close to 250% compared to the DC average at ~150%. In comparison to the average movie, Marvel movies don't have a noticeably higher rating but have a much higher ~RoI, with Marvel at 250% vs the average movie at ~150%.

# <a id='conclusions'></a>
# # Conclusions
# 
# To summarize our findings:
# 1. Higher budget movies tend to have lower return on invesment than lower budget movies
# 2. Movies released during the months of May, June and December tend to perform better commercially than movies released at other times of the year
# 3. Longer movies, i.e. movies with longer runtime, tend to do better commercially than shorter ones
# 4. Genre doesn't seem to be a good indicator of whether a movie is well received and whether it performs well commercially
# 5. Marvel movies tend to perform better commercially and have higher rating than DC comics movies as well as the average movie
# 
# It should be noted that these findings are simply correlations and simple comparisons. They do NOT indicate causation, e.g. higher budget doesn't necessary cause less profit. Typically a combination of factors will impact how well a movie does in terms of public reception and commercial success. More rigorous analysis and machine learning is required to predict which movies will end up doing well and which will do poorly.
