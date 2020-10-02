#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigating Movie Data
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#limitations">Limitations</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > The dataset is from The Movie Database (TMDb) which is a collection of information from about 10,000 movies. Information includes, title, cast, director, user ratings, revenue, budget, genre, release date, etc. The investigation will explore revenue and how they are affected by genres, popularity, ratings, etc. 

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties

# In[4]:


df_movie = pd.read_csv('../input/tmdb_movies.csv')
df_movie.head()


# In[5]:


df_movie.describe()


# * So initially we see the counts for each of the numerical data has the same which is 10866 so there isn't any missing data for those columns. 
# * The minimum for budget, revenue, budget_adj, and revenue_adj is 0 which would probably indicate the data maybe was not available considering I don't think a movie could be made without funds or would not have garnered any revenue. These will be investigated.
# * The minimum for runtime is 0 as well which probably indicates an issue in data because movies have to have a runtime to exist. 
# * The vote_average is logical because the scale is from 1-10 and the minimum and maximum are within those ranges with the mean being 5.975922 which is mostly centered. 
# * The earliest release year is 1960 and latest is 2015 which is around when the dataset was created so that follows logic, with the mean release year being 2001. 
# * The popularity rating seems skewed because the maximum is 32.985763 but the mean is 0.646441, so this will need further investigation. 

# In[6]:


df_movie.info()


# * This shows the datatypes and counts of entries per column.
# * There is a significant lack of data in the "homepage," "tagline," "keywords," and "production_companies," which those columns could probably be tossed out. 

# In[7]:


df_movie.query('runtime == 0')['original_title']


# * A query was ran to see the titles of the movies with runtime equaling 0.
# * Some of the titles are recognizable and because this is a userbased website where users contribute the data for these films, that is most likely the reason for 0 input for some of these titles. I could google search each runtime and enter it in manually but since it is such a small portion I'm going to just eliminate these rows most likely. 

# In[8]:


df_movie.query('budget_adj == 0')['original_title'].count()


# * A query for budget_adj equaling 0 was ran in order to see how many movies lacked budget info. Since this is a user-contributed site, that is most likely the reason for the lack of data and there are 5696 missing rows which would be impossible to fill manually. Also since a movie has to be funded somehow there is no point in analyzing a "free" movie. 

# 
# 
# ### Data Cleaning 

# In[9]:


df_movie2 = df_movie.drop(['homepage','imdb_id','tagline','keywords','overview','production_companies','release_date','budget','revenue','cast'], axis=1)


# * Got rid of the following columns: homepage, imdb_id, tagline, keywords, overview, production_companies, release_date, budget, revenue, cast
# * These lacked data in most rows and didnt seem relevant to the information trying to be obtained.
# * Also we have the release year so release date was tossed out and the adjusted revenue and budget based on inflation is included so the original values were not deemed useful for analysis.

# In[10]:


df_movie3 = df_movie2.query('revenue_adj != 0').query('budget_adj != 0')
df_movie3.info()


# * removed rows where adjusted budget and revenue was 0 because it was missing data which would skew the results

# In[11]:


df_movie3.to_csv('movies_clean.csv', index=False)


# In[12]:


df_movie3['genres_2'] = df_movie['genres'].str.split('|').str[0]


# * Wanted to drop the multiple genres for films and choose the first one as the official genre for the movie, so a new column 'genres_2' was created

# In[13]:


df_movie3.drop(['genres'], axis=1, inplace=True)


# In[14]:


df_movie3.drop(['id'], axis=1, inplace=True)


# * Dropped the original genres and id columns because they were unnecessary for the analysis

# In[15]:


df_movie3.genres_2.value_counts()


# * a more concise and workable categories for genres found and since there was only one type found under TV Movie, the row will be dropped

# In[16]:


df_movie3.query("genres_2 == 'TV Movie'")


# In[17]:


df_movie3.drop(8615, inplace=True)


# In[ ]:





# In[18]:


df_movie3.vote_average.describe(), df_movie3.popularity.describe()


# In[19]:


be_votes = [2.2, 5.7, 6.2, 6.7, 8.4]
be_pop = [0.001117, 0.463068, 0.797723, 1.368403, 32.985763]


# In[20]:


bin_names_votes = ['Low', 'Below Average', 'Above Average', 'High']
bin_names_pop = ['Non Popular', 'Semi Popular', 'Popular', 'Very Popular']


# In[21]:


df_movie3['rating_level'] = pd.cut(df_movie3['vote_average'], be_votes, labels=bin_names_votes)
df_movie3['pop_level'] = pd.cut(df_movie3['popularity'], be_pop, labels=bin_names_pop)


# In[22]:


df_movie3.info()


# In[23]:


df_movie3["pop_level"] = df_movie3["pop_level"].astype('object')
df_movie3["rating_level"] = df_movie3["rating_level"].astype('object')


# * Changed new columns from type category to objects for easier use with labels on graphs for later. 

# In[24]:


df_movie3.info()


# In[25]:


df_movie3.isnull().sum()


# In[ ]:





# In[26]:


df_movie3.dropna(axis=0, inplace=True)


# In[27]:


df_movie3.info()


# * Found one NaN value in director column, one in the rating_level column, and one in the pop_level column. So these were removed and 3851 samples remain. 

# In[28]:


df_movie3.to_csv('movies_clean2.csv', index=False)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 

# ### Is the popularity of the movie associated with higher revenues?

# In[29]:


sns.relplot(x="popularity", y="revenue_adj", data=df_movie3);


# * A simple scatter plot can show a positive correlation between level of popularity and revenue.
# * A more appealing approach would be to categorize the levels of popularity based on the quartile levels of popularity. 
# * Then a pie chart could visually see the variance in total revenue based on popularity. 

# In[30]:


tot_rev = df_movie3.groupby('pop_level')['revenue_adj'].sum()
tot_rev


# In[31]:


level_pops = df_movie3.pop_level.unique()
level_pops.sort()


# In[32]:


def pie_chart():
    sns.set(context="notebook")
    plt.pie(tot_rev, labels=level_pops, autopct='%1.1f%%')
    plt.legend(title="Popularity",
              loc="center right",
              bbox_to_anchor=(1.5, 0, 0.5, 1));


# In[ ]:





# ### Which genres of movies have the higher revenues on average?

# In[33]:


df_movie3.groupby('genres_2').revenue_adj.describe()


# * the describe function can give general statistics on each genre of film and in reference to revenue
# * Some boxplots of this data would be easier to see which genres fair better in terms of revenue. 

# In[34]:


boxplot_data = df_movie3.query('revenue_adj < 500000000')


# * A query for revenue less than $500,000,000 was chosen to show where most of the data is clustered to make the boxplots more visually appealing. 

# In[35]:


def boxplot():
    sns.set(style="ticks", font_scale=1.75)
    sns.catplot(x="genres_2", y="revenue_adj", kind="box", height=10, aspect=2, data=boxplot_data)
    plt.xlabel("Genres")
    plt.ylabel("Revenue in $100,000,000s")
    plt.xticks(rotation=45);


# In[36]:


pie_chart()


# In[37]:


boxplot()


# <a id='conclusions'></a>
# ## Conclusions
# 
# 
# > **Pie Chart**: 
# * The total revenue divided among the popularity levels shows that more popular movies generate higher revenues. This is most likely because the popularity of a movie generates more "hype" which can create interest with more people willing to see a movie. 
# * According to the TMDb website, the popularity rating is based on:
#      Number of votes for the day
#      Number of views for the day
#      Number of users who marked it as a "favourite" for the day
#      Number of users who added it to their "watchlist" for the day
#      Release date
#      Number of total votes
#      Previous days score
# * Therefore any correlation between revenue and popularity is independent of one another and does not directly contribute to the score. 
# 
# > **Box Plot**:
# * The boxplot shows that Animation has the overall highest sales of revenue between all the genres. 
# * The median is significantly higher than the others which would indicate more data points in higher grossing revenue range. 
# * Documentary had the lowest revenues along with Mystery.
# 

# <a id='limitations'></a>
# ## Limitations
# 
# * There was a substantial amount of movie data excluded based on missing data values in the budget and revenue columns. 
# * Certain movies had multiple genres listed under them but only the first one listed was used in this analysis. 
# * The sample size for certain genres was limited which might affect the results. 
# 
