#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Movies

# ## The Dataset

# In[34]:


import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))


# The data from our dataset can be properly loaded into pandas dataframes, setting each sample's index to its id, using the *load_tmdb_credits* and *load_tmdb_movies* functions available [here](https://www.kaggle.com/sohier/tmdb-format-introduction).

# In[35]:


def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")


# Let's take a look at some of the samples in our dataset.

# In[36]:


credits.head()


# In[37]:


movies.head()


# For consistency between the two dataframes, let's rename the 'id' field in the movies dataframe as 'movie_id'.

# In[38]:


movies.rename(columns={"id": "movie_id"}, inplace=True)


# ## Data Cleaning

# We can see that some of the data in certain fields are very raw in both the credits and movies datasets. Most of these contain a lot of information we don't need, so lets clean them up a bit. First, let's start with the credits dataset, and view a sample of the credits' cast and crew fields.

# In[39]:


credits["cast"].iloc[0]


# In[40]:


credits["crew"].iloc[0]


# Following the methodology outlined [here](https://www.kaggle.com/sohier/tmdb-format-introduction), we can replace the credits dataframe with a people dataframe that displays the role played by actors and crew members on various films in a much cleaner and manageable way..

# In[41]:


credits.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['cast']], axis=1)
credits.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['crew']], axis=1)
credits.apply(lambda row: [person.update({'order': order}) for order, person in enumerate(row['crew'])], axis=1)

cast_list = []
credits["cast"].apply(lambda x: cast_list.extend(x))
cast = pd.DataFrame(cast_list)
cast["type"] = "cast"

crew_list = []
credits["crew"].apply(lambda x: crew_list.extend(x))
crew = pd.DataFrame(crew_list)
crew["type"] = "crew"

people = pd.concat([cast, crew], ignore_index=True, sort=True)
del credits


# In[42]:


people.head()


# Let's now move on to our movies dataset. Some of the fields in movies contain a lot of unnecessary information, and can also be quite difficult to read. These fields are namely the genres, keywords, production_companies, production_countries, and spoken_languages columns. The data in these fields might be better suited to appearing in their own dataframe for each field, so that we can better discern which factors for a given field appear for a given film. We shall therefore create a separate dataframe for each of the above mentioned fields, and drop them from our movies dataset.

# In[43]:


movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['genres']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['keywords']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['production_companies']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['production_countries']], axis=1)
movies.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['spoken_languages']], axis=1)

genres = []
movies["genres"].apply(lambda x: genres.extend(x))
genres = pd.get_dummies(pd.DataFrame(genres).drop("id", axis=1).set_index("movie_id")).sum(level=0)
genres.rename(columns = lambda x: str(x)[5:], inplace=True)

keywords = []
movies["keywords"].apply(lambda x: keywords.extend(x))
keywords = pd.get_dummies(pd.DataFrame(keywords).drop("id", axis=1).set_index("movie_id")).sum(level=0)
keywords.rename(columns = lambda x: str(x)[5:], inplace=True)

production_companies = []
movies["production_companies"].apply(lambda x: production_companies.extend(x))
production_companies = pd.get_dummies(pd.DataFrame(production_companies).drop("id", axis=1).set_index("movie_id")).sum(level=0)
production_companies.rename(columns = lambda x: str(x)[5:], inplace=True)

production_countries = []
movies["production_countries"].apply(lambda x: production_countries.extend(x))
production_countries = pd.get_dummies(pd.DataFrame(production_countries).drop("iso_3166_1", axis=1).set_index("movie_id")).sum(level=0)
production_countries.rename(columns = lambda x: str(x)[5:], inplace=True)

spoken_languages = []
movies["spoken_languages"].apply(lambda x: spoken_languages.extend(x))
spoken_languages = pd.get_dummies(pd.DataFrame(spoken_languages).drop("iso_639_1", axis=1).set_index("movie_id")).sum(level=0)
spoken_languages.rename(columns = lambda x: str(x)[5:], inplace=True)

movies.drop(["genres", "keywords", "production_companies", "production_countries", "spoken_languages"], axis=1, inplace=True)


# Viewing our movies dataset now we get the following:

# In[44]:


movies.head()


# There are some fields in movies that probably aren't of much use to us. There is probably not too much useful information we can yield from having a film's homepage or it's original title so we will delete those fields. Since we already have information about a given film's genres and keywords relating to it, we probably don't also need to have an overview of the film or its tagline.

# In[45]:


movies.drop(["homepage", "original_title", "overview", "tagline"], axis=1, inplace=True)


# Presumably the status of most films would be 'Released' so we can probably scrap that field as well, but just for curiosity's sake, let's look at a bar graph showcasing the frequency of film statuses in our dataset.

# In[46]:


sns.countplot(movies["status"])


# Clearly the status of films is overwhelming 'Released', so we will simply remove that field from our dataset.

# In[47]:


movies.drop(["status"], axis=1, inplace=True)


# The original language of the films in our dataset may also be overwhelming English, and thereby not yield much useful information for us. Let's take a look at the distribution of top original languages in our movies dataset.

# In[48]:


movies["original_language"].value_counts().sort_values().tail().plot.barh()


# As we can see, the films in our dataset are predominantly in English. Therefore, this field probably won't yield too much useful information for us and can be dropped.

# In[49]:


movies.drop(["original_language"], axis=1, inplace=True)


# When it comes to a film's release date, we'd presumably be less interested in the exact day it was released and would be more curious to see what year it was released, or what month of the year it was released in. We shall therefore add two fields to our movies dataset, one for the year of release and one for the month of release, and remove the release_date field.

# In[50]:


movies["release_date"] = pd.to_datetime(movies["release_date"])
movies["release_year"] = movies["release_date"].dt.year
movies["release_month"] = movies["release_date"].dt.month
movies.drop(["release_date"], axis=1, inplace=True)


# We will also set the movie_id to the index for our movies dataframe.

# In[51]:


movies.set_index("movie_id", inplace=True)


# We now have the following much cleaner movies dataset.

# In[52]:


movies.head()


# Let's now have some fun exploring the data a bit, and see what insights we can gather from it.

# ## Data Distributions

# ### Movie Data Distributions

# Let's see how different features of the movies dataset are distributed.

# In[53]:


fig, axarr = plt.subplots(4, 2, figsize=(24, 8))
sns.kdeplot(movies["budget"], ax=axarr[0][0])
axarr[0][0].xaxis.set_ticks(np.arange(0, 4.25e8, 0.25e8))
sns.kdeplot(movies["revenue"], ax=axarr[0][1])
axarr[0][1].xaxis.set_ticks(np.arange(0, 3e9, 0.25e9))
sns.kdeplot(movies["runtime"], ax=axarr[1][0])
sns.kdeplot(movies["popularity"], ax=axarr[1][1])
axarr[1][1].xaxis.set_ticks(np.arange(0, 900, 50))
sns.kdeplot(movies["vote_average"], ax=axarr[2][0])
axarr[2][0].xaxis.set_ticks(np.arange(0, 11, 1))
sns.kdeplot(movies["vote_count"], ax=axarr[2][1])
sns.kdeplot(movies["release_year"], ax=axarr[3][0])
sns.countplot(movies["release_month"], ax=axarr[3][1])
fig.tight_layout()


# Looking at the above results, we can see that most films in our dataset have a budget of less than \$10 million (presumably USD) and an accumulated revenue of less than $100 million. We can also see that most films have a runtime of around 100 minutes or 1 hour and 45 minutes. In terms of film acclaim and recognition in our dataset, we can see that most films have a popularity score around 20, with a vote average score usually between 6.0 and 7.0, and a vote count numbering less than 500. Looking at the release dates of films in our dataset, we can see that it primarily contains films released after 1990, fairly evenly distributed across all months of the year, with most being released in September.

# ### People Data Distributions

# Let's first see the distribution between cast and crew in our dataset.

# In[54]:


sns.countplot(people["type"])


# As we can see, this indicates there are more crew members than cast members in our dataset. However, there can be multiple instances of the same cast/crew member in the people dataframe indicating different roles for different films that they have worked on. Let's visualize the number of unique instances of cast and crew members.

# In[55]:


sns.countplot(people.drop_duplicates(["id"])["type"])


# Interestingly, there are actually more unique cast members than crew members in our dataset. Let's visualize the gender distribution of these unique people.

# In[56]:


fig, axarr= plt.subplots(1, 3, figsize=(24, 4))
sns.countplot(people.drop_duplicates(["id"])[people["type"] == "cast"]["gender"], ax=axarr[0])
sns.countplot(people.drop_duplicates(["id"])[people["type"] == "crew"]["gender"], ax=axarr[1])
sns.countplot(people.drop_duplicates(["id"])["gender"], ax=axarr[2])
axarr[0].set_title("Cast")
axarr[1].set_title("Crew")
axarr[2].set_title("Overall")
for i in range(3):
    axarr[i].set_xticklabels(["Undefined", "Male", "Female"])
fig.tight_layout()


# Unfortunately, due to the vast number of persons having an undefined gender in our dataset, it is unclear what is the gender distribution in our dataset.

# Fortunately, we can see the distribution of departments that crew members work in, as well as the ten most prevalent jobs in our dataset.

# In[57]:


fig, axarr = plt.subplots(1, 2, figsize=(24, 4))
sns.countplot(y=people["department"], ax=axarr[0])
people["job"].value_counts().head(10).plot.barh(ax=axarr[1])
axarr[1].set_ylabel("job")
fig.tight_layout()


# Looking at the above visualization, we can see that most crew members are in the production department, with most crew members having the job title of "Producer". Surprisingly, there are a very small number of crew members designated as being in the "Actors" department. Let's take a look at these persons in our people dataset.

# In[58]:


people[people["department"] == "Actors"]


# As we can see, these crew members were stunt doubles, and not considered part of the cast.

# ### Miscellaneous Data Distributions

# Let's also view the data distributions of the most prevalent instances in our other datasets.

# In[59]:


fig, axarr = plt.subplots(3, 2, figsize=(20, 8))
genres.sum().plot.barh(ax=axarr[0][0])
keywords.sum().sort_values().tail(10).plot.barh(ax=axarr[0][1])
production_companies.sum().sort_values().tail(10).plot.barh(ax=axarr[1][0])
production_countries.sum().sort_values().tail(10).plot.barh(ax=axarr[1][1])
spoken_languages.sum().sort_values().tail().plot.barh(ax=axarr[2][0])
axarr[0][0].set_ylabel("genre")
axarr[0][1].set_ylabel("keyword")
axarr[1][0].set_ylabel("production_company")
axarr[1][1].set_ylabel("production_country")
axarr[2][0].set_ylabel("spoken_language")
axarr[2][1].axis("off")
fig.tight_layout()


# As one may expect, the most prevalent film genre in our dataset is 'Drama', followed by 'Comedy', with 'TV Movie' and 'Foreign' films being the least represented. Viewing the most popular keywords in this data, we can also see how diverse the content of these films as well as their production can be. The most popular production companies are also ones that most people would expect, with Warner Bros., Universal Pictures, and Paramount Pictures being head of the pack. The majority of these movies were also produced in the United States, with English being the film's spoken language.

# ## Blockbusters

# With some initial exploration of the distribution of data in our datasets now completed, let's explore what factors have the greatest impact on a film's box office success. Let's start by observing the top ten films with the highest revenue.

# In[60]:


ax = movies.nlargest(10, "revenue").iloc[::-1].plot.barh(x="title", y="revenue", legend=False)
ax.set_xlabel("revenue")
ax.set_ylabel("film")


# As we can see, all of these filmed have garnered over $1 billion revenue, with 'Avatar' clearly being the highest grossing film.

# Let's now begin to see the relationship different factors may have with box office success.

# In[ ]:


fig, axarr = plt.subplots(4, 2, figsize=(20, 24))
p_color = dict(color="C0")
l_color = dict(color="C1")
sns.regplot(x="budget", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[0][0])
sns.regplot(x="runtime", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[0][1])
sns.regplot(x="release_year", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[1][0])
sns.boxplot(x="release_month", y="revenue", data=movies, ax=axarr[1][1])
sns.regplot(x="popularity", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[2][0])
sns.regplot(x="vote_average", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[2][1])
sns.regplot(x="vote_count", y="revenue", data=movies, fit_reg=True, scatter_kws=p_color, line_kws=l_color, ax=axarr[3][0])
fig.tight_layout()


# As we can see, there is a pronounced correlation between film budget and revenue, with films with higher production costs generally collecting more revenue. There is also interestingly a positive correlation, though less pronounced, between movie runtime and revenue, with longer movies generally accumulating higher revenue. The year a film was released in also doesn't seem to have much impact on the revenue it generates, so we cannot necessarily expect newer films to generate more revenue than older movies. Viewing the boxplots comparing film release month with revenue, we can see that most high grossing films seem to have been released in the months of May, June, November, and December, with films released in January, September, and October seeming to be less financially successful. The more revenue a films accumulates unsurprisingly also seems to correlate with its popularity score and the number of votes it receives. High grossing films also seem to garner higher vote ratings, though this positive correlation is much less pronounced than that between revenue and popularity and vote count.

# Let's explore what other factors correlate with film revenue.

# ## To be continued...

# In[ ]:




