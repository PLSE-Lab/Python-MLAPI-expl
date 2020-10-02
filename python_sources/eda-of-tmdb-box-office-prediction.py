#!/usr/bin/env python
# coding: utf-8

# # TMDB Box Office Prediction
# 
# In this competition we need to predict overall worldwide box office revenue for movies.
# 
# My main goal is to conduct exploratory data analysis (EDA) of dataset and answer to below questions:
# - What movies makes the most money at the box office?
# - How much does a director matter?
# - How much does the budget matter?
# 
# Before we will dive into code you need to know a little bit about me, so you will know what to expect. I am new to data science and before this kernel I only tried to predict survivals of the Titanic, prices of houses and digit recognition (MNIST). As you see I do not have much experience, so if you have any suggestions or find mistakes let me know.
# 
# Moreover I am not a native English speaker so you need to expect some mistakes.

# # EDA

# In[1]:


# Import libraries
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

import json
import ast

from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.probability import FreqDist

from collections import Counter

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)


# In[2]:


# Load train and test data
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

# Print shape of train and test data
print('Train data shape: ', train.shape)
print('Test data shape: ', test.shape)


# There are 3000 movies in train data and 4398 movies in test data. Both sets have 22 features (train have +1 because of box office revenue).

# In[ ]:


# Lets inspect some sample rows from train data
train.sample(3)

# Lots of features are dictionaries so we will need to take closer look to them.
# Some features like homepage cound be binarized.


# In[ ]:


# Describe train data
train.describe()


# In[ ]:


# Inspect test data
test.sample(3)


# In[ ]:


# Check missing data
pd.concat([train.isnull().sum(), test.isnull().sum()], keys=['train','test'], axis=1, sort=False)


# I will go through all features keeping the order form above table and try to analyze the data.

# In[ ]:


# How many missing values have feature in train and test set.
def how_many_missing_values(feature):
    print('Missing {} in train set: {}/{} ({}%)'.format(feature,
                                                        train[feature].isnull().sum(), 
                                                        train[feature].size,
                                                        train[feature].isnull().sum()/train[feature].size))

    print('Missing {} in test set: {}/{} ({}%)'.format(feature,
                                                       test[feature].isnull().sum(), 
                                                       test[feature].size,
                                                       test[feature].isnull().sum()/test[feature].size))


# ## id
# 
# Just the id of the movie. Starts from number 1. Feature not important and can be deleted.

# In[ ]:


# Remove id feature
#train.drop(labels=['id'], axis=1, inplace=True)
#test.drop(labels=['id'], axis=1, inplace=True)


# ## belongs_to_collection
# 
# This feature represents if a movie belongs to a series of movies. For example James Bond Collection.

# In[ ]:


# Count uniqe values for belongs_to_collection
train['belongs_to_collection'].value_counts().head(10)


# In[ ]:


# Get the collection name
def get_collection_name(dataset):
    return dataset['belongs_to_collection'].fillna(0).apply(lambda x: eval(x)[0]['name'] if x != 0 else 0)

# Check if movie belongs to collection
def is_belonging_to_collection(dataset):
    return dataset['belongs_to_collection'].fillna(0).apply(lambda x: 1 if x != 0 else 0)

# New feature collection name
train['collection_name'] = get_collection_name(train)
test['collection_name'] = get_collection_name(test)

# Change belongs_to_collection to bool
train['belongs_to_collection'] = is_belonging_to_collection(train)
test['belongs_to_collection'] = is_belonging_to_collection(test)


# **How many movies belongs to collection?**

# In[ ]:


sns.countplot(train['belongs_to_collection'])
train['belongs_to_collection'].value_counts()


# There are 604 movies that belongs to collection.

# **How many movies are in each collection?**

# In[ ]:


# train['collection_name'].value_counts()
train[train['belongs_to_collection'] == 1]['collection_name'].value_counts().head(10).plot(kind='bar')


# We can see top 10 biggest collections. Not suprisingly 'James Bond Collection' is in the first place.

# **Will movie make more money on box office if belongs to collection?**

# In[ ]:


sns.boxplot(x='belongs_to_collection', y='revenue', data=train)


# As we can see, if a movie belongs to collection, then it have greater chances of bigger box office revenue.

# **Which collection had the biggest revenue?**

# In[ ]:


(train[train['belongs_to_collection'] == 1].groupby('collection_name')['revenue']
 .agg('sum').sort_values(ascending=False).head(20).plot(kind='bar'))


# One could suspect that James Bond Collection will have number one spot. But when you closely look at the plot you will see that it did not reflect the true revenue, because some collections (like James Bond) have many movies and other (like Finding Nemo Collection) only couple. 
# 
# So which collection really had the biggest revenue?
# 
# Lets plot collections revenue with the selection of individual films.

# In[ ]:


g = sns.catplot(x='collection_name', y='revenue', data=train[train['belongs_to_collection'] == 1],
                order=train[train['belongs_to_collection'] == 1].groupby('collection_name')['revenue']
                .agg('sum').sort_values(ascending=False).head(20).index, aspect=3)
g.set_xticklabels(rotation=90)


# Now it is much more clear that some collections with only couple of movies had much more bigger revenue per movie.

# **Which collection made the most money per movie?**

# In[ ]:


(train[train['belongs_to_collection'] == 1].groupby('collection_name')['revenue']
 .agg('mean').sort_values(ascending=False).head(20).plot(kind='bar'))


# As I mention before some collections had big revenue because had multiple movies. In case of revenue per movie in collection it all looks differently. 
# 
# What can't be surprise that almost all movies are from last two decades. This is because of inflation (changes in prices of currency). I wonder where James Bond Collection would be if price of currency would be the same in all past years. 

# ## budget
# 
# Budget of a movie.

# **Which movies had the biggest budget?**

# In[ ]:


train[['title','budget']].sort_values(by='budget',ascending=False).head(10)


# **How the budget affects revenue?**

# In[ ]:


sns.lmplot(x='budget', y='revenue', data=train, height=5, aspect=2)


# The bigger the budget the higher the revenue (in general).

# **Did movies from collection had bigger budget?**

# In[ ]:


sns.boxplot(x='belongs_to_collection', y='budget', data=train)


# If a movie belongs to a collection, then it probably has bigger budget.
# 
# To really see if budget matters the most in box office revenue, we need to see what ratio between budget and revenue movies have. Also, I will create feature representing profit of the movie.

# In[ ]:


# Create new features for ratio and profit
train = train.assign(ratio = lambda df: df['revenue']/df['budget'])
train = train.assign(profit = lambda df: df['revenue']-df['budget'])


# In data there are movies with budget equal to 0, which probably means that budget is unknown. I won't be using those movies in analysis below.

# In[ ]:


# Get Top 10 movies with biggest revenue/budget ratio.
train[train['budget'] > 0][['title','budget','revenue','ratio','profit']].sort_values(by='ratio', ascending=False).head(10)


# Low budget for those movies probably indicate corrupted data. For the Top 5 movies data is definitely corrupted.
# 
# Let see how ratio looks like for movies with top budget.

# In[ ]:


# Get Top 10 movies with the biggest budget
train[['title','budget','revenue','ratio','profit']].sort_values(by='budget', ascending=False).head(10)


# Looking at this data we can see that some movies with top budget didn't bring big profit.

# **Which movie had the biggest profit?**

# In[ ]:


# Plot movies with the biggest profit
train.sort_values(by='profit', ascending=False).head(10).plot(x='title', y='profit', kind='barh')


# It is worth mention Jurassic Park on the list because it was released in 1993. Respect.

# **How many movies made profit?**

# In[ ]:


# Get movies that made profit
train['made_profit'] = train['profit'] > 0
sns.countplot(x='made_profit', data=train)
train['made_profit'].value_counts()


# Most movies made profit. If this wouldn't be the case movies wouldn't be popular and the number of movies released every year would be low.

# ## genres
# 
# What genre the film belongs to.
# 
# List of dictionaries (id of genre, genre name).

# In[ ]:


# Show top five rows
train['genres'].head(5)


# In[ ]:


# Parse json data
def parse_json(x):
    try: return json.loads(x.replace("'", '"'))
    except: return ''
    
# Parse genres
train['genres'] = train['genres'].apply(parse_json)


# **How many genres movies have?**

# In[ ]:


# Get number of genres for movies
train['genres'].apply(len).hist(bins=8)
print('Mean: ', train['genres'].apply(len).mean())


# Average movie has 2-3 genres, but there are some exaples of films with more categories. For example movies with 7 genres. 
# Also there are some movies with 0 genres which means that data is missing.

# In[ ]:


# Get the number of genres as a new feature
train['number_of_genres'] = train['genres'].apply(len)


# **How number of genres affects budget, revenue and profit?**

# In[ ]:


# Get budget, revenue, profit data for movies based on number of genres
(train.groupby('number_of_genres')[['budget','revenue','profit']].agg(['mean','median','count'])
 .sort_values(by=('profit','mean'), ascending=False))


# Movies with four genres had the highest revenue and profit. Movies with six genres had secend best result in revenue, but based on profit they are on third position. In case of movies with three genres their revenue is number three in chart, but they had a bigger profit, so their final position is number two.
# 
# The rest of the movies are in correct places based on both mean revenue and mean profit.
# 
# Conclusion: it is worth to get top 4 genres as new features.

# In[ ]:


# Get top 4 genres
def top_genres(genres):
    if len(genres) == 1:
        return pd.Series([genres[0]['name'], '', '', ''], 
                         index=['genre1', 'genre2', 'genre3', 'genre4'])
    if len(genres) == 2:
        return pd.Series([genres[0]['name'], genres[1]['name'], '', ''], 
                         index=['genre1', 'genre2', 'genre3', 'genre4'])
    if len(genres) == 3:
        return pd.Series([genres[0]['name'], genres[1]['name'], genres[2]['name'], ''], 
                         index=['genre1', 'genre2', 'genre3', 'genre4'])
    if len(genres) > 3:
        return pd.Series([genres[0]['name'], genres[1]['name'], genres[2]['name'], genres[3]['name']], 
                         index=['genre1', 'genre2', 'genre3', 'genre4'])
    return pd.Series(['','','',''], index=['genre1','genre2','genre3','genre4'])

train[['genre1', 'genre2', 'genre3', 'genre4']] = train['genres'].apply(top_genres)


# **Which genre is most popular?**

# In[ ]:


genres_df = pd.concat([train['genre1'].value_counts(), train['genre2'].value_counts(), 
                       train['genre3'].value_counts(), train['genre4'].value_counts()], 
                      axis=1, sort=False)
genres_df['sum'] = genres_df.sum(axis=1)

# Show genres data
genres_df


# **Which genre is the most popular from all fourth first genres of a movie (excluding missing genre)?**

# In[ ]:


genres_df[~genres_df.index.isin([''])]['sum'].plot(kind='pie', figsize=(9,9))


# **Which genre is the most popular based on its position (excluding missing genre)?**

# In[ ]:


genres_df[~genres_df.index.isin([''])][['genre1','genre2','genre3','genre4']].plot(kind='pie', subplots='True', 
                                                                                   figsize=(15,15), layout=(2,2), 
                                                                                   legend=False, 
                                                                                   title=['Genre 1','Genre 2',
                                                                                          'Genre 3','Genre 4'])


# **How first choice genre (genre1) affects revenue?**

# In[ ]:


sns.catplot(x='genre1', y='revenue', data=train, 
            order=train.groupby('genre1')['revenue'].mean().sort_values(ascending=False).index, 
            kind='box', height=4, aspect=5)


# Adventure movies have the highest mean revenue. Surprisingly drama is not in the top 5 (it is in the six spot from the end). Movies that are filled in some capacity with action or are family/animation have the highest revenue. It is worth mention that action movies are highly unstable in box office revenue (very high number of outliers). So if you want to make a movie that will have higher revenue, you should make adventure or animated movie.

# **How genre 1 affects budget?**

# In[ ]:


sns.catplot(x='genre1', y='budget', data=train, 
            order=train.groupby('genre1')['budget'].mean().sort_values(ascending=False).index, 
            kind='box', height=4, aspect=5)


# You need the biggest budget for adventure, science fiction and animated movies.
# 
# This plot shows the same conclusion with the one stated before, that bigger budget means higher revenue.
# 
# On the other hand, to make adventure/science fiction/(animated ?) movies you need a lot money for scenography to make them look realistic.

# **How genre 1 affects profit?**

# In[ ]:


sns.catplot(x='genre1', y='profit', data=train, 
            order=train.groupby('genre1')['profit'].mean().sort_values(ascending=False).index, 
            kind='box', height=4, aspect=5)


# Still on number one spot adventure movies. What is different is that family movies are in second place. I think it is not surprising because a lot of parents take kids to cinema. Even when parents do not like a movie or it got bad reviews, kids will still want to go to the cinema and in most cases that will happen.
# 
# As I metioned before, action movies are unstable in case of their profits (looking at this data you can't be sure if profits will be hight or low).

# ## homepage
# 
# URL address of movie homepage.

# **How many homepage addresses are filled?**

# In[ ]:


train['homepage'].notnull().value_counts().plot(kind='pie', autopct='%1.1f%%')


# There are only 31.5% of filled homepage addresses (68.5% is missing). Lets create new feature indicating if a movie has homepage or not. After that this feature (with homepage addresses) can be dropped.

# In[ ]:


# Create new feature describing if a movie have a homepage.
train['has_homepage'] = train['homepage'].notnull()
test['has_homepage'] = test['homepage'].notnull()


# In[ ]:


# Remove homepage feature
#train.drop(labels=['homepage'], axis=1, inplace=True)
#test.drop(labels=['homepage'], axis=1, inplace=True)


# **Do movies with homepage have higher revenue?**

# In[ ]:


sns.boxplot(x='has_homepage', y='revenue', data=train)


# Generally, movies with homepage have higher box office revenue.

# **Does a budget decide about having a homepage?**

# In[ ]:


sns.boxplot(x='has_homepage', y='budget', data=train)


# If a movie has a bigger budget, it means it should have a homepage. 

# **How budget and heving homepage affects profit?**

# In[ ]:


plt.figure(figsize=(15,5))
sns.lineplot(x='budget', y='profit', hue='has_homepage', data=train, style='has_homepage')


# Having a bigger budget and having a homepage increases chances of getting bigger profit.

# ## imdb_id
# 
# URL address of IMDB page related to a movie.
# 
# This feature could be used to gather more data or fill in missing ones but in this kernel I won't be doing this.
# So feature can be safely removed.

# In[ ]:


# Delete imdb_id feature
# train.drop(labels=['imdb_id'], axis=1, inplace=True)
# test.drop(labels=['imdb_id'], axis=1, inplace=True)


# ## original_language
# 
# Original language in which movie was released.

# **Which original language is the most used?**

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='original_language', data=train, order=train['original_language'].value_counts().index)


# No surprises, English is the most popular original language. Probably the reasons for that is that most movies are produced by studios in USA and their movies are the most popular worldwide.
# Another reason may be that they mainly collected English movies from TMDB to this dataset.

# **Which original language has the highest revenue?**

# In[ ]:


(train.groupby('original_language')['revenue'].mean().sort_values(ascending=False)
 .plot(kind='barh', figsize=(12,6), title='Which original language had the highest revenue?'))


# The highest revenue have movies with English original language but movies with Chinese language are only slighty behind. Chinese language is maybe not that surprising considering China population but the next languages on the list are (third place belongs to Turkish ?!). I suspected that Hindi will be higher on the list because of Bollywood and that Spanish language movies will have higher revenue.
# 
# The most important thing is that these data is not precise due to the small number of other movies than English.

# **How orginal language affects profit?**

# In[ ]:


(train.groupby('original_language')['profit'].mean().sort_values(ascending=False)
 .plot(kind='barh', figsize=(12,6), title='How original language affects profit?'))


# As before, we should not be sure of these results, given the small number of non-English films. But while working on what we have, we see that Chinese and Turkish films have slightly higher profits than English. Another funny thing is that movies with some languages do not bring profit, but losses. Conclusion: you should never produce movies in these languages.

# ## original_title
# 
# Movie orginal title.

# **What are the most popular words in orginal title?**

# In[ ]:


# Get all orginal titles
text = ' '.join(train['original_title'])

# Generate WordCloud from titles
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)

# Display wordcloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


# Get top 10 words in orginal title
stopwords = set(STOPWORDS)
words = text.lower().split(' ')
words = [word for word in words if (word not in stopwords and len(word) > 2)]
top_words = FreqDist(words).most_common(10)
top_words


# **Having in orginal title popular words increases revenue?**

# In[ ]:


# Get only words from top_words
words = []
for word, _ in top_words:
    words.append(word)

has_popular_words = train['original_title'].apply(lambda title: len([True for word in words if (word in title.lower())]))
(pd.concat([train['revenue'], has_popular_words], axis=1).groupby('original_title')['revenue'].mean()
 .plot(kind='bar', title='Popular words in title vs revenue'))


# Movies that contains top popular words in their titles have higher revenue. Having one word increases revenue, and having two words increases revenue even more.

# ## overview
# 
# Movie overwiew.

# **Which words in overviews are most frequent?**

# In[ ]:


text = ' '.join(train['overview'].astype(str))

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# FUTURE WORKS: try to rate overview in case of popularity/revenue of the movie. Which overviews have positive impact on revenue? How overview should look like to interest viewer?

# ## popularity
# 
# Popularity of the movie.

# **How popularity rating looks like?**

# In[ ]:


train['popularity'].describe()


#  - No data missing. 
#  - Average popularity rating is 8.463274.
#  - Standard deviation is 12.104.
#  - Max value looks wierd (anomaly like) based on mean +/- std.

# **Which movies have the highest popularity? Are there any outliers (anomalies)?**

# In[ ]:


train[['original_title','popularity']].sort_values(by='popularity', ascending=False).head(10)


# High popularity of some movies means that is buzz around them (people talking about them). We can see on the list movies that are released in last decade or are critically acclaimed. Conclusion: no outliers.

# **How budget affect popularity?**

# In[ ]:


sns.jointplot(x='budget', y='popularity', data=train, kind='reg')


# Looking at this data we can say that budget is not that important in case of movie popularity despite slightly positive slope. We need to take into account distribution of the data (low number of movies with high budget in contrast to high number of movies with low budget). Movies with the biggest budget are almost all under regression line.

# **Does a popularity of a movie affects box office revenue?**

# In[ ]:


sns.lmplot(x='popularity', y='revenue', data=train)


# Popularity of a movie has a positive impact on box office revenue.

# **And what about profit?**

# In[ ]:


sns.lmplot(x='popularity', y='profit', data=train)


# Popular movies tend to have bigger profits. I check this data because high revenue does not always mean high profit.

# ## poster_path
# 
# Path to image of the movie poster.
# 
# You could analyse posters to find out what features are important for movies with high revenue.
# How poster should look like to draw a viewer to watch a movie?
# 
# I will skip this feature and delete it.

# In[ ]:


# Drop poster_path feature
#train.drop(labels=['poster_path'], axis=1, inplace=True)
#test.drop(labels=['poster_path'], axis=1, inplace=True)


# ## production_companies
# 
# Companies that produce a movie.

# **How many companies take part in producing a single movie?**

# In[ ]:


train['production_companies'] = train['production_companies'].apply(parse_json)
train['production_companies'].apply(len).value_counts().sort_index().plot(kind='bar')


# In most cases movie is produce by one to three companies, sometimes by four. But in dataset we have movies that are produce by even 16 companies.

# **Which companies produce the most movies?**

# In[ ]:


# Get only list of names from list of directories
def parse_to_names(l):
    return [str(x['name']) for x in l]

companies = Counter(','.join(train['production_companies'].apply(parse_to_names)
                             .apply(lambda x: ','.join(x))).split(',')).most_common(21)
c = []
v = []
for x in companies[1:-1]:
    c.append(x[0])
    v.append(x[1])

df = pd.DataFrame(data={'company':c, 'count':v})

g = sns.barplot(x='company', y='count', data=df).set_xticklabels(rotation=90, labels=df['company'])


# Warner Bros produced the largest number of movies. Second place belongs to Universal Pictures.
# Only four companies have produced more than 100 movies.

# In[ ]:


# Get full list of unique values from selected json feature
def get_list(df, feature):
    l = []
    for row in df[feature].apply(parse_to_names):
        l.extend(x for x in row if x not in l)
    return l

# Get companies names
companies = get_list(train, 'production_companies')

# Show five top entries
companies[:5]


# In[ ]:


# One hot encode all elements from list l with all movies based on json feature
# Returns new dataframe with one hot encoded data
def list_one_hot_encoding(df, feature, l):
    new_df = df.copy()
    for x in l:
        new_df[x] = df[feature].apply(parse_to_names).apply(lambda y: 1 if x in y else 0)
    return new_df
            
# One hot encoding of prodction companies
companies_df = list_one_hot_encoding(train, 'production_companies', companies)


# In[ ]:


# Show top rows of encoded production companies
companies_df[companies[:5]].head()


# In[ ]:


# Create new dataframe with aggregated data for passed list 'l' based on 'df' dataframe.
# name - name of elements in list
# Return new dataframe(name, movies_produced, most_popular_genre1, genre_1_count, top_language, language_count,
#                      mean_popularity, mean_budget, mean_revenue, mean_profit)
def aggregate_data(df, l, name):
    aggregated_df = pd.DataFrame(columns=[name, 'movies_produced', 'most_popular_genre1', 'genre1_count',
                                          'top_language', 'language_count', 'mean_popularity', 'mean_budget', 
                                          'mean_revenue', 'mean_profit'])
    for x in l:
        # Group data by element from the list and get its group
        group_df = df.groupby(x).get_group(1)
        
        # Create new row data and append it to dataframe
        aggregated_df = aggregated_df.append({name: x,
                                              'movies_produced': group_df['id'].count(),
                                              'most_popular_genre1': group_df['genre1'].value_counts().index[0],
                                              'genre1_count': group_df['genre1'].value_counts()[0],
                                              'top_language': group_df['original_language'].value_counts().index[0],
                                              'language_count': group_df['original_language'].value_counts()[0],
                                              'mean_popularity': group_df['popularity'].mean(),
                                              'mean_budget': group_df['budget'].mean(),
                                              'mean_revenue': group_df['revenue'].mean(),
                                              'mean_profit': group_df['profit'].mean()},
                                             ignore_index=True)
        
    # Return dataframe with aggregate data
    return aggregated_df
        
# Get aggregate companies data
companies_df = aggregate_data(companies_df, companies, 'company')

# Show top rows of companies dataframe
companies_df.head()


# In[ ]:


# Plot with number of movies produced by each company was done earlier by different method.
# Which company produced the most movies?
#sns.barplot(x='company', y='movies_produced', data=companies_df.sort_values(by='movies_produced', ascending=False).head(20))


# **Based on companies that produced the most movies which genre was most popular?**

# In[ ]:


g = sns.catplot(x='company', y='genre1_count', hue='most_popular_genre1', 
                data=companies_df.sort_values(by='movies_produced', ascending=False).head(20), aspect=3)
g.set_xticklabels(rotation=90)


# The most popular genres in companies are Action, Comedy and Drama. One different value is for Adventure movies which Walt Disney Pictures produces the most.

# **Which company has the best mean movie popularity?**

# In[ ]:


plt.figure(figsize=(15,5))
g = (sns.barplot(x='company', y='mean_popularity', 
                data=companies_df.sort_values(by='mean_popularity', ascending=False).head(20))
     .set_xticklabels(rotation=90, labels=companies_df.sort_values(by='mean_popularity', ascending=False)
                      .head(20)['company']))


# Honestly, I do not know these companies except DC Entertainment and Marvel Studios. I assume that these companies have only a few movies and are therefore at the top of the list.

# **How many movies our top companies produced?**

# In[ ]:


companies_df.sort_values(by='mean_popularity', ascending=False)[['company','movies_produced']].head(20)


# The result coincides with what I predicted earlier. Almost all companies have produced only one movie.

# So once again. 
# 
# **Which company has the best mean movie popularity from companies that have produced minimum six movies?**

# In[ ]:


tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_popularity', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='company', y='mean_popularity', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])


# Now, in my opinion, this plot looks better. After removing companies with less then six movies, we get more recognizable companies. What is more, we get results that are more accurate, because the production of only one popular movie can be a coincidence. In general, top places are for companies associated with production of movies based on comic books. Including Syncopy, which is Chrstopher Nolan's company, which produced The Dark Knight Trilogy, Man of Steel and Justic League, which are from DC Universe.

# **Which company gives the biggest average budget to its movies?**
# 
# As before, I skip companies that have produced less than six movies.

# In[ ]:


tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_budget', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='company', y='mean_budget', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])


# On average, Jerry Bruckheimer Films gives biggest budget to its movies. Reason for that is that this company produced Pirates of the Caribbean movies which are the most expensive. In the leading places, we also have companies that produce comics adaptarions and Pixar which produces animations.

# **Which company had the biggest average revenue form its movies?**
# 
# As before, I skip companies that have produced less than six movies.

# In[ ]:


tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_revenue', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='company', y='mean_revenue', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])


# The biggest average revenue have companies that produce blockbuster movies (comics adaptations, Pirates of the Caribbean, The Lord of the Rings, Star Wars or animations).

# **Which company had the biggest average profit from its movies?**
# 
# As before, I skip companies that have produced less than six movies.

# In[ ]:


tmp = companies_df[companies_df['movies_produced'] > 5].sort_values(by='mean_profit', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='company', y='mean_profit', data=tmp).set_xticklabels(rotation=90, labels=tmp['company'])


# The results are generally the same as before. Maybe with one reservation which is that Jerry Bruckheimer Films are the most expensive and should have bigger profits.
# 
# Conclusion: it is not worth to overpay for the production of movies. Above a certain budget value, profit does not go hand in hand.

# In[ ]:


# Delete companies dataframe from memory
del companies_df


# And that will be all for production companies. It is worth mention that this analysis is not so accurate because some movies are produce by many companies which have different number of shares. Thay give diffrent amount of money for budget and then gets suitable percentage of revenues. And in this analysis I treated that if a company produces a movie, it givens the whole budget and has all revenues for itself. In other words, every movie data is included in whole in every company that produced it.

# ## production_countries
# 
# Countries where movie was made.
# 
# Dictionary with names of countries and their ISO codes.

# In[ ]:


# Show random values
train['production_countries'].sample(5)


# **In how many countries single movie is produced?**

# In[ ]:


train['production_countries'].apply(parse_json).apply(len).value_counts().plot(kind='bar')


# In most cases, movies are produced in one country. Sometimes in two, rarely in more then two (max 8). 0 production countries for a movie means missing value.

# In[ ]:


# Create new feature 'number of production countries'
train['production_countries_count'] = train['production_countries'].apply(parse_json).apply(len)
test['production_countries_count'] = test['production_countries'].apply(parse_json).apply(len)


# In[ ]:


# Parse countries to json
train['production_countries'] = train['production_countries'].apply(parse_json)


# In[ ]:


# Get list of unique production countries
countries = get_list(train, 'production_countries')

# Get countries data
countries_df = aggregate_data(list_one_hot_encoding(train, 'production_countries', countries), countries, 'country')

# Show top rows of countries dataframe
countries_df.head(5)


# **How many different production countries are there?**

# In[ ]:


len(countries)


# **Which production country has the greatest mean popularity?**
# 
# I am taking into consideration production countries with minimum six movies.

# In[ ]:


tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_popularity', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='country', y='mean_popularity', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])


# New Zealand has the greatest mean popularity, but it only produced 17 movies. For example United Kingdom (3rd place) produced 380 movies. It is worth to remember that because of different number of produced movies it is hard to say that movies produced in one country are more popular than movies produced in other country. However in case of New Zealand first place in the list could be true because in this country they shot movies like: The Lord of the Rings trilogy, Hobbit trilogy or King Kong.

# **In which production country movies had the biggest budget on average?**
# 
# 
# I am taking into consideration production countries with minimum six movies.

# In[ ]:


tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_budget', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='country', y='mean_budget', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])


# In prevoius plot I mention movies produced in New Zealand, so it is not surprise that they are still in number one place. This huge advantage over other countries, such as  the USA and the UK (with exception of the Czech Republic), results from the small number of produced movies. Strangely, the Czech Republic is in second place with small loss to New Zealand. Movies shot in Czech Republic with a bigger budget are: Casino Royale, The Chronicles of Narnia: Prince Caspian, Van Helsing and Mission: Impossible - Ghost Protocol.

# **In which production country movies had the biggest revenue on average?**
# 
# I am taking into consideration production countries with minimum six movies.

# In[ ]:


tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_revenue', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='country', y='mean_revenue', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])


# No surprises, the New Zealand in first place. With only couple of **such** films there couldn't be any other winner.

# **In which production country movies had the biggest profit on average?**
# 
# I am taking into consideration production countries with minimum six movies.

# In[ ]:


tmp = countries_df[countries_df['movies_produced'] > 5].sort_values(by='mean_profit', ascending=False).head(20)

plt.figure(figsize=(15,5))
g = sns.barplot(x='country', y='mean_profit', data=tmp).set_xticklabels(rotation=90, labels=tmp['country'])


# I will not mention the New Zealand and the Czech Republic this time. The United Arab Emirates are that high because in this country they also shot Mission: Impossible - Ghost Protocol (like in the Czech Republic). The UK is on fourth place in case of profit and was in eighth place in case of revenue.

# In[ ]:


# Delete coutries dataframe from memory
del countries_df


# ## release_date
# 
# Date when movie was released.
# 
# MM/DD/YY

# In[ ]:


# Top 'release_date' rows
train['release_date'].head(10)


# In[ ]:


# Convert correctly year from YY to YYYY format.
def convert_date(date):
    date = date.split('/')
    if int(date[2]) < 19:
        date[2] = '20' + date[2]
    else:
        date[2] = '19' + date[2]
    return '/'.join(date)


# In[ ]:


# Convert year (when you only use pandas to_datatime function, you will get wrong dates)
train['release_date'] = train['release_date'].apply(convert_date)

# Convert data to daytime64 type
train['release_date'] = pd.to_datetime(train['release_date'])

# Show converted data
train['release_date'].head(5)


# In[ ]:


# Split data to parts as new features
train['release_date_year'] = train['release_date'].map(lambda x: x.year)
train['release_date_month'] = train['release_date'].map(lambda x: x.month)
train['release_date_day'] = train['release_date'].map(lambda x: x.day)
train['release_date_day_of_week'] = train['release_date'].apply(lambda x: x.dayofweek)


# **How many movies were produced every year?**

# In[ ]:


tmp = train['release_date_year'].value_counts()

plt.figure(figsize=(15,5))
ax = sns.lineplot(x=tmp.index, y=tmp.values)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))


# A larger increase in produced movies began just before the 1980s, when movies began to gain in popularity. Since then, more movies have been made with each year.

# **Which month is the most popular when releasing a movie?**

# In[ ]:


sns.countplot(x='release_date_month', data=train)


# More movies released in September and October could be connected with Academy Awards. Later released movies are 'fresh' in memory so it have greater chance that academy nominates them. Personaly I thought that more movies would be released on summer when kids, teens have vacation and more time to go to a cinema.

# **Which day is the most popular when releasing a movie?**

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='release_date_day', data=train)


# On the 1st of each month more movies are released. Also around the middle of the month, more movies appear than usual.

# **Which day of the week is the most popular when releasing a movie?**
# 
# Day of week with Monday = 0 and Sunday = 6.

# In[ ]:


sns.countplot(x='release_date_day_of_week', data=train)


# Most movies are released on Friday when weekend starts. Sometimes movies are released on Thursday or Wednesday. Almost no movies are released on Monday, Tuesday, Saturday and Sunday.

# **How does the popularity of movies change with their release year?**

# In[ ]:


tmp = train.groupby(by='release_date_year')['popularity'].agg('mean')

plt.figure(figsize=(15,5))
ax = sns.lineplot(x=tmp.index, y=tmp.values)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))


# The popularity of the oldest movies is unstable. After 1950, the average popularity of movies begins to smooth around the value 9. Only new movies (released over the last year) have sky-high popularity (unlike other years).

# **Which month has more poppular movies released?**

# In[ ]:


tmp = train.groupby(by='release_date_month')['popularity'].agg('mean')

plt.figure(figsize=(15,5))
ax = sns.barplot(x=tmp.index, y=tmp.values)


# Popular movies are released more often in summer and in March.

# **Which day of month has more poppular movies released?**

# In[ ]:


tmp = train.groupby(by='release_date_day')['popularity'].agg('mean')

plt.figure(figsize=(15,5))
ax = sns.barplot(x=tmp.index, y=tmp.values)


# More popular movies were released at the end of the month (30th) or in the middle of the month (16th).

# **Which day of the week has more poppular movies released?**

# In[ ]:


tmp = train.groupby(by='release_date_day_of_week')['popularity'].agg('mean')

plt.figure(figsize=(15,5))
ax = sns.barplot(x=tmp.index, y=tmp.values)


# When most movies are released on Friday, more popular ones are released on Tuesday and Wednesday.

# **How has the average budget of movies changed over the years?**

# In[ ]:


tmp = train.groupby(by='release_date_year')['budget'].agg('mean')

plt.figure(figsize=(15,5))
ax = sns.lineplot(x=tmp.index, y=tmp.values)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))


# The budget over the years grow gradually until 1995, when in the blink of the eye it doubles. This is probably related to the greater popularity of movies and the production of more movies.

# **In which months do movies have an average higher budget?**

# In[ ]:


tmp = train.groupby(by='release_date_month')['budget'].agg('mean')

plt.figure(figsize=(15,5))
sns.barplot(x=tmp.index, y=tmp.values)


# There are no surprises. When the summer begins, it is the time for blockbusters productions to hit the cinema. And these movies generaly have the largest budget. At the end of the year, the budget is also bigger, which is probably also releated to blockbuster productions (for example the Hobbit movies that where released in December).

# **In what days do movies have a larger budget?**

# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(15,10))

tmp = train.groupby(by='release_date_day')['budget'].agg('mean')
sns.barplot(x=tmp.index, y=tmp.values, ax=ax[0])
ax[0].set_title('Average budget in case of day of releasing a movie')

tmp = train.groupby(by='release_date_day_of_week')['budget'].agg('mean')
sns.barplot(x=tmp.index, y=tmp.values, ax=ax[1])
ax[1].set_title('Average budget in case of releasing a movie in different day of the week')


# The day of the month is not important in case of budget size, but day of the week is. Movies have bigger budget on Tuesday and Wednesday which is probably connected with their popularity (popular movies are more often released on these days and what is more popular movies in most cases are blockbusters).

# **How has the average movie box office revenue and profits changed over the years?**

# In[ ]:


plt.figure(figsize=(15,5))

tmp = train.groupby(by='release_date_year')['revenue'].agg('mean')
ax = sns.lineplot(x=tmp.index, y=tmp.values, label='Revenue')

tmp = train.groupby(by='release_date_year')['profit'].agg('mean')
ax = sns.lineplot(x=tmp.index, y=tmp.values, label='Profit')

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))


# At the beginning revenue and profit were almost the same till the 1980 when they began to diverge. Since then revenue and profit difference is more or less constant.

# **How are average movie box office revenue and profit changes over months and days?**

# In[ ]:


fix, ax = plt.subplots(2, 1, figsize=(15,10))

sns.set_color_codes('pastel')
tmp = train.groupby(by='release_date_month')['revenue'].agg('mean')
sns.barplot(x=tmp.index, y=tmp.values, ax=ax[0], label='Revenue', color='r')

sns.set_color_codes('muted')
tmp = train.groupby(by='release_date_month')['profit'].agg('mean')
sns.barplot(x=tmp.index, y=tmp.values, ax=ax[0], label='Profit', color='r')

ax[0].legend(loc="upper left")
ax[0].set_title('Revenue and profit difference over the months')

sns.set_color_codes('pastel')
tmp = train.groupby(by='release_date_day')['revenue'].agg('mean')
sns.barplot(x=tmp.index, y=tmp.values, ax=ax[1], label='Revenue', color='b')

sns.set_color_codes('muted')
tmp = train.groupby(by='release_date_day')['profit'].agg('mean')
sns.barplot(x=tmp.index, y=tmp.values, ax=ax[1], label='Profit', color='b')

ax[1].legend(loc="upper left")
ax[1].set_title('Revenue and profit difference over the days')


# Movies have bigger revenue in summer and in the end of the year. They also have bigger revenue around  the middle of the month and on the last day of the month. As you can see the difference in revenue and profit is stable over the passing months and days.

# **How are average movie box office revenue and profit changes over months?**

# In[ ]:


plt.figure(figsize=(15,5))

sns.set_color_codes('pastel')
tmp = train.groupby(by='release_date_day_of_week')['revenue'].agg('mean')
ax = sns.barplot(x=tmp.index, y=tmp.values, label='Revenue', color='r')

sns.set_color_codes('muted')
tmp = train.groupby(by='release_date_day_of_week')['profit'].agg('mean')
ax = sns.barplot(x=tmp.index, y=tmp.values, label='Profit', color='r')

ax.legend(loc="upper left")


# Movies have higher revenue/profit at the beginning of the week, especially on Wednesday. As before, revenue and profit difference is stable.

# ## runtime
# 
# Runtime of a movie.

# **How runtime distribution looks like?**

# In[ ]:


train['runtime'].hist(bins=15)
print('Average runtime: ', train['runtime'].mean())
print('Std of runtime: ', train['runtime'].std())
print('Maximum runtime: ', train['runtime'].max())


# On average, movies last 108 minutes +/- 22 minutes. The longest movie is 338 minutes long.

# **How is the runtime related to budget, revenue and profit, if it is at all?**

# In[ ]:


plt.figure(figsize=(20, 6))
sns.lineplot(x='runtime', y='budget', data=train, label='budget')
sns.lineplot(x='runtime', y='revenue', data=train, label='revenue')
sns.lineplot(x='runtime', y='profit', data=train, label='profit').set_title('Impact of runtime to budget, revenue and profit')


# The runtime doesn't have huge impact on budget, revenue and profit. Maybe in case of movies longer than 190 minutes, because they have low revenue and profit. In addition, movies longer than 275 minuts have more costs than profits. Conclusion: you should not create movies that are too long because you will not have profit of them.

# **How does the runtime change for main (first) movie genre?**

# In[ ]:


plt.figure(figsize=(20,5))
tmp = train.groupby(by='genre1')['runtime'].mean()
sns.barplot(x=tmp.index, y=tmp.values)


# The first bar represents movies with missing genre data. The longest movies by genre are history and war. The shortest movies by genre are animation, tv movie and foreign.

# ## spoken_languages
# 
# Dictionary of languages used in a movie.

# In[ ]:


# Show top rows of spoken_languages
train['spoken_languages'].head()


# In[ ]:


# Parse spoken_languages to json
train['spoken_languages'] = train['spoken_languages'].apply(parse_json)


# **How many languages are used in the movie?**

# In[ ]:


train['spoken_languages'].apply(len).hist(bins=10)
print('Average: ', train['spoken_languages'].apply(len).mean())
print('Maximum languages used: ', train['spoken_languages'].apply(len).max())


# In most movies, the actors speak only in one language. Sometimes they speak in two languages. There is a movie that uses nine languages. This is the maximum number of languages in one movie. Movies without spoken language mean missing data.

# In[ ]:


# Create new feature representing number of languages spoken in a movie
train['spoken_languages_count'] = train['spoken_languages'].apply(len)


# **In which movies actors are speaking with more than five languages?**

# In[ ]:


train[train['spoken_languages_count'] >= 6][['original_title','release_date','spoken_languages_count']].sort_values(by='spoken_languages_count')


# There is even one movie from James Bond Collection (Die Another Day) on the list.

# **Does the number of languages used affect the budget?**

# In[ ]:


sns.barplot(x='spoken_languages_count', y='budget', data=train)


# If in a movie actors are speaking multiple languages, budget of that movie must be larger. In case of one, two, three or even four languages there is no that much of a difference in a budget, but in case of five languages budget grows exponentially. It is also worth mentioning that the grater the number of used languages, the greater deviations from the average budget.

# **Does the number of languages used affect revenue/profit?**

# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x='spoken_languages_count', y='revenue', data=train, label='revenue')
sns.lineplot(x='spoken_languages_count', y='profit', data=train, label='profit')


# Revenues and profits from movies using one to four languages are stable. Starting from five languages there are large deviations in revenues and profits (it is difficult to say whether a movie will have commercial success). Moreover, in case of six and more languages there is a possibility of a loss of money.

# **How many languages are used based on genre?**

# In[ ]:


train.groupby(by='genre1')['spoken_languages_count'].mean()


# Typically, one to two languages are used in each genre.

# In[ ]:


# Get list of languages
languages = get_list(train, 'spoken_languages')

# Get languages data
languages_df = aggregate_data(list_one_hot_encoding(train, 'spoken_languages', languages), languages, 'language')

# Show top rows of languages dataframe
languages_df.head(5)


# **Which movies are more popular based on the language used?**
# 
# Considering the languages used in at least 21 movies.

# In[ ]:


tmp = languages_df[languages_df['movies_produced'] > 20].sort_values(by='mean_popularity', ascending=False)

plt.figure(figsize=(20,5))
g = sns.barplot(x='language', y='mean_popularity', data=tmp).set_xticklabels(rotation=90, labels=tmp['language'].values)

tmp[['language','mean_popularity','movies_produced']]


# The most popular are movies in which Latin is used. The second place belongs to German and the third to Arabic. Movies that have used Russian are not that popular (fourth place from the end), taking into account that there are 152 movies in this language.

# **Which language has the average largest budget for a movie?**
# 
# Considering the languages used in at least 21 movies.

# In[ ]:


tmp = languages_df[languages_df['movies_produced'] > 20].sort_values(by='mean_budget', ascending=False)

plt.figure(figsize=(20,5))
g = sns.barplot(x='language', y='mean_budget', data=tmp).set_xticklabels(rotation=90, labels=tmp['language'].values)

tmp[['language','mean_budget','movies_produced']]


# Movies that use Latin have the largest budget. Next are: Thai, Arabic and Standard Mandarin.

# **Which language has the average biggest profit/revenue for a movie?**
# 
# Considering the languages used in at least 21 movies.

# In[ ]:


tmp = languages_df[languages_df['movies_produced'] > 20].sort_values(by='mean_profit', ascending=False)

plt.figure(figsize=(20,5))

sns.set_color_codes('pastel')
ax = sns.barplot(x='language', y='mean_revenue', data=tmp, label='mean revenue', color='b')

sns.set_color_codes('muted')
ax = sns.barplot(x='language', y='mean_profit', data=tmp, label='mean profit', color='b')

ax.legend(loc='upper right')

tmp[['language','movies_produced','mean_revenue','mean_profit']]


# The biggest profit have those movies that use Latin, Thai, Swedish, Standard Mandarin and Arabic. Swedish has lower revenue than Standard Mandarin and Arabic, but has better profit.

# ## status
# 
# Status of the movie.

# In[ ]:


# Print unique values for status
print('From train set: ', train['status'].unique().tolist())
print('From test set: ', test['status'].unique().tolist())


# **Which movies have status 'rumored' in train set?**

# In[ ]:


train[train['status'] == 'Rumored'][['original_title','release_date','revenue','status']]


# All these movies have been released. So we need to update their status.

# **Which movies have status 'rumored' in test set?**

# In[ ]:


test[test['status'] == 'Rumored'][['original_title','release_date','status']]


# All these movies have been released. So we need to update their status.

# **Which movies have status 'rumored' in test set?**

# In[ ]:


test[test['status'] == 'Post Production'][['original_title','release_date','status']]


# All 'post production' movies were also released, so we need to update their status.

# **Which movies have missing status in test set?**

# In[ ]:


test[test['status'].isnull()][['original_title','release_date','status']]


# As before, these movies were released.

# Conclusion: after examining unique values other than 'Released' in both train and test set, I can state that all movies should have status set to 'Released'. By doing this I will get feature with only one possible value and this feature will not give any information during training the model, so it should be removed.

# In[ ]:


# Delete status feature
#train = train.drop(labels=['status'], axis=1)
#test = test.drop(labels=['status'], axis=1)


# ## tagline
# 
# Movie tagline - a catchphrase or slogan (short, memorable description) that becomes identified with a movie. Used in ads.

# **How many taglines are missing?**

# In[ ]:


how_many_missing_values('tagline')


# In[ ]:


# Create new feature indicating if a movie has a tagline
train['has_tagline'] = train['tagline'].notnull()
test['has_tagline'] = test['tagline'].notnull()


# In[ ]:


# Print sample taglines
train[train['has_tagline']][['original_title','tagline']].sample(10)


# **What are the most popular tagline words?**

# In[ ]:


# Get all taglines
text = ' '.join(train[train['has_tagline']]['tagline'])

# Generate WordCloud from titles
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)

# Display wordcloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# One world, one life, one love.

# Future work: We could try to rate tagline in case of revenue. Which tagline has positive impact on box office revenue?
# Generally, the same as with overviews and movie posters.

# **Having a tagline affects revenue?**

# In[ ]:


sns.boxplot(x='has_tagline', y='revenue', data=train)


# Movies with tagline have bigger revenue. But proportion of movies with and without tagline is not equal, so you can't be 100% sure.

# ## title
# 
# International movie title.

# **Which movies have different original title and title?**

# In[ ]:


print('Movies with different titles: ', train[train['title'] != train['original_title']].shape[0])

# Print sample movies with different titles
train[train['title'] != train['original_title']][['original_title', 'title']].sample(10)


# There are 351 movies with a different original title and title. In most cases, non-English movies have different titles.

# In[ ]:


# Create new feature indicating if a movie has different original title and title
train['has_different_titles'] = train['original_title'] != train['title']
test['has_different_titles'] = test['original_title'] != test['title']


# **Having different titles affects revenue?**

# In[ ]:


sns.boxplot(x='has_different_titles', y='revenue', data=train)


# Movies with different titles have lower revenues. As with tagline, there are only few movies with different titles, so it is difficult to generalize to all cases. 

# ## Keywords
# 
# Keywords describing a movie.
# 
# Dictionary of IDs and names. 

# **How many movies do not have keywords?**

# In[ ]:


how_many_missing_values('Keywords')


# In[ ]:


# Parse to json
train['Keywords'] = train['Keywords'].apply(parse_json)


# **How many keywords does the movie have?**

# In[ ]:


print('Train set Keywords parameters:\n', train['Keywords'].apply(len).describe())
ax = train['Keywords'].apply(len).hist(bins=30)
ax = test['Keywords'].apply(parse_json).apply(len).hist(bins=30)

ax.legend(labels=['train','test'])


# On average, movies have almost seven keywords that describe them (+/- 6 keywords). One movie has 149 keywords, which is strange. Let's see which movie is it.

# **Which movie has 149 keywords?**

# In[ ]:


train[train['Keywords'].apply(len) > 100][['original_title','release_date','production_countries','Keywords']]


# It is some Hungarian production released in year 2000.

# **Which keywords are the most popular?**

# In[ ]:


# Get all keywords
text = ','.join(train['Keywords'].apply(parse_to_names).apply(lambda x: ','.join(x)))

# Generate WordCloud from titles
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)

# Display wordcloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# **Which keywords are the most popular?**
# 
# Represented as a list of 200 most common keywords.
# 
# I will skip first element because it represents missing keyword.

# In[ ]:


keywords = Counter(text.split(',')).most_common(201)[1:]

# Print keywords
keywords


# In[ ]:


# Get only keywords from most common keywords list
keywords = [x[0] for x in keywords]


# In[ ]:


# One hot encode most common keywords
keywords_df = list_one_hot_encoding(train, 'Keywords', keywords)


# **How each of selected 200 keywords affects revenue?**

# In[ ]:


fig, ax = plt.subplots(40,5, figsize=(20,160))
for x in range(40):
    for y in range(5):
        sns.boxplot(x=keywords[(x*5)+y], y='revenue', data=keywords_df, ax=ax[x][y])


# It is a lot of boxplots. Let's find meaningful keywords based on those plots.
# 
#  - women director: it is the most popular keyword and as we can see movies with women director have lower box office revenue.
#  - independent film: second most popular keyword. If a movie is independent film, there is no chance for a real profit.
#  - duringcreditsstinger: if a movie have scene after some credits, its revenue is way higher.
#  - murder: [skip it]
#  - based on novel: if a movie is based on novel, its revenue is higher.
#  - violence: if a movie is violent, it has a slightly higher revenue.
#  - sport: [skip it]
#  - biography: doesn't really matter.
#  - dystopia: if it is a dystopian movie, it will make more money.
#  - revenge: doesn't really matter. (There is only one movie about revange that had huge box office revenue.)
#  - aftercreditsstinger: if there is a scene after all credits, then movie will have a bigger revenue.
#  - sequel: if a movie is a sequel, it will make more money.
#  - frendship: doesn't really matter.
#  - suspense: doesn't really matter. (There is only one suspense movie that had huge box office revenue.)
#  - sex: [skip it]
#  - love: [skip it]
#  - police: doesn't really matter.
#  - teenager: doesn't really matter.
#  - nudity: [skip it]
#  - drug: [skip it]
#  - female nudity: [skip it]
#  - los angeles: if action of the movie takes place in Los Angeles, then the movie has greater chance for a little bigger revenue.
#  - high school: doesn't really matter.
#  - new york: if action of the movie takes place in New York, then movie has greater chance for a little bigger revenue.
#  - prison: doesn't really matter.
#  - musical: doesn't really matter.
#  - kidnapping: doesn't really matter.
#  - invastigation: doesn't really matter.
#  - family: doesn't really matter.
#  - father son relationship: doesn't really matter.
#  - 3d: if a movie is in 3D, it will definitely have a bigger revenue.
#  - wedding: doesn't really matter.
#  - detective: doesn't really matter.
#  - paris: doesn't really matter.
#  - based on comic: if movie is based on comic, it will definitely have a bigger revenue.
#  - robbery: doesn't really matter.
#  - brother brother relationship: doesn't really matter.
#  - prostitute: doesn't really matter.
#  - escape: if a movie have keyword 'escape', then it has small chance for a slightly bigger revenue.
#  - rape: doesn't really matter.
#  - alien: if a movie is about alien(s), it should have bigger revenue.
#  - based on true story: doesn't really matter. (I thought this keyword would make a difference.)
#  - london england: if action of a movie takes place in London, England, then movie should have a bigger revenue. It is probably connected with the fact that movies produced in UK have bigger revenue.
#  - death: doesn't really matter.
#  - superhero: if a movie is about superhero, it will definitely have a bigger revenue. In most cases connected with 'based on comic'.
#  - corruption: if a movie is about corruption, it should have a slightly bigger revenue.
#  - new york city: doesn't really matter. ('new york' and 'new york city' should have been merge into one keyword.)
#  - martial arts: if there are some martial arts in a movie, then movie has a very small chance for a slightly better revenue.
#  - dying and death: if a movie is about dying and death, it has small chance for a slightly bigger revenue. (probably most films should have this keyword.)
#  - serial killer: [skip it]
#  - suicide: [skip it]
#  - brother sister relationship: if a movie is about brother sister relationship, it has a small chance for a slightly bigger revenue.
#  - soldier: if a movie is about soldier, it has a small chance for a slightly bigger revenue.
#  - vampire: doesn't really matter.
#  - world war ii: doesn't really matter. (and should have!)
#  - hospital: doesn't really matter.
#  - future: if a movie is about future times, it has a chance for bigger revenue.
#  - best friend: doesn't really matter.
#  - remake: doesn't really matter.
#  - party: doesn't really matter.
#  - blood: [skip it]
#  - friends: doesn't really matter.
#  - male nudity: [skip it]
#  - adultery: doesn't really matter.
#  - lawyer: if a movie is about lawyer, it has a chance for bigger revenue.
#  - daughter: doesn't really matter. (maybe a little bit.)
#  - small town: doesn't really matter.
#  - war: doesn't really matter.
#  - jealousy: doesn't really matter.
#  - post-apocalyptic: if it is a post-apocalyptic movie, it has a slightly better chance for a small improvement with revenue.
#  - magic: if a movie is about magic, it has a slightly better chance for a slightly bigger revenue.
#  - explosion: if a movie contains explosions, it has a chance for bigger revenue.
#  - time travel: if a movie is about time travel, it has a slightly better chance for slightly bigger revenue.
#  - hitman: doesn't really matter.
#  - dog: doesn't really matter.
#  - money: doesn't matter at all.
#  - wife husband relationship: doesn't really matter.
#  - divorce: doesn't really matter.
#  - romantic comedy: if a movie is a romantic comedy, then it has a chance for bigger revenue.
#  - love triangle: doesn't really matter.
#  - hostage: doesn't really matter.
#  - journalist: doesn't really matter.
#  - monster: if a movie is about monster, it has a chance for bigger revenue.
#  - gay: doesn't matter.
#  - secret: doesn't really matter.
#  - helcopter: if there is a helicopter in a movie, then movie has a slight chance for bigger revenue.
#  - rescue: if a movie is about rescue, it should have a bigger revenue.
#  - assassin: if a movie is about assassin, it should have a bigger revenue.
#  - cia: if a movie is about cia, it should have bigger revenue.
#  - airplane: doesn't really matter.
#  - infidelity: [skip it]
#  - terrorist: if a movie is about terrorist, it has a chance for bigger revenue.
#  - fight: if a movie has any fights in it, it has a slightly higher chance for a slightly bigger revenue.
#  - gang: doesn't matter.
#  - fbi: if a movie is about fbi, it should have a bigger revenue.
#  - scientist: if a movie is about scientists, it should have a bigger revenue.
#  - zombie: if a movie is about zombie, it will definitely have a bigger revenue.
#  - ghost: doesn't matter.
#  - hotel: [skip it]
#  - college: doesn't matter.
#  - psychopath: if a movie is about psychopath, it has a slightly higher chance for a slightly bigger revenue.
#  - survival: if a movie is about surviving, it should have a bigger revenue.
#  - romance: doesn't really matter.
#  - chace: doesn't matter.
#  - marriage: doesn't matter. (maybe a little bit.)
#  - dark comedy: [skip it]
#  - competition: if a movie is about competition, it has a slightly higher chance for a slightly bigger revenue.
#  - relationship: doesn't really matter.
#  - alcohol: doesn't matter.
#  - spy: if it is a spy movie, it should have a bigger revenue.
#  - teacher: doesn't matter.
#  - flashback: doesn't matter.
#  - slasher: doesn't matter.
#  - found footage: if in a movie someone finds some footage, then movie has a slightly higher chance for a slightly bigger revenue.
#  - drug dealer: doesn't matter.
#  - shootout: doesn't matter. (maybe a little bit.)
#  - doctor: doesn't really matter.
#  - blackmail: doesn't matter.
#  - desert: [skip it]
#  - crime: doesn't really matter.
#  - sister sister relationship: if a movie is about sisters relationship, it has a chance for bigger revenue.
#  - dream: [skip it]
#  - motorcycle: if a movie has any/is about motorcycle, then it has a slightly higher chance for a slightly bigger revenue.
#  - gangster: doesn't matter.
#  - gore: [skip it]
#  - new love: doesn't matter.
#  - pregnancy: doesn't matter.
#  - extramarital affair: doesn't matter.
#  - criminal: doesn't matter.
#  - england: if action of a movie takes place in England (or is somehow connected to it), then movie should have a bigger revenue.
#  - conspiracy: if a movie is about conspiracy, it should have a bigger revenue.
#  - fbi agent: if a movie is about fbi agent, it has a chance for bigger revenue.
#  - reporter: doesn't matter. (maybe a little bit.)
#  - island: if action of a movie takes place on island, it has a chance for slightly bigger revenue.
#  - neo-noir: [skip it]
#  - faith: if a movie is about faith, it has a chance for slightly bigger revenue.
#  - amnesia: if in a movie someone has amnesia, it should have a bigger revenue.
#  - undercover: if a movie is about someone undercover, it should have a bigger revenue.
#  - baby: if a movie is about baby, it has a chance for bigger revenue.
#  - texas: [skip it]
#  - marvel comic: if a movie is based on Marvel comic, it will definitely have a big revenue.
#  - gun: doesn't matter.
#  - california: if action of a movie takes place in California, it should have a bigger revenue.
#  - assassination: if a movie is about assassination, it should have a bigger revenue.
#  - government: if a movie is about government, it has a chance for slightly bigger revenue.
#  - music: doesn't matter.
#  - military: if action of a movie is connected with military, then movie has a small chance for slightly higher revenue.
#  - based on play or musical: doesn't matter.
#  - coming of age: [skip it]
#  - army: if a movie is about army, it should have a bigger revenue.
#  - mountain: if action of a movie takes place in mountain, then movie should have a bigger revenue.
#  - torture: doesn't matter.
#  - road trip: doesn't matter.
#  - animation: if it is an animated movie, it will definitely have a bigger revenue.
#  - holiday: if action of a movie takes place on holiday, then movie has a chance for bigger revenue.
#  - comedy: if it is comedy movie, it has a chance for slightly higher revenue.
#  - politics: doesn't matter.
#  - robot: if a movie is about robot(s), it has a small chance for slightly higher revenue.
#  - baseball: doesn't matter.
#  - alcoholic: doesn't matter.
#  - forest: doesn't matter.
#  - writer: doesn't matter.
#  - priest: doesn't matter.
#  - police officer: if a movie is about police officer, it has a vary small chance for slightly higher revenue.
#  - parent child relationship: if a movie is about parent child relationship, it has a small chance for slightly higher revenue.
#  - male female relationship: doesn't matter.
#  - car crash: if a movie is about car crash, then it has a very small chance for slightly higher revenue.
#  - car chase: if in a movie is car chase, then movie has a chance for slightly higher revenue.
#  - japan: doesn't matter.
#  - jungle: if action of a movie takes place in jungle, then movie has a chance for slightly higher revenue.
#  - ship: doesn't matter.
#  - usa president: if a movie is about USA president, then it has a small chance for slightly higher revenue.
#  - fire: if movie contains fire, then it has a small chance for slightly higher revenue.
#  - australia: [skip it]
#  - secret identity: if a movie is about someone with secret identity, then it should have a bigger revenue.
#  - thief: doesn't matter.
#  - based on young adult novel: if a movie is based on young adult novel, then it has a chance for bigger revenue.
#  - christmas: if action of a movie takes place on Christmas, then movie has a small chance for bigger revenue.
#  - kiss: doesn't matter.
#  - satire: doesn't matter.
#  - marijuana: doesn't matter.
#  - washington d.c.: if action of a movie takes place in Washington D.C., then movie has a small chance for slightly bigger revenue.
#  - super powers: if a movie is about super powers, then it definitely have a bigger revenue.
#  - train: doesn't matter.
#  - killer: doesn't matter.
#  - training: doesn't matter.
#  - snow: if there is snow in a movie, then movie has a small chance for slightly bigger revenue.
#  - mutant: if a movie is about mutant, then it will definitely have a bigger revenue.
#  - restaurant: if a movie is about restaurant, then it has a small chance for slightly bigger revenue.
#  - sheriff: [skip it]
#  - artificial intelligence: unfortunately, it doesn't matter.
#  - movster: doesn't matter.
#  - hero: if a movie is about some hero, then it will definitely have a bigger revenue.
#  - dancting: if a movie is about dancing, than it has a chance for bigger revenue.
#  - world war i: [skip it]
#  - wilderness: doesn't matter.
#  - student: [skip it]
#  - space: if action of a movie takes place in space, then movie has a chance for bigger revenue.
#  - airport: if action of a movie takes place in airport, then movie has a small chance for slightly higher revenue.

# In[ ]:


# From above list let's select keywords that I thought have some impact on revenue.
keywords = ['woman director',
            'independent film',
            'duringcreditsstinger',
            'based on novel',
            'violence',
            'dystopia',
            'aftercreditsstinger',
            'sequel',
            'los angeles',
            'new york',
            '3d',
            'based on comic',
            'escape',
            'alien',
            'london england',
            'superhero',
            'corruption',
            'martial arts',
            'dying and death',
            'brother sister relationship',
            'soldier',
            'future',
            'lawyer',
            'post-apocalyptic',
            'magic',
            'explosion',
            'time travel',
            'romantic comedy',
            'monster',
            'helicopter',
            'rescue',
            'assassin',
            'cia',
            'terrorist',
            'fight',
            'fbi',
            'scientist',
            'zombie',
            'psychopath',
            'survival',
            'competition',
            'spy',
            'found footage',
            'sister sister relationship',
            'motorcycle',
            'england',
            'conspiracy',
            'fbi agent',
            'island',
            'faith',
            'amnesia',
            'undercover',
            'baby',
            'marvel comic',
            'california',
            'assassination',
            'government',
            'military',
            'army',
            'mountain',
            'animation',
            'holiday',
            'comedy',
            'robot',
            'police officer',
            'parent child relationship',
            'car crash',
            'car chase',
            'jungle',
            'usa president',
            'fire',
            'secret identity',
            'based on young adult novel',
            'christmas',
            'washington d.c.',
            'super powers',
            'snow',
            'mutant',
            'restaurant',
            'hero',
            'dancing',
            'space',
            'airport'
           ]

# List is longer than I assume.


# In[ ]:


# Create from selected keywords new features
train = list_one_hot_encoding(train, 'Keywords', keywords)


# In[ ]:


# Remove keywords_df from memory
del keywords_df


# ## cast
# 
# Cast of a movie.
# 
# Dictionary of: 
#  - cast_id:
#  - character: character name.
#  - credit_id:
#  - gender: gender of character/actor. (1 - female, 2 - men, 0 - unknown (male or female))
#  - id: id of an actor.
#  - name: name of an actor.
#  - order: importance of a character (from most to less important).
#  - profile_path: path to photo of an actor.
#  

# In[ ]:


# Parse cast data.
train['cast'] = train['cast'].replace('\'','\"').apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})


# **How many people are in the cast?**

# In[ ]:


print(train['cast'].apply(len).describe())
train['cast'].apply(len).hist(bins=15)


# On average, there are 20 actors in the cast (+/- 16). The maximum number of actors in the cast is 156 people.

# **Which movies have over 80 actors in the cast?**

# In[ ]:


(train[train['cast'].apply(len) >= 80][['original_title','release_date','revenue','cast']]
 .assign(cast_count=lambda x: x['cast'].apply(len)).sort_values(by='cast_count'))


# There are 40 movies with 80 or more actors. Each cast includes all actors who played in the movie, no matter how big or small their role were.

# **What are the most popular words in characters names?**

# In[ ]:


# Get all character names
text = ','.join(train['cast'].apply(lambda x: [actor['character'] for actor in x]).apply(lambda x: ','.join(x)))

# Generate WordCloud from characters names
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)

# Display wordcloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# **What are the top 25 most popular character names?**

# In[ ]:


Counter(text.split(',')).most_common(25)


# The most popular character name is for 'actors' that don't even get a character name (probably extras). Next places are for Himself/Herself. Generally, most character names from the list are names describing job position. Probably they are characters that play meaningless roles in the movie. 
# 
# The most popular name on the list is Jack.

# In[ ]:


# Count gender values form 'df' dataframe from 'feature' column.
# Returns Series with gender value counts.
def gender_counts(df, feature):
    gender_count = pd.Series(data=[0,0,0], index=[0,1,2])
    for row in df[feature]:
        for person in row:
            gender_count.loc[person['gender']] += 1
    return gender_count


# **Which gender is the most popular among actors?**

# In[ ]:


tmp = gender_counts(train, 'cast')
print(tmp)

sns.barplot(x=tmp.index, y=tmp.values)


# The largest group represents actors (2 - male). The actresses are half as much (1 - female). Group describe as 0 represents unknown gender (male or female) which is probably connected with people that play extras in mostly one movie.

# In[ ]:


# Create new dataframe with aggregated actors data.
# Returns new dataframe with columns:
#  [id, name, movies_played, mean_gender, mean_order, start_year, end_year, 
#   mean_movie_popularity, mean_movie_budget, mean_movie_revenue, mean_movie_profit]
def aggregate_actors_data(df):
    
    #Create new dataframe for actors data
    actors_df = pd.DataFrame(columns=['id','name','movies_played','mean_gender','mean_order',
                                      'start_year','end_year','mean_movie_popularity','mean_movie_budget',
                                      'mean_movie_revenue','mean_movie_profit'])
    
    # Iterate through movies
    for index, movie in df.iterrows():
        
        #Iterate through cast list
        for actor in movie['cast']:
            
            #Chack if actor is already in dataframe
            if actors_df['id'].isin([actor['id']]).sum() == 0:
                
                # If not create new entry for actor
                actors_df = actors_df.append({'id': actor['id'],
                                              'name': actor['name'],
                                              'movies_played': 1,
                                              'mean_gender': actor['gender'],
                                              'mean_order': actor['order'],
                                              'start_year': movie['release_date_year'],
                                              'end_year': movie['release_date_year'],
                                              'mean_movie_popularity': movie['popularity'],
                                              'mean_movie_budget': movie['budget'],
                                              'mean_movie_revenue': movie['revenue'],
                                              'mean_movie_profit': movie['profit']
                                             },
                                             ignore_index=True)
            else:
                # If exists, then update values
                actors_df.loc[actors_df['id'] == actor['id'], 'movies_played'] += 1
                actors_df.loc[actors_df['id'] == actor['id'], 'mean_gender'] += actor['gender']
                actors_df.loc[actors_df['id'] == actor['id'], 'mean_order'] += actor['order']
                actors_df.loc[actors_df['id'] == actor['id'], 'start_year'] = min(actors_df.loc[actors_df['id'] == actor['id'], 'start_year'].values[0], 
                                                                                  movie['release_date_year'])
                actors_df.loc[actors_df['id'] == actor['id'], 'end_year'] = max(actors_df.loc[actors_df['id'] == actor['id'], 'end_year'].values[0], 
                                                                                  movie['release_date_year'])
                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_popularity'] += movie['popularity']
                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_budget'] += movie['budget']
                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_revenue'] += movie['revenue']
                actors_df.loc[actors_df['id'] == actor['id'], 'mean_movie_profit'] += movie['profit']

    # Get mean values
    actors_df['mean_gender'] = actors_df['mean_gender'] / actors_df['movies_played']
    actors_df['mean_order'] = actors_df['mean_order'] / actors_df['movies_played']
    actors_df['mean_movie_popularity'] = actors_df['mean_movie_popularity'] / actors_df['movies_played']
    actors_df['mean_movie_budget'] = actors_df['mean_movie_budget'] / actors_df['movies_played']
    actors_df['mean_movie_revenue'] = actors_df['mean_movie_revenue'] / actors_df['movies_played']
    actors_df['mean_movie_profit'] = actors_df['mean_movie_profit'] / actors_df['movies_played']
    
    # Change data types to numeric ones.
    actors_df['id'] = actors_df['id'].astype('int64')
    actors_df['movies_played'] = actors_df['movies_played'].astype('int64')
    actors_df['mean_gender'] = actors_df['mean_gender'].astype('float64')
    actors_df['mean_order'] = actors_df['mean_order'].astype('float64')
    actors_df['mean_movie_popularity'] = actors_df['mean_movie_popularity'].astype('float64')
    actors_df['mean_movie_budget'] = actors_df['mean_movie_budget'].astype('float64')
    actors_df['mean_movie_revenue'] = actors_df['mean_movie_revenue'].astype('float64')
    actors_df['mean_movie_profit'] = actors_df['mean_movie_profit'].astype('float64')
    
    # Return new dataframe
    return actors_df


# In[ ]:


# Get aggregated actors data
#actors_df = aggregate_actors_data(train)

# Save actors dataframe to file
#actors_df.to_csv('../actors.csv', index=False)

# Load actors dataframe from file
actors_df = pd.read_csv('../input/tmdb-box-office-prediction-cast-crew/actors.csv')

# Show sample actors data
actors_df.sample(10)


# **Which names are the most popular among actors names?**

# In[ ]:


# Get all actors names
text = ','.join(actors_df['name'])

# Generate WordCloud from actors names
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=600).generate(text)

# Display wordcloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# **How many actors are in dataframe?**

# In[ ]:


print('Actors number: ', actors_df.shape[0])


# **How many actors played in only one movie?**

# In[ ]:


print('Number of actors that played in only one movie: ', actors_df[actors_df['movies_played'] == 1].shape[0])


# In[ ]:


# Lets drop all actors that played in only one movie.
# (Those actors have no meaningful informations. Would only obscure the real picture)
actors_df = actors_df.drop(actors_df[actors_df['movies_played'] == 1].index, axis=0)


# **Which actor/actress has played in largest number of movies?**

# In[ ]:


g = sns.barplot(y='name', x='movies_played', hue='mean_gender', 
                data=actors_df.sort_values(by='movies_played', ascending=False).head(20), palette='rocket')


# Robert De Niro and Samuel L. Jackson played in the largest number of movies, which is 30. It is worth mentioning that on the top 25 list there are only two actresses: Susan Sarandon (4th place with 25 movies) and Sigourney Weaver (20th place with 21 movies). The rest are male actors.

# **How many people are each gender?**

# In[ ]:


actors_df['mean_gender'].value_counts()


# Actresses (1) are half as numerous as actors (2). Also, there is a third group (0) that represents unknown gender.

# **Who played in more movies, actors or actresses?**

# In[ ]:


actors_df.groupby(by='mean_gender')['movies_played'].mean().plot(kind='pie', legend=False, autopct='%1.0f%%', 
                                                                 labels=['unknown','female','male'])


# Slight advantage in favour of actors. But we need to remember that in unknown group may be more actresses and the plot would look differently and about that actresses are less numerous than actors.

# **Who plays more important roles on average, acotrs or actresses?**

# In[ ]:


sns.barplot(x='mean_gender', y='mean_order', data=actors_df)


# Actresses slightly often play more important roles than actors. But we need to remember how numerous each group is. In this plot we can see that unknown group (0) genrerally should be treated as extras. And this is the confirmation of what I already said earlier about this group.

# **Which actor most often played the most important roles?**
# 
# Excluding all actors that played in less than 8 productions.

# In[ ]:


tmp = actors_df[actors_df['movies_played'] > 7].sort_values(by='mean_order').head(25)

plt.figure(figsize=(10,10))
sns.barplot(x='mean_order', y='name', data=tmp)


# John Travolta, Clint Eastwood, Michael Douglas almost allways play the main characters.

# **Which actor is/was the longest professionally active?**
# 
# Excluding all actors that played in less than 8 productions.

# In[ ]:


# Calculate years of professional activity
actors_df = actors_df.assign(years_active=lambda x: x['end_year'] - x['start_year'])

plt.figure(figsize=(10,10))
tmp = actors_df[actors_df['movies_played'] > 7].sort_values(by='years_active', ascending=False).head(25)

sns.barplot(x='years_active', y='name', data=tmp)


# Based on the plot Paul Newman was the longest professionally active actor (63 years difference between his first movie and last). But when we take a closer look, his last movie in which he was credited was released in 2017 and he died in 2008. So this movie contains his archival footage not his live work. Taking this into account, the longest professionally active actor or should I say actress, should be Lois Smith. She is still living (she's 88 years old) and still playing in the movies. From our data she has been professionally active for 61 years.

# **Which movies are the most popular on average beased on actor that played in it?**

# In[ ]:


sns.barplot(x='mean_movie_popularity', y='name', data=actors_df.sort_values(by='mean_movie_popularity', 
                                                                            ascending=False).head(25))


# This is probably not what I suspected. I need to select only actors that played in more movies than only just a couple.

# To be sure let's check in how many movies all of above actors played.

# In[ ]:


actors_df.sort_values(by='mean_movie_popularity',ascending=False).head(25)[['name','movies_played']]


# All of them played only in 2 movies.

# **Which movies are the most popular on average beased on actor that played in it?**
# 
# Excluding all actors that played in less than 10 productions.

# In[ ]:


tmp = actors_df[actors_df['movies_played'] >= 10].sort_values(by='mean_movie_popularity', ascending=False).head(40)
plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_popularity', y='name', data=tmp)


# Emma Thompson is in the lead. Second and third places belongs to actors that played in The Lord of the Rings Trilogy. Andy Serkis is before Ian McKellen probably because he also played in Planet of the Apes movies. Jon Hamm is surprise to me in his fourth position. It is probably connected with popularity of Baby Driver. Jamie Foxx also played in this movie and he is also in high place. Between them is Vin Disel with his popularity based on Fast & Furious series. There are also numerous actors that played in superheros movies which was obvious.

# **Which actor played in movies with the biggest revenue and the biggest budget?**
# 
# Excluding all actors that played in less than nine movies.

# In[ ]:


tmp = actors_df[actors_df['movies_played'] > 8].sort_values(by='mean_movie_revenue', ascending=False).head(50)

traces = []
for _, row in tmp.iterrows():
    traces.append(go.Scatter(
        x=[row['mean_movie_budget']],
        y=[row['mean_movie_revenue']],
        name=row['name'],
        mode='markers'
    ))

layout = go.Layout(
    title = go.layout.Title(text='Actors based on mean movies revenue and budget'),
    xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(text='Mean movie budget')), 
    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text='Mean movie revenue'))
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig, show_link=False)

# Seaborn version (no interactive)
#plt.figure(figsize=(16,10))
#sns.stripplot(x='mean_movie_budget', y='mean_movie_revenue', hue='name', data=tmp)


# The unquestionable king is Andy Serkis. Movies in which he plays have the biggest budget and the biggest revenue. Just to remind you, he played among others Gollum in The Lord of the Rings trilogy and in Hobbit trilogy, Caesar in Planet of the Apes trilogy or in Avengers: Age of Ultron. Movies with Andy Serkis have around 50M dollars advantage over movies with Orlando Bloom who is in second place.

# **Which actor played in movies with the largest profits?**
# 
# Excluding all actors that played in less than 8 productions.

# In[ ]:


tmp = actors_df[actors_df['movies_played'] > 7].sort_values(by='mean_movie_profit', ascending=False).head(25)

plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_profit', y='name', data=tmp)


# Stan Lee with his cameo appearances in Marvel movies based on his comics is in first place with huge advantage. Second place belongs to Jess Harnell who played in Transformers movies. In third is Tyrese Gibson who played in Fast & Furious movies.

# In[ ]:


# Get IDs for top 100 actors that have biggest mean movie revenue.
# Excluding all actros that played in less than nine movies.
ids = (actors_df[actors_df['movies_played'] > 8]
       .sort_values(by='mean_movie_revenue', ascending=False)['id'].head(100).values.tolist())


# In[ ]:


# Check if person worked in a movie.
def is_person_in_cast(cast, ids):
    for person in cast:
        if person['id'] in ids:
            return True
    return False


# In[ ]:


# Create new feature indicating if in a movie played any of actors from the above list.
train['has_top_actor'] = train['cast'].apply(lambda x: is_person_in_cast(x, ids))


# In[ ]:


# Get top 6 actors
def top_actors(cast):
    actors = pd.Series(data=['', '', '', '', '', ''],
                       index=['actor0', 'actor1', 'actor2', 'actor3', 'actor4', 'actor5'])
    for n in range(min(6, len(cast))):
        actors.loc['actor{}'.format(n)] = cast[n]['name']
    return actors


# In[ ]:


# Create new feature for top 6 actors in the cast
train[['actor0', 'actor1', 'actor2', 'actor3', 'actor4', 'actor5']] = train['cast'].apply(top_actors)


# In[ ]:


# Remove actors dataframe to free memory
del actors_df


# **Do movies with top actors really have bigger revenue?**

# In[ ]:


tmp = train.groupby(by='has_top_actor')['revenue'].mean()
sns.barplot(x=tmp.index, y=tmp.values)

print('How numerous each group are:\n', train['has_top_actor'].value_counts())


# It is just a confirmation of what I said earlier. If actor from the list plays in the movie, then movie have bigger revenue. 

# ## crew
# 
# Crew responsible for producing a movie.
# 
# Dictionary of:
#  - credit_id:
#  - department: department to which job belongs to
#  - gender: gender of crew member
#  - id: id of crew member
#  - job: job name
#  - name: name of crew member
#  - profile_path: path to image of crew member

# In[ ]:


# Parse crew data
train['crew'] = train['crew'].replace('\'','\"').apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})


# **How many people are on average in the crew?**

# In[ ]:


# Show histogram
train['cast'].apply(len).hist()

# Describe data
train['cast'].apply(len).describe()


# On average there are 20 people (+/- 16) in movie crew.

# **Which gender is the most popular in the crew?**

# In[ ]:


gender = gender_counts(train, 'crew')

print('Gender counts:\n', gender)

sns.barplot(x=gender.index, y=gender.values)


# The most numerous gender is the unknown gender. They are probably not decision making crew members. Among the remaining people, men prevail. We could even say that crew mostly consists of men. I would assume that women in the crew are responsibe for more creative jobs, such as directing, writing, casting.

# In[ ]:


# Return list of departments
def get_departments(df):
    departments = set()
    for crew in df:
        for person in crew:
            departments.add(person['department'])
    return list(departments)


# In[ ]:


# Return list of jobs by departments
def get_jobs_by_department(df, departments):
    jobs = {department:set() for department in departments}
    for crew in df:
        for person in crew:
            jobs[person['department']].add(person['job'])
    return jobs


# **What departments exist and what positions do they have?**

# In[ ]:


# Get list of departments
departments = get_departments(train['crew'])

# Get list of jobs by departments
jobs = get_jobs_by_department(train['crew'], departments)

# Print jobs
jobs


# **How many different jobs are in each department?**

# In[ ]:


['{}: {}'.format(x, len(jobs[x])) for x in jobs]


# It will be wise to select only jobs that matter in production (jobs that have creative impact on the movie), for example 'Director'.

# In[ ]:


# I manually selected below jobs (by my intuition)
selected_jobs =  ['Casting', 'Original Music Composer', 'Screenplay', 'Director', 'Director of Photography', 
                  'Editor', 'Costume Design', 'Producer', 'Executive Producer']


# From this point forward all analysis for crew will be restricted only to selected jobs.

# In[ ]:


# Get list of person jobs from crew data
def get_person_jobs(df, person_id):
    person_jobs = set()
    for crew in df:
        for person in crew:
            if person['id'] == person_id:
                person_jobs.add(person['job'])
    return list(person_jobs)


# In[ ]:


# Aggregate crew data and return new dataframe with features:
# [id, name, gender, job, movies_created, start_year, end_year, 
#  mean_movie_popularity, mean_movie_budget, mean_movie_revenue, mean_movie_profit]
def aggregate_crew_data(df, jobs):
    
    # Create new dataframe
    crew_df = pd.DataFrame(columns=['id', 'name', 'gender', 'job', 'movies_created', 'start_year', 'end_year',
                                    'mean_movie_popularity', 'mean_movie_budget','mean_movie_revenue',
                                    'mean_movie_profit'])
    
    # Iterate through movies
    for index, movie in df.iterrows():
        
        # Iterate through crew members
        for person in movie['crew']:
            
            # Check if person works in one of selected jobs
            if person['job'] in jobs:
            
                # Chack if person is already in dataframe
                if crew_df['id'].isin([person['id']]).sum() == 0:

                    # If not, then add new entry for crew member
                    crew_df = crew_df.append({'id': person['id'],
                                              'name': person['name'],
                                              'gender': person['gender'],
                                              'job': get_person_jobs(df['crew'], person['id']),
                                              'movies_created': 1,
                                              'start_year': movie['release_date_year'],
                                              'end_year': movie['release_date_year'],
                                              'mean_movie_popularity': movie['popularity'],
                                              'mean_movie_budget': movie['budget'],
                                              'mean_movie_revenue': movie['revenue'],
                                              'mean_movie_profit': movie['profit']
                                             },
                                             ignore_index=True)

                else:

                    # If exists, then update values
                    crew_df.loc[crew_df['id'] == person['id'], 'movies_created'] += 1
                    crew_df.loc[crew_df['id'] == person['id'], 'start_year'] = min(crew_df.loc[crew_df['id'] == person['id'], 'start_year'].values[0],
                                                                                   movie['release_date_year'])
                    crew_df.loc[crew_df['id'] == person['id'], 'end_year'] = max(crew_df.loc[crew_df['id'] == person['id'], 'end_year'].values[0],
                                                                                   movie['release_date_year'])
                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_popularity'] += movie['popularity']
                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_budget'] += movie['budget']
                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_revenue'] += movie['revenue']
                    crew_df.loc[crew_df['id'] == person['id'], 'mean_movie_profit'] += movie['profit']
            
    # Get mean values
    crew_df['mean_movie_popularity'] = crew_df['mean_movie_popularity'] / crew_df['movies_created']
    crew_df['mean_movie_budget'] = crew_df['mean_movie_budget'] / crew_df['movies_created']
    crew_df['mean_movie_revenue'] = crew_df['mean_movie_revenue'] / crew_df['movies_created']
    crew_df['mean_movie_profit'] = crew_df['mean_movie_profit'] / crew_df['movies_created']
    
    # Change data types to numeric ones
    crew_df['id'] = crew_df['id'].astype('int64')
    crew_df['gender'] = crew_df['gender'].astype('int64')
    crew_df['movies_created'] = crew_df['movies_created'].astype('int64')
    crew_df['mean_movie_popularity'] = crew_df['mean_movie_popularity'].astype('float64')
    crew_df['mean_movie_budget'] = crew_df['mean_movie_budget'].astype('float64')
    crew_df['mean_movie_revenue'] = crew_df['mean_movie_revenue'].astype('float64')
    crew_df['mean_movie_profit'] = crew_df['mean_movie_profit'].astype('float64')
    
    # Return crew dataframe
    return crew_df


# In[ ]:


# Get aggregated crew data
#crew_df = aggregate_crew_data(train, selected_jobs)

# Save crew dataframe to file
#crew_df.to_csv('../crew.csv', index=False)

# Load crew dataframe from file
crew_df = pd.read_csv('../input/tmdb-box-office-prediction-cast-crew/crew.csv')

# Show sample crew data
crew_df.sample(10)


# In[ ]:


# Remove all crew members with only one movie created and with popularity less than 9.
crew_df = crew_df[~((crew_df['movies_created'] == 1) & (crew_df['mean_movie_popularity'] < 9))]


# **Who worked in the largest number of movies?**

# In[ ]:


tmp = crew_df[['name','job','movies_created']].sort_values('movies_created', ascending=False).head(25)
tmp['job'] = tmp['job'].apply(lambda x: x[0])

fig, ax = plt.subplots(1,2, figsize=(16,10), gridspec_kw = {'width_ratios':[1, 2]})
sns.barplot(x='movies_created', y='name', data=tmp, ax=ax[0])

ax[1].axis('off')
table = ax[1].table(cellText=tmp.values, 
                  rowLabels=tmp.index, 
                  colLabels=tmp.columns, 
                  bbox=[0,0,1,1])
table.auto_set_font_size(False)
table.set_fontsize(14)


# Avy Kaufman is the leader with 50 movies in which she was responsible for casting. 
# 
# Type of job shown in the table corresponds to the first value from the list of jobs that given person ever did. It some cases, position is not correspond to job that given person is most famous for, for example Robert Rodriguez - Steadicam Operator (director would be more appropriate).

# **Which crew member works the longest in business?**

# In[ ]:


crew_df = crew_df.assign(years_active=lambda x: x['end_year'] - x['start_year'])

tmp = crew_df[['name','start_year','end_year','years_active']].sort_values(by='years_active', ascending=False).head(25)

fig, ax = plt.subplots(1,2, figsize=(16,10), gridspec_kw = {'width_ratios':[1, 2]})
sns.barplot(x='years_active', y='name', data=tmp, ax=ax[0])

ax[1].axis('off')
table = ax[1].table(cellText=tmp.values, 
                  rowLabels=tmp.index, 
                  colLabels=tmp.columns, 
                  bbox=[0,0,1,1])
table.auto_set_font_size(False)
table.set_fontsize(14)


# First two names on the list are wrong. Raoul Walsh died in 1980 so he couldn't have work till 2015. In reality he worked 51 years as a director. Philip Glass was born in 1937 and started working as a composer in 1964 (he is still professionally active). So he shouldn't be able to start working in 1931. So proper leader should be Ennio Morricone  with 49 years of working as a composer.
# 
# On the list are three female crew members. The highest place among them belongs to Dede Allen with 41 years. She worked as an editor.

# **Which crew member has the highest mean popularity of movies in which he/she worked?**
# 
# Excluding all crew that worked on less than 8 movies.

# In[ ]:


tmp = crew_df[crew_df['movies_created'] > 7].sort_values('mean_movie_popularity', ascending=False).head(50)

plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_popularity', y='name', data=tmp)


# Top 5 names on the list:
#  1. Craig Wood (editor) - Pirates of the Caribbean series, Guardians of the Galaxy movies.
#  2. David Hoberman (producer) - Beauty and the Beast
#  3. Kevin Feige (producer) - movies from Marvel Universe.
#  4. Christopher Nolan (director) - no need to introduce.
#  5. Stan Lee (screenplay by his comic books, producer) - no need to introduce.
# 
# Generally, most of them worked on blockbuster movies.

# In[ ]:


# Get the list of 'popular' crew members IDs
crew_popularity_ids = tmp['id'].values.tolist()


# In[ ]:


# Create new feature indicating if a movie was created by one of 'popular' crew members
train['has_popular_crew'] = train['crew'].apply(lambda x: is_person_in_cast(x, crew_popularity_ids))


# **Which crew member worked on movies with the biggest revenue and the biggest budget?**
# 
# Excluding all crew members that worked on less than eight movies.

# In[ ]:


tmp = crew_df[crew_df['movies_created'] > 7].sort_values(by='mean_movie_revenue', ascending=False).head(50)

traces = []
for _, row in tmp.iterrows():
    traces.append(go.Scatter(
        x=[row['mean_movie_budget']],
        y=[row['mean_movie_revenue']],
        name=row['name'],
        mode='markers'
    ))

layout = go.Layout(
    title = go.layout.Title(text='Crew members based on mean movies revenue and budget'),
    xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(text='Mean movie budget')), 
    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text='Mean movie revenue'))
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig, show_link=False)


# Three out of five people with more than 600M revenue worked on The Lord of the Rings Trilogy:
#  - Fran Walsh (screenplay) [wife of Peter Jackson]
#  - Peter Jackson (director)
#  - Liz Mullane (casting)
# The other two: Joss Whedon and Kevin Feige worked together on Avengers and Avengers: Age of Ultron.
# 
# Just below 600M revenue there is Andrew Lesnie (Director Of Photography) that also worked on The Lord of the Rings Trilogy.
# 
# The biggest budget have movies that Craig Wood produced.

# In[ ]:


# Get IDs for top 100 crew members that have biggest mean movie revenue.
# Excluding all crew members that worked on less than eight movies.
crew_revenue_ids = crew_df[crew_df['movies_created'] > 7].sort_values(by='mean_movie_revenue', ascending=False).head(100)


# In[ ]:


# Create new feature indicating if a movie was created by one of crew members with biggest mean revenue
train['has_top_crew'] = train['crew'].apply(lambda x: is_person_in_cast(x, crew_revenue_ids))


# **Which crew members worked on movies with the biggest profits?**
# 
# Excluding all crew members that worked on less then eight movies.

# In[ ]:


tmp = crew_df[crew_df['movies_created'] > 7].sort_values(by='mean_movie_profit', ascending=False).head(50)

plt.figure(figsize=(15,10))

sns.set_color_codes('pastel')
ax = sns.barplot(x='mean_movie_revenue', y='name', data=tmp, label='mean revenue', color='b')

sns.set_color_codes('muted')
ax = sns.barplot(x='mean_movie_profit', y='name', data=tmp, label='mean profit', color='b')

ax.legend(loc='lower right')


# In most cases, these are the same people as before. Just in a slightly different order. Still LotR and Marvel in front.

# **Who are the best crew member based on job type in case of movie revenue?**
# 
# Excluding all crew members that worked in less than eight movies.

# In[ ]:


fig = plt.figure(figsize=(15,30))

for x, job in enumerate(selected_jobs):
    tmp = (crew_df[(crew_df['job'].apply(lambda x: job in x)) & (crew_df['movies_created'] > 7)]
           .sort_values('mean_movie_revenue', ascending=False).head(5))

    ax = fig.add_subplot(5,2,x+1)
    sns.barplot(x='mean_movie_revenue', y='name', data=tmp, ax=ax)
    ax.set_title(job)


# In all cases, top places belongs to people that worked on movies based on comics (one DC, rest Marvel) or on The Lord of the Rings trilogy.
# 
# Original Music Composer and Producer are aligned categories. There are no big differences between the leaders. Casting, Director of Photography and Costume Design have clear leader.

# **Who are the best crew member from directors in case of movie revenue?**
# 
# Excluding all crew members that worked in less than eight movies.

# In[ ]:


tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Director' in x)) & (crew_df['movies_created'] > 7)]
       .sort_values('mean_movie_revenue', ascending=False).head(25))

plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_revenue', y='name', data=tmp)


# Top 5 places:
#  1. Joss Whedon - Avengers
#  2. Peter Jackson - The Lord of the Rings trilogy, Hobbit movies
#  3. John Lasseter - Toy Story series
#  4. Michael Bay - Transformers collection
#  5. George Lucas - Star Wars movies, Indiana Jones

# **Who are the best crew member from casting in case of movie revenue?**
# 
# Excluding all crew members that worked on less than eight movies.

# In[ ]:


tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Casting' in x)) & (crew_df['movies_created'] > 7)]
       .sort_values('mean_movie_revenue', ascending=False).head(25))

plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_revenue', y='name', data=tmp)


# Top 3 places belongs to The Lord of the Rings.

# **Who are the best crew member from directors of photography in case of movie revenue?**
# 
# Excluding all crew members that worked in less than eight movies.

# In[ ]:


tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Director of Photography' in x)) & (crew_df['movies_created'] > 7)]
       .sort_values('mean_movie_revenue', ascending=False).head(25))

plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_revenue', y='name', data=tmp)


# Top 3 places:
#  1. Andrew Lesine - The Lord of the Rings
#  2. Dariusz Wolski - Pirates of the Caribbean
#  3. Amir Mokri - Man of Steel, Transformers 3, Trnasformers: Age of Extinction

# **Who are the best crew member from costume design in case of movie revenue?**
# 
# Excluding all crew members that worked in less than eight movies.

# In[ ]:


tmp = (crew_df[(crew_df['job'].apply(lambda x: 'Costume Design' in x)) & (crew_df['movies_created'] > 7)]
       .sort_values('mean_movie_revenue', ascending=False).head(25))

plt.figure(figsize=(10,10))
sns.barplot(x='mean_movie_revenue', y='name', data=tmp)


# There are three very clear leaders in this job with huge advantage over the rest costume designers.
#  1. Lindy Hemming - Wonder Woman, The Dark Knight Trilogy
#  2. Deborah Lynn Scott - Titanic, Avatar, Transformers
#  3. Judianna Makovsky - latest Marvel movies

# In[ ]:


# Get person name of job role for the movie
def get_crew_role_name(crew, job):
    for person in crew:
        if person['job'] == job:
            return pd.Series(data=[person['name']], index=[job])
    return pd.Series(data=[''], index=[job])


# In[ ]:


# Create new feature for director
train['director'] = train['crew'].apply(lambda x: get_crew_role_name(x,'Director'))


# **Which director have the biggest mean movie revenue?**

# In[ ]:


directors = train.groupby('director')['revenue'].mean().sort_values(ascending=False).head(50)

plt.figure(figsize=(10, 10))
sns.barplot(x=directors.values, y=directors.index)


# Double check. Directors that are in this list are different from the earlier list. This is the case because in earlier list I show all 'directors' that worked on more than seven movies and what is also important I classify them as directors no matter how many movies they directed.
# 
# Top 5:
#  1. Joss Whedon (Avengers)
#  2. Byron Howard (Zootopia, Tangled, Bolt)
#  3. David Yates (Harry Potter)
#  4. James Gunn (Guardians of the Galaxy)
#  5. Roger Allers (The Lion King)
# 
# In general, the movie's revenue depends on the choice of the person responsible for directing. Directors that are popular, that created movies that viewer seen, have bigger chance for bigger movie revenue in future.

# In[ ]:


# Remove crew dataframe from memory
del crew_df


# ## The End
# 
# Just kidding. It is the end of EDA, I think. In the future I will try to engineer some new features and try to predict box office revenue.
# 
# See you soon.

# **References**:
#  - [medium.com/@nagalr_63588/tmdb-box-office-prediction-kaggle-com-6e14e013955b](https://medium.com/@nagalr_63588/tmdb-box-office-prediction-kaggle-com-6e14e013955b)
#  - [kaggle.com/artgor/eda-feature-engineering-and-model-interpretation](https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation)
