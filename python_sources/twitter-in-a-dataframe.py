#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-datasets-images/62335/120439/c77d6f98f9cff64f2a2fe4749eedcc05/data-original.png?t=2018-10-09-19-30-31)

# # Twitter in a DataFrame
# ## Functions and Recipes for the Social Media Data Scientist (Twitter API)
# 
# #### Overview
# 
# The [`twitter` data API](https://advertools.readthedocs.io/en/master/advertools.twitter.html) module is part of the [`advertools`](https://github.com/eliasdabbas/advertools) package, which provides productivity and analysis tools for online marketing. 
# 
# The module is built on top of `twython` which is one of the main libraries that interface with Twitter's API.  
# Big thanks to the authors [@ryanmcgrath](https://twitter.com/ryanmcgrath) and [@mikehelmick](https://twitter.com/mikehelmick).  
# The functions in this module are simply wrappers around a selection of `twython`'s functions.  
# **What this module does:**
# 1. **Get the results in a DataFrame:** With the exception of three functions that return a list of ID's, everything else returns a `pandas` DataFrame, ready to use. This allows you to spend more time analyzig data, and less time figuring out the structure of the JSON response object. It's not complicated or anything, just takes time. 
# 2. **Manage looping and merging:** there is a limit to how many results you get per request (typically in the 100 - 200 range), several requests have to be made, and merged together. Not all responses have the same structure, so this is also handled. You only have to provide the number of responses you want through the `count` parameter where applicable (provided you are within your app's rate limits). 
# 3. **Unnesting nested objects:** Many response objects contain very rich embedded data, which is usually meta data about the response. For example, when you request tweets, you get a user object along with that. This is very helpful in better understanding who made the tweet, and how influential / credible they are.
# 4. **Documentation:** All available parameters are included in the function signatures, to make it easier to explore interactively, as well as descriptions of the parameters imported from the Twitter documentation. 
# 
# This module provides the ability to request data from Twitter but not to post it. If you want to tweet, follow someone or engage programmatically, you can do it directly through `twython`. This is mainly for retrieving and analyzing data.
# 
# #### API
# 
# Using the API is almost identical to using it with `twython`. 
# Below is a quick comparison.  
# Before starting you will have to create an app through [developer.twitter.com](https://developer.twitter.com/), and then you can get your authentication keys from your dashboard. 
# There are several ways to authenticate, depending on what you want to do and how your app is setup. You can learn more about that from [twython's documentation](https://twython.readthedocs.io/en/latest/usage/starting_out.html)
# 
# **To install: **
# 
# `pip install advertools`

# In[ ]:


import pandas as pd
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.max.columns', None) # 70+ columns to explore! 


# In[ ]:


# get these from your dashboard on developer.twitter.com: 
auth_params = {
    'app_key': 'YOUR_APP_KEY',
    'app_secret': 'YOUR_APP_SECRET',
    'oauth_token': 'YOUR_OAUTH_TOKEN',
    'oauth_token_secret': 'YOUR_OAUTH_TOKEN_SECRET',
}

# twython:
from twython import Twython
twitter = Twython(**auth_params) 
# twitter.search(q='basketball') 
# or any other function and / or parameters 

# advertools: 
import advertools as adv
adv.twitter.set_auth_params(**auth_params)
# adv.twitter.get_user_timeline(screen_name='twitter') 
# or some other function


# You can check what your auth params are for any function by calling `adv.twitter.<function_name>.get_auth_params()` and you will get a dictionary of the keys and parameters.  
# 
# Then all functions are called the same way. So basically, once you have authenticated, you only need to add the `adv` package prefix, and the rest is the same. 
# 
# 
# `tweet_mode='extended'` Say that three times, `tweet_mode='extended'`  
# It took me a while to figure out why I was getting half tweets until I realized that because tweets were originally only 140 characters and now they can have up to 280, there is a transition, and now in order to make sure you get the full tweet text, you need to pass this parameter explicitly.  
# The result is that you will get a column called `full_text` as opposed `text` if you don't pass this parameter. Keep this in mind.  
# In cases where there are embedded data with the respnose, like user data, the column names will have a prefix of `tweet_` and `user_` for example to differentiate betweent them.  
# Some columns like `created_at` and `id` for example, have the same name for users and tweets, so this was done to make sure things are clear, and avoid duplication. 

# #### Exploring the main response `DataFrame`  
# 
# Let's start by exploring what the typical response looks like, what the conventions are, and what you might be interested in doing. I've authenticated with my keys and now ready to get some data.  
# Let's take a look at Python for example, and see what people are tweeting. I'm going to use the query '#python' to get tweets containing that hash tag.

# In[ ]:


# python = adv.twitter.search(q='#python', count=1000, tweet_mode='extended', lang='en')
# python.to_csv('python_tweets.csv', index=False) 


# As you can see above, we just created a thousand-tweet Twitter dataset, with 74 columns, in a tidy format, ready for analysis, and ready to publish on Kaggle datasets, with one line of code. 
# You might want to give it a try, last year Kaggle gave out [ten thousand bucks every month for the best datasets! :)](https://www.kaggle.com/about/datasets-awards/datasets)
# 
# The above code is commented so as to ensure reproducibility. All other code used to import data is commented, and datasets are read from disk. Feel free to run it again, play with the parameters and see what you get.  

# In[ ]:


python = pd.read_csv('../input/python_tweets.csv', 
                     parse_dates=['tweet_created_at', 'user_created_at'])
print(python.shape)
python.head(3)


# In[ ]:


print('Columns starting with "tweet_" :', python.columns.str.contains('tweet_').sum()) 
print('Columns starting with "user_" :', python.columns.str.contains('user_').sum()) 


# So, for every tweet, we have 32 columns related to the tweet, and 44 related to the user who tweeted that tweet. They are sometimes more or less, depending on the parameters, but around these numbers.  
# There are many of those that you will probably never use, and some are going to be depracated soon, so it's up to you to filter what is meaningful for you. 

# In[ ]:


python[['tweet_created_at', 'user_created_at']].dtypes


# You can use these two columns to filter data based on datetime atributes.   
# For example, let's see how many users have a Twitter account that was created before 2010:

# In[ ]:


# we actually need to deduplicate users, but just for a quick demo:
python['user_created_at'].dt.year.lt(2010).sum()


# In[ ]:


python['tweet_created_at'][0]


# The resolution is down to the second, so you can really dig into detailed times.  
# Let's take a look at the tweets that were tweeted by users with the most followers:

# In[ ]:


(python
 .sort_values(['user_followers_count'], ascending=False)
 [['tweet_full_text', 'user_screen_name', 
   'user_followers_count', 'user_description']]
 .head(10))


# I usually like to know how many users tweeted those 1,000 tweets, let's check how many of the user ID's are duplicated: 

# In[ ]:


# number of rows - number of duplicates:
python.shape[0] - python['user_id'].duplicated().sum()


# I'm curious if there are people with a disprportionate number of tweets: 

# In[ ]:


python['user_screen_name'].value_counts().head(10)


# In[ ]:


python['user_screen_name'].value_counts().cumsum().head(10)


# Basically, around 20% of the tweets were tweeted by ten people, and one of them made 108 tweets. Good to know.  
# Some columns contain nested data, which are dictionaries that you can explore further. For example, let's take a look at the `tweet_entities` column. 

# In[ ]:


python['tweet_entities'][0]


# In[ ]:


python['tweet_entities'][1]


# This shows the structured entities that were used in this particular tweet.  
# I think these are very important, as they include hashtags, mentions, URLs, media (images or videos), all structured entities, that give more meaning to the text of the tweet (or the user description).  
# As you can see the nesting is deep and inconsistent, depending on whether or not the data are avilable for the particular tweet. You still have access to the full data, as Python dictionaries, to get a better idea.  
# I'm working on a way to extract them in the same response DataFrame without issues, hopefully by the next release.  
# > Update: Twitter is now sending entities' data in a consistent way, and in the latest advertools release you will get five additional columns, each with a comma-separated list of entities (hashtags, mentions, symbols, URLs, and optionally media.  
# `pip install --upgrade advertools`  
# 
# I have another tutorial if you are interested in how you can [extract those entities, and some ideas on how to analyze them.](https://www.kaggle.com/eliasdabbas/extract-entities-from-social-media-posts) (includes emoji, which are not extracted by the Twitter API).
# 
# The numeric columns are easy to filter (they include 'count'): 

# In[ ]:


python.filter(regex='count', axis=1).head() # user ID and tweet ID are integers but not 'numeric' in this sense


# In[ ]:


print('Boolean columns: ')
python_bool = python.select_dtypes(bool)
python_bool.head()


# In[ ]:


(python_bool
 .mean()
 .to_frame().T
 .rename(axis=0, mapper={0: 'mean:'})
 .style.format("{:.2%}"))


# Keep in mind that the `user_` columns have duplicated values, which have to be filtered out when making calculations. This doesn't matter in the case of `tweet_` columns, because all tweets are unique. Let's double check. 

# In[ ]:


python['tweet_id'].duplicated().sum()


# How "Python" really is this group of people? Are they into the Python programming language, are they talking about the snake, or did they simply happen to tweet that one tweet with that keyword in it?  
# One place to get some hints is the user description. If someone has "python" in their description it is likely that they are.

# In[ ]:


(python['user_description']
 .drop_duplicates()
 .str.contains('python')
 .sum())


# 27, which doesn't mean they aren't, but it's a hint. We can look for web development, or data science: 

# In[ ]:


(python['user_description']
 .drop_duplicates()
 .str.contains('django|flask|web')
 .mean().round(3))


# In[ ]:


(python['user_description']
 .drop_duplicates()
 .str.contains('data|mining|machine learning|ai ')
 .mean().round(3))


# In[ ]:


(python['user_description']
 .drop_duplicates()
 .str.contains('developer|development|programming')
 .mean().round(3))


# In[ ]:


(python['user_description']
 .drop_duplicates()
 .str.contains('developer|development|programming|data|django|flask|web|python|machine learning|ai |mining')
 .mean().round(3))


# How many people don't have a description? 

# In[ ]:


python['user_description'].isna().sum()


# Out of the boolean values, I'm usually interested look at the percentage of `user_default_profile_image`, which means the user doesn't have a profile image. A strong hint it may be a completely new account, or maybe a fake one, or a bot. In our case we have 2.27%. I've seen cases where it's much higher. 
# Another interesting one for me, is whether the user id verified, which shows a higher amount of trust, and being a public / important figure.

# In[ ]:


(python
 .drop_duplicates(subset=['user_id'])['user_verified']
 .apply(['mean', 'count']))


# Another piece of information which might be interesting, about the users, is how they tweet. Meaning the app that they use to interact with Twitter. `tweet_source` and `tweet_source_url` have this information. 

# In[ ]:


python['tweet_source'].value_counts().head(10)


# In[ ]:


print('Number of unique apps:', python['tweet_source_url'].nunique())
python['tweet_source_url'].value_counts().head(10)


# These apps are not necessarily mobile apps. If you use your app on developer.twitter.com to tweet, the name of your app would be shown here as the source.  
# You might want know more about the app by going to the URL, if something looks interesting, or suspicious.  

# We have now formed a general overview of the data, and segmented columns into a few groups. We started to form an initial opinion at a very high level, and we can now dig into the tweets, and user descriptions, which are the richest in information and contain structured and un-structured information.  
# There are obviously many text mining techniques, and it is up to you to choose the one that suits you. I have one function and one thing to say about this:  
# 
# **1. The thing:** when you have numbers and facts describing a text list (tweets in this case, but it could also be movie titles, keywords, news article titles, etc), it's much more informative to take these numbers into consideration, than to analyze the text on its own. Which takes me to...  
# **2. The function:** `word_frequency` does two counts for any text list with a number list describing it. A regular 'absolute count' of the words, and a 'weighted count' based on any of the numbers you have. In this case I will use the number of followers as the weight to use.
# 

# In[ ]:


adv.word_frequency(text_list=python['tweet_full_text'], 
                   num_list=python['user_followers_count']).head(15)


# What, nothing related to websites and web development?  
# Or maybe Django and Flask are too big that people tweet about them without needing to say 'Python' as well?  
# Let's see the same numbers, sorted by the absolute count (`abs_freq`):  

# In[ ]:


(adv.word_frequency(text_list=python['tweet_full_text'], 
                    num_list=python['user_followers_count'])
 .sort_values(['abs_freq'], ascending=False)
 .head(15))


# So this means that web development-related keywords are used more often together with '#python', but the users with the most followers seem to be more into the data science / machine learning topics.  
# If you are interested in more details about this, check out my [tutorial on abosolute vs. weighted word frequency on DataCamp.](https://www.datacamp.com/community/tutorials/absolute-weighted-word-frequency)
# 
# Let's now take a look at the available functions and what you can do with them. The functions are not ordered in any particular way, so feel free to jump to whichever one is interesting to you. This is a recipe-style documentation, along with thoughts and examples of what you can do.  

# ## `advertools.twitter` Module Functions:
# 
# #### [Get Application Rate Limit Status](#get_application_rate_limit_status):
# Returns the current rate limits for methods belonging to the
#         specified resource families.
# 
# #### [Get Available Trends](#get_available_trends):
# Returns the locations that Twitter has trending topic information for.
# 
# #### [Get Favorites](#get_favorites):
# Returns the 20 most recent Tweets favorited by the authenticating
#         or specified user.
# 
# #### [Get Followers IDs](#get_followers_ids):
# Returns a cursored collection of user IDs for every user
#         following the specified user.
# 
# #### [Get Followers List](#get_followers_list):
# Returns a cursored collection of user objects for users
#         following the specified user.
# 
# #### [Get Friends IDs](#get_friends_ids):
# Returns a cursored collection of user IDs for every user the
#         specified user is following (otherwise known as their "friends").
# 
# #### [Get Friends List](#get_friends_list):
# Returns a cursored collection of user objects for every user the
#         specified user is following (otherwise known as their "friends").
# 
# #### [Get Home Timeline](#get_home_timeline):
# Returns a collection of the most recent Tweets and retweets
#         posted by the authenticating user and the users they follow.
# 
# #### [Get List Members](#get_list_members):
# Returns the members of the specified list.
# 
# #### [Get List Memberships](#get_list_memberships):
# Returns the lists the specified user has been added to.
# 
# #### [Get List Statuses](#get_list_statuses):
# Returns a timeline of tweets authored by members of the specified list.
# 
# #### [Get List Subscribers](#get_list_subscribers):
# Returns the subscribers of the specified list.
# 
# #### [Get List Subscriptions](#get_list_subscriptions):
# Obtain a collection of the lists the specified user is subscribed to.
# 
# #### [Get Mentions Timeline](#get_mentions_timeline):
# Returns the 20 most recent mentions (tweets containing a users's
#         @screen_name) for the authenticating user.
# 
# #### [Get Place Trends](#get_place_trends):
# Returns the top 10 trending topics for a specific WOEID, if
#         trending information is available for it.
# 
# #### [Get Retweeters IDs](#get_retweeters_ids):
# Returns a collection of up to 100 user IDs belonging to users who
#         have retweeted the tweet specified by the ``id`` parameter.
# 
# #### [Get Retweets](#get_retweets):
# Returns up to 100 of the first retweets of a given tweet.
# 
# #### [Get Supported Languages](#get_supported_languages):
# Returns the list of languages supported by Twitter along with
#         their ISO 639-1 code.
# 
# #### [Get User Timeline](#get_user_timeline):
# Returns a collection of the most recent Tweets posted by the user
#         indicated by the ``screen_name`` or ``user_id`` parameters.
# 
# #### [Lookup Status](#lookup_status):
# Returns fully-hydrated tweet objects for up to 100 tweets per
#         request, as specified by comma-separated values passed to the ``id``
#         parameter.
# 
# #### [Lookup User](#lookup_user):
# Returns fully-hydrated user objects for up to 100 users per request,
#         as specified by comma-separated values passed to the ``user_id`` and/or
#         ``screen_name`` parameters.
# 
# #### [Retweeted Of Me](#retweeted_of_me):
# Returns the most recent tweets authored by the authenticating user
#         that have been retweeted by others.
# 
# #### [Search](#search):
# Returns a collection of relevant Tweets matching a specified query.
# 
# #### [Search Users](#search_users):
# Provides a simple, relevance-based search interface to public user
#         accounts on Twitter.
# 
# #### [Show Lists](#show_lists):
# Returns all lists the authenticating or specified user subscribes to,
#         including their own.
# 
# #### [Show Owned Lists](#show_owned_lists):
# Returns the lists owned by the specified Twitter user.

# <a id="get_application_rate_limit_status"></a>
# **`get_application_rate_limit_status`**  
# You need to keep track of your consumption of your app's rate limit, and there is a very simple way to do that:

# In[ ]:


# rate_limit = adv.twitter.get_application_rate_limit_status(consumed_only=False)
# rate_limit.to_csv('data/app_rate_limit_status.csv', index=False)


# In[ ]:


rate_limit = pd.read_csv('../input/app_rate_limit_status.csv')
rate_limit.sample(10)


# The names are clear in general. The reset time is when your limit resets (which is usually every fifteen minutes.  
# In order not get lost in the details of all kinds of limits, you can run the same function with the default `consumed_only=True`, every now and then (or before requesting a large amount of data).   
# It gets the current rate limit status, and filters the rows where you have consumed part of the limit.

# In[ ]:


# adv.twitter.get_application_rate_limit_status()


# #### What's Trending?
# <a id="get_available_trends"></a>
# **`get_available_trends`**  
# First you need to know what locations are available, then you can query the ones that are of interest. You would usually run this only once, and refer to it when needed.

# In[ ]:


# available_trends = adv.twitter.get_available_trends()
# available_trends.to_csv('data/available_trends.csv', index=False)


# In[ ]:


available_trends = pd.read_csv('../input/available_trends.csv')
print(available_trends.shape)
available_trends.sample(15)


# In[ ]:


(available_trends
 [available_trends['name'].duplicated(keep=False)]
 .sort_values(['name']))


# Watch out, there are some duplicated city names.
# 
# Once you figure out which countries and / or cities you are interested in, you can request the trending topics by using each location's `woeid` number.  
# Let's take a look at Spain, because it's one of my favorite places!

# In[ ]:


spain_ids = available_trends.query('country == "Spain"')
spain_ids


# Now let's see what's trending in Spain. Which takes us to another function:

# <a id="get_place_trends"></a>
# **`get_place_trends`**  
# Once you have identified the `woeid`'s of the locations of interest, you can simply pass them to this function, and get the trends. You can pass one integer value, or a list of integers.

# In[ ]:


# spain_trends = adv.twitter.get_place_trends(ids=spain_ids['woeid'])
# spain_trends.to_csv('data/spain_trends.csv', index=False)


# In[ ]:


spain_trends = pd.read_csv('../input/spain_trends.csv')
print(spain_trends.shape)
spain_trends.sample(15)


# Just to make it easier to read: 

# In[ ]:


spain_trends[['name', 'tweet_volume', 'location', 'time']].sample(15)


# Note the inconsistency in the naming conventions. In the available trends DataFrame, `name` refers to the name of the location. Once we get the place trends, `name` refers to the name of the topic / hash tag.  
# 
# By the way, you usually get `NaN`'s in `tweet_volume`, probably because these are topics that are starting to trend, but there is no meaningful tweet volume to return yet, I'm not sure. But that's why tweet volume is a float and not an integer. 
# 
# It's interesting to know how related or dependent the trends are across cities, and in the country as a whole. 

# In[ ]:


(spain_trends[['name', 'tweet_volume', 'location', 'time']]
 .groupby(['location']).head(2)) # this works because data are sorted based on tweet_volume, 
                                 # otherwise you have to sort


# It's good to know whether or not there are differences across cities.    
# Even if it's not visible from the top 3-4 trends, there might be more interesting hidden relations between the cities and topics. Spain has eleven locations, and returns a total of around 500 trends for example. 

# Or maybe you are just interested in one city: 

# In[ ]:


spain_trends.query('location=="Madrid"')[['name','tweet_volume','location', 'time']]


# My suggestion is to run this on a continuous basis, and accumulate a large evolving dataset. 
# You can specify all the locations of interest, and schedule this function to run for those locations, once or maybe even more every day. You can save the result in one file, and see if there are relations the come up, if certain locations tend to dominate, or any other suprising, interesting observations. 

# <a id="get_favorites"></a>
# **`get_favorites`**  
# Get the favorites "likes" of a certain user. Let's see what the @twitter account likes on Twitter.

# In[ ]:


# twtr_favs = adv.twitter.get_favorites(screen_name='twitter', 
#                                       count=3000, tweet_mode='extended')
# twtr_favs.to_csv('data/twitter_favorites.csv', index=False)


# In[ ]:


twtr_favs = pd.read_csv('../input/twitter_favorites.csv')
twtr_favs.head()


# Who's content does @twitter like the most? 

# In[ ]:


(twtr_favs
 .user_screen_name
 .value_counts(normalize=False)
 .head(10))


# In[ ]:


(twtr_favs
 .user_screen_name
 .value_counts(normalize=True)
 .cumsum()
 .head(30))


# You 'like' yourself a little too much @twitter! :)
# 
# This is the same DataFrame as the one we first explored, so no need to get into its details again.  
# But you have a little less than 3,000 tweets that @twitter likes. This can uncover interesting trends in the behavior of accounts / organizations that you follow (or can give insights on what your accounts like on Twitter, and whether or not you want to continue engaging in this way). 

# <a id="get_followers_ids"></a>
# **`get_followers_ids`**  
# This is one of the few functions that doesn't return a DataFrame. It simply returns a list of IDs as the name suggests.

# Who follows @Kaggle?  
# Import and save the data to disk: 

# In[ ]:


# kaggle_followers = adv.twitter.get_followers_ids(screen_name='kaggle', count=5000)

# import json
# with open('data/kaggle_follower_ids.json', 'wt') as file:
#     json.dump(kaggle_followers, file)


# In[ ]:


import json
with open('../input/kaggle_follower_ids.json', 'rt') as file:
    kaggle_follower_ids = json.load(file)


# In[ ]:


print(kaggle_follower_ids.keys(), '\n')
print('previous cursor: ', kaggle_follower_ids['previous_cursor'])
print('next cursor: ', kaggle_follower_ids['next_cursor'])
print('Follower IDs:', kaggle_follower_ids['ids'][:10])
print('List length:', len(kaggle_follower_ids['ids']))


# `ids` is the list of interest.  
# `previous_cursor` and `next_cursor` are important if you are requesting a large number of IDs. In that case your limit will run out. So you will have to run several ruquests and combine them.  
# In order to prevent duplicate data, for every new request you need to provide the `next_cursor`.  
# Note that the `previous_cursor` is 0 for the first request, meaning there aren't any IDs before that to retrieve. 

# <a id="get_followers_list"></a>
# **`get_followers_list`**  
# Similar to `get_followers_ids`, but this one returns a full user object. This is the same as the second half of the first DataFrame we received above ('#python' tweets).  
# Since we're talking Python, let's get the followers of @ThePSF:

# In[ ]:


# the_psf = adv.twitter.get_followers_list(screen_name='ThePSF', count=2500, 
                                        # include_user_entities=True)
# the_psf.to_csv('data/the_psf_followers.csv', index=False)


# In[ ]:


the_psf = pd.read_csv('../input/the_psf_followers.csv')
print(the_psf.shape)
the_psf.head()


# <a id="get_friends_ids"></a>
# **`get_friends_ids`**  
# Same as `get_followers_ids` but this one shows the IDs of users that the account follows.
# 
# <a id="get_friends_list"></a>
# **`get_friends_list`**  
# Same as `get_followers_list`
# 
# <a id="get_home_timeline"></a>
# **`get_home_timeline`**  
# This gets your own timeline, which is a list of tweet objects, exactly like the first one we got.
# 
# ### Twitter Lists
# These are the collections of users (screen names) that we create, and form a tailored timeline. Some interesting functions can help here:
# 
# <a id="get_list_members"></a>
# **`get_list_members`**  
# User objects of all members in a certain list. Exactly like `get_followers_list`. 
# 
# <a id="get_list_memberships"></a>
# **`get_list_memberships`**  
# Let's explore what list objects look like, again we look at Kaggle.  
# This shows to which lists @kaggle has been added by other people. People who create such lists in such a niche are very much into this topic, so it might be interesting to see who those people are.  
# As with tweets, you get a DataFrame that is partly about the list and partly about the user who created the list. 

# In[ ]:


# kaggle_list_memberships = adv.twitter.get_list_memberships(screen_name='kaggle', count=500)
# kaggle_list_memberships.to_csv('data/kaggle_list_memberships.csv', index=False)


# In[ ]:


kaggle_list_memberships = pd.read_csv('../input/kaggle_list_memberships.csv')
kaggle_list_memberships.shape


# In[ ]:


kaggle_list_memberships.head()


# Similar to the search response, the column names have been prefixed with `list_` and `user_`.

# <a id="get_list_statuses"></a>
# **`get_list_statuses`**  
# Nothing different from all other functions that return tweet objects.  
# My recommendation is that for your daily work in following a certain market, it can be good to create a special list of all the influencers, and relevant people in that market, and constantly get their statuses.  
# This list would evolve over time, based on changes in the market, newcomers, and changing conditions.
# 
# <a id="get_list_subscribers"></a>
# **`get_list_subscribers`**  
# Returns user objects like `get_followers_list`.
# 
# <a id="get_list_subscriptions"></a>
# **`get_list_subscriptions`**  
# Returns list objects like `get_list_memberships`.
# 
# <a id="get_mentions_timeline"></a>
# **`get_mentions_timeline`**  
# Returns tweet objects, same as `search`.
# 
# <a id="get_retweeters_ids"></a>
# **`get_retweeters_ids`**  
# Returns a list of retweeter IDs same as `get_followers_ids`.
# 
# <a id="get_retweets"></a>
# **`get_retweets`**  
# Returns tweet objects (retweets of a certain tweet).
# 
# 
# <a id="get_supported_languages"></a>
# **`get_supported_languages`**  
# Nothing terribly useful. Just a reference to the available languages.  
# Refer to it if you want to make sure you are using the right language code when constructing your queries. 

# In[ ]:


# languages = adv.twitter.get_supported_languages()
# languages.to_csv('data/languages.csv', index=False)


# In[ ]:


languages = pd.read_csv('../input/languages.csv')
print(languages.shape)
languages.head()


# <a id="get_user_timeline"></a>
# **`get_user_timeline`**  
# Returns tweet objects but for a specific user. Excellent to analyze a certain account, which could be a partner, a competitor, or simply an interesting account(s).
# 
# <a id="lookup_status"></a>
# **`lookup_status`**  
# Tweet objects, but you need to supply the IDs of those tweets. These could have been retrieved by other functions, and now you need to dig deeper into those tweets.  
# 
# <a id="lookup_user"></a>
# **`lookup_user`**  
# Returns user objects. These users are the full user objects based on the IDs that you need to specify. Useful If you hve a list of user IDs, and need the full data about them.
# 
# <a id="retweeted_of_me"></a>
# **`retweeted_of_me`**  
# Returns tweet objects where people retweeted your tweets.
# 
# <a id="search_users"></a>
# **`search_users`**  
# Returns user objects. Very useful when you are exploring a certain trend or topic. You basically search for a certain query "basketball", "finance", "makeup", etc. and you get the most relevant accounts that match those criteria. 
# 
# 
# <a id="show_lists"></a>
# **`show_lists`**  
# Returns list objects. 
# 
# <a id="show_owned_lists"></a>
# **`show_owned_lists`**  
# Returns list objects.
# 
# <a id="search"></a>
# **`search`**  
# This is the big one, and the most versatile in terms of query flexibility. Although the retun value is simply a bunch of tweet objects, together with their users, the available search operators are quite powerful. Let's get them from the documentation.

# In[ ]:


# search_operators = pd.read_html('https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators', 
#                                header=0)[0]
# search_operators.to_csv('data/twitter_search_operators.csv', index=False)


# You can have up to 500 characters in your query, and you can combine different queries, to narrow / broaden your search.   
# Here's what I find interesting: 
# 
# - The "OR" operator: makes it flexible to have any of a group of keywords. 
# - The "exact match": where you look for a certain phrase. Useful with full names, brands, topics, and general disambiguation. `dodge OR charger` will get you all kinds of tweets about electrical chargers, but `"dodge charger"` will keep you out of trouble.  
# - Exclusion with the minus sign: `coffee -beans`, `cars -cheap`, `movies -free` and so on. 
# - Mentions vs. replies: `to:NASA` is someone replying to this account while `@NASA` is someone simply mentioning them. Replying is more of a reactive thing than proactive mentioning is. Depending on what you need you can select one or the other. Better, create two data sets and compare them. 
# - Filtering retweets: `keyword -filter:retweets` for the same reason, I also like to get tweets that were native proactive tweets, as opposed to retweets. These are what people wanted to tweet out of their own desire, and not in response to someone saying something. I feel it's a stronger indicator of people's interests / desires. But again, you might want to compare the two, and see if there are any differences. 
# - The url operator: tweets containing a certain word and linking to a certain domain. Which pages are people linking to, in your site? Which pages are they linking to, in your competitor's site?  
# 
# Explore the operators below and see which ones make sense for you. 

# In[ ]:


search_operators = pd.read_csv('../input/twitter_search_operators.csv')
pd.set_option('display.max_colwidth', 160)
search_operators


# As you can see, it's very easy to create a dataset with one function call and start with a ready DataFrame right away.   
# You can also see how rich the data about those tweets and users are.  
# An important source of data that wasn't discussed here are the images that users upload in their tweets. These can be found under the entities columns, and you will need to extract their URLs and download them for further analysis. 
# Hope you have fun getting Twitter data!
# If you come across any issues or anything is not clear, feel free to leave a comment.  
# 
# `pip install advertools`  
# Happy analyzing:)
# 
