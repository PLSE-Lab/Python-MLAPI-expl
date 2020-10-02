#!/usr/bin/env python
# coding: utf-8

# # Russian Tweet Network and Time Series Visualization
# 
# The goal of this notebook is to clean the messy data -- filled to the brim with natural language -- and analyze it via time series analysis. Since a lot of this data is based in a contemporary political context, it's important to note how the data aligns with certain political events during the period in question. No machine learning classification will be done in this project for now: it's purely a visual exploration of the data to understand it.
# 
# I would love to know any tips and tricks people have for working with time series data in python!
# 
# TODO:
# 3. Annotate TS plots with important political dates

# In[ ]:


import numpy as np #linear algebra
import pandas as pd #data processing

import seaborn as sns #visualization
import matplotlib
import matplotlib.pyplot as plt #visualization
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')

from datetime import datetime


# In[ ]:


users = pd.read_csv('../input/users.csv')

users.head()


# ## Time Series Visualizations
# This first section looks at a number of metrics and places them in a time series visualization. The first set is merely a distribution of the number of statuses by certain users, and a distribution of the number of followers. Both plots are based on the users set. Following this, I format the date the twitter user was created and then show two bar plots for users created by year and by month. Finally, I show this same information in a single time series plot using the matplotlib.pyplot.plot function -- this is because the seaborn.tsplot function doesn't work very easily and requires a lot of tinkering to get it working properly. I gave up and chose to just go the easier route. I might change it to a seaborn.pointplot later on.
# 
# The second section follows a similar process, but for the tweets dataset. I import the data, clean the date-time values, then extract the months and years and plot this in another time series chart using the matplotlib.pyplot.plot function. One important insight stands out immediately from these visualizations: the vast majority of users in this dataset created their accounts in 2013 or 2014, with a few trickling later on in 2015 and 2016, BUT the vast majority of tweets came during 2016, particularly during the fall. This timeline coincides with the post-convention general election campaigns along with a number of political events like the Wikileaks dump of DNC emails. Even following Election Day in November of 2016, a number of tweets still came in during the transition period and shortly thereafter. 
# 
# A couple of notes on the code:
# 1. In the first section I queried the user data by those users with non-nan values, as the seaborn.distplot did not automatically clean these out and constantly returned an error: useful to note for later analysis. 
# 2. I used df.assign to create my new dataframe columns when working the date-time metrics. This was an easy and logically straightforward method of creating new values that didn't return errors such as the dreaded "cannot be indexed on a slice" error that Pandas will throw often. In addition, for the date-time data it was necessary to change the index of the dataframe to the date-time data. This made it easy to group the data by months and years. This is of course rather different from another method, which would have been to merely create two new columns -- one for months and one for years. However, I found that this made it more difficult to create a time series representation of the data; when I grouped the data by these two columns, as I originally tried, it created two indices, which constantly returned errors when attempting to plot it. 

# In[ ]:


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.distplot(users[np.isfinite(users.statuses_count)].statuses_count, ax=ax[0])
sns.distplot(users[np.isfinite(users.followers_count)].followers_count, ax=ax[1])
plt.show()


# In[ ]:


form = '%a %b %d %H:%M:%S %z %Y'
users = users.assign(date = users.created_at.map(
    lambda x: datetime.strptime(str(x), form).date() if x is not np.nan else None))
users = users.set_index(pd.DatetimeIndex(users.date))


# In[ ]:


monthseries = users.groupby(by=[users.index.month]).count()
YearSeries = users.groupby([users.index.year]).count()
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.barplot(monthseries.index, monthseries.id, ax=ax[0])
sns.barplot(YearSeries.index, YearSeries.id, ax=ax[1])
plt.show()


# In[ ]:


TimeSeries = users.groupby([users.index.year, users.index.month]).count()
plt.figure(figsize=(12,6))
TimeSeries.id.plot()
plt.xticks(rotation=45)
plt.ylabel('Number of New Users')
plt.xlabel('Year, Month')
plt.show()


# In[ ]:


tweets = pd.read_csv('../input/tweets.csv')


# In[ ]:


form = '%Y-%m-%d %H:%M:%S'
tweets = tweets.assign(date = tweets.created_str.map(
    lambda x: datetime.strptime(str(x), form).date() if x is not np.nan else None))
tweets = tweets.set_index(pd.DatetimeIndex(tweets.date))


# In[ ]:


timeseries = tweets.groupby([tweets.index.year, tweets.index.month]).count()
plt.figure(figsize=(12,6))
timeseries.user_id.plot()
plt.xticks(rotation=45)
plt.ylabel('Number of New Tweets')
plt.xlabel('Year, Month')
plt.show()


# ## Textual Analysis, Cleaning, and Twitter Handle Networks
# This second section in the project was new ground for me. Textual analysis is something I haven't spent much time working with, and this dataset represented a good first start to it. Thankfully, I learned that Pandas has functions dedicated to dealing with text in a REGEX format. As the source code shows, I first made sure to copy the next I wanted to work with before doing anything to it. Then, I extracted twitter handles of tweets that were merely retweets from other people, and by extracting these handles I could determine who the most retweeted accounts were. Following this clean, I replaced a number of string formats with empty strings to make it easier to show in a word cloud the most important words. Https website links were removed, along with the twitter handles, RT, amp, and co. Without removing these they showed up in a large format in the word cloud.
# 
# The word cloud itself is something also new to me. I followed some code found elsewhere online, created one long string of text from the previously cleaned set of text data, then plotted the word cloud image without axes. Clearly, Donald Trump, Trump, Obama, Hillary Clinton, and Hillary are the most used words in the dataset -- which follows since these were mostly politically motivated tweets. 
# 
# After creating the word cloud, I analyzed the retweeted users. I checked to see if any of these retweets were from other members of the dataset. Originally, I found that there weren't any: however, I realized I was conducting the analysis incorrectly. My retweets extracted included the @ symbol and the colon (:) symbol. I removed the colon symbol in order to get rid of anything that might still be in the string after it and leave just the @user_name string. After this, I got a single value of each name in the list of user_keys from the tweets dataset and the retweets already collected. I did this by doing a count of each unique username using df.value_counts() then getting the index of that dataframe (the df.value_counts().values returns the counted numbers rather than the names, a mistake I originally made). In order to make sure names lined up, I added a @ symbol to the beginning of each user_key obtained from the original dataset. From this, I obtained a list of user_names that were simultaneously PART of the dataset, and retweeted BY the dataset: this creates a networking cascade effect, which I then attempted to quantify.
# 
# After extracting the user names from both sets, I was able to quantify it after a lot of trial and error: pandas.Index has an intersection function of a second set of values, such as index1.intersection(values) that allowed me to get a series of usernames along with the number of tweets contained in the dataset!
# 
# Couple of things stand out: first off, the total number of retweets is ~37000, and the total number of retweets from users in the dataset is ~35000, so nearly all of the retweets are from users within the dataset. Second, that number is roughly 18% of the total malicious tweets dataset, so if there was a cascade effect it likely wasn't very large compared. 

# In[ ]:


tags = tweets.text.copy()

# This code extracts where the retweet is from, as it follows a "RT @XXXXX:" format
retweets = tags.str.extract('(@.*:)', expand=True)

# Gets rid of website links
tags = tags.replace('https.*$','',regex=True)
# Gets rid of twitter handles
tags = tags.replace('@.*:','',regex=True)
# Gets rid of RT
tags = tags.replace('RT|amp|co','',regex=True)


# In[ ]:


from wordcloud import WordCloud, STOPWORDS

text = ' '.join([str(x) for x in tags.values])

wc = WordCloud(stopwords=STOPWORDS,background_color='white',max_words=200,scale=3).generate(text)
plt.figure(figsize=(15,15))
plt.axis('off')
plt.imshow(wc)
plt.show()


# In[ ]:


retweets = retweets.replace(':.*','',regex=True)
print(retweets[0].value_counts().describe())


# In[ ]:


user_retweeted = retweets[0].value_counts().index[~pd.isnull(retweets[0].value_counts().index)]
retweeted_user = ['@'+x for x in tweets.user_key.value_counts().index]

cascade = [x for x in retweeted_user if x in user_retweeted]


# In[ ]:


# easy method to remove the @ symbol again and make a clean user_key set
cascade = pd.Series(cascade).replace('@','',regex=True)
network = tweets.user_key.value_counts()
# wish I knew of this previously, kept trying to join two pd.Series which pandas isn't a fan of
network = network[network.index.intersection(cascade.values)]
network


# In[ ]:


print('Total number of retweets from users contained within the dataset: {}'.format(network.values.sum()))
print('Percentage of total dataset: {}'.format(network.values.sum()/len(tweets)))


# If you liked this notebook or can think of other things I might try and do with it let me know! I will likely come back to this if I come up with some other kind of ideas for it.
