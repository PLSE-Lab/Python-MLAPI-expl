#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook and its data is to better understand cancer care in Canada and how it has changed since March 1, 2020. This notebook is intended to help answer "How are patterns of care changing for current patients (i.e., cancer patients)?". Data used in this notebook came from Twitter and so the perspective on how cancer care has changed is from the general public/Twitter community from the 6 larget cities (plus 200km radius) in Canada. The included .csv file are summary results from scraped tweets, not the tweets themselves. Details on how the tweets were obtained and summarized are provided at the end of this notebook.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input/covid19-canada-tweet-summary'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# load tweet summary data and inspect to understand what we're working with
df= pd.read_csv('../input/covid19-canada-tweet-summary/covid19_tweet_summary.csv')
df.head()


# # Understanding the Tweet Summary Data
# The first five columns contain information used to retrieve the tweets. The remaining columns contain information about the retrieved tweets themselves. For example, for row 0, tweets that contained the words 'covid19' and 'cancer' (not case-sensitive) that were published within 200km of Toronto between March 1 and March 7, 2020 (inclusive) were retrieved. A total of 2 tweets met this criteria. 1 user published these tweets and is a member of the general public. The tweets themselves were about fear or concern regarding the virus.
# 
# # Visualizing the Tweet Summary Data
# Since we can't tell based on the head of the dataframe, let's see how many tweets were retrieved, the search terms that were used to retrieve the tweets, and the Canadian cities that the tweets were published from.

# In[ ]:


group_terms= df.groupby('Search terms')['Number of tweets'].sum()
print('Total tweets retrieved across Canada from March 1 to April 18, 2020:',np.sum(group_terms))
print('Search terms used:', df['Search terms'].unique())
print('Where tweets are from:', df['City +200km radius'].unique())


# Let's visualize the distribution of tweets - how many tweets contain each of the search terms and where people are tweeting from.

# In[ ]:


ax= group_terms.plot.bar(title= 'Count of Tweets Containing Certain Words, all cities')
ax.set_xlabel('Words Tweet Contains')
ax.set_ylabel('Number of Tweets')

plt.figure()
group_city= df.groupby('City +200km radius')['Number of tweets'].sum()
ax2= group_city.plot.bar(title= 'Count of Tweets from Each City, all search terms')
ax2.set_xlabel('City +200km radius')
ax2.set_ylabel('Number of Tweets')

plt.figure()
term_city= df.groupby(['City +200km radius', 'Search terms'])['Number of tweets'].sum()
ax3= term_city.plot.bar(title= 'Count of Tweets from Each City by Search Term')
ax3.set_xlabel('City +200km radius, Search terms')
ax3.set_ylabel('Number of tweets')


# **A large proportion of tweets related to Covid-19 and cancer contain the words "covid 19" and "cancer" and are being tweeted from the Toronto and Vancouver areas.
# **
# 
# We can also verify that not all tweets are coming from a small number of users and see who is tweeting (general public, doctors or healthcare organizations, news, or government).

# In[ ]:


#get sum of each column of users tweeting and put in array so it can be easily plotted
users= np.array([['General public','Doctors/Healthcare organizations','News','Government'],[df["Who's tweeting - public"].sum(),df["Who's tweeting - doctors/healthcare org"].sum(),df["Who's tweeting - news"].sum(),df["Who's tweeting - gov"].sum()]])
print('Number of unique Twitter users tweeting about Covid-19 and cancer:', df['Number of unique users'].sum())
plt.pie(users[1,:], labels= users[0,:], rotatelabels= True)
plt.title("Distribution of Who's Tweeting about Covid-19 and Cancer")


# **The majority of people tweeting about Covid-19 and cancer are members of the general public.** These users showed no indication of being a doctor, healthcare organization, news outlet, or member of a government anywhere in their twitter handle or username.
# 
# We can also visualize the distribution of when the tweets were published.

# In[ ]:


group_date= df.groupby('Week number')['Number of tweets'].sum()
ax_date= group_date.plot.bar(title= 'Count of Tweets from Each Week (Mar 1 - Apr 18, 2020)')
ax_date.set_ylabel('Number of tweets')


# **From this plot, it is evident that there has been an increase in discussion among the general public regarding Covid-19 and cancer since March 1, 2020.** To better understand this discussion, the 545 scraped tweets were read and categorized into one of five categories: concern regarding the virus, concern regarding cancer, description of a change in personal cancer treatment, support/information related to Covid-19 and/or cancer, and other (not directly related to virus and/or cancer). Visualization of the distribution of the tweets regarding their content can be seen below.

# In[ ]:


#get sum of each column of counts of tweet content and put in array so it can be easily plotted
tweet_cont= np.array([['Virus concern','Cancer concern','Change in cancer treatment','Support/Info','Other'],[df['Content - virus fear/concern'].sum(), df['Content - cancer treatment concern'].sum(), df['Content - change to personal cancer treatment'].sum(), df['Content - support/information'].sum(),  df['Content - misc (not directly related to virus or cancer)'].sum()]])
plt.pie(tweet_cont[1,:], labels= tweet_cont[0,:], rotatelabels= True)


# # Conclusion
# The purpose of this notebook was to better understand cancer care in Canada and how it has changed throughout the Covid-19 pandemic. Using Twitter as a data source and the particular tweets that were scraped and used in this notebook, we can conclude that there has been an increase in conversations about Covid-19 and cancer in Canada since March 1, 2020 as indicated by the increase in the number of tweets containing words related to Covid-19 and cancer. The majority of Canadians talking about Covid-19 and cancer are from the Toronto and Vancouver areas and are of the general public. Although it is evident that the conversation has increased, this Twitter data indicates that there has been little to no change in cancer care in Canada since March 1, 2020; the majority of tweets published during this time are not related to a change in one's cancer treatment. **This doesn't mean that cancer care hasn't changed since March 1, 2020.** To further investigate how cancer care in Canada has changed since March 1, 2020, more tweets that contain different search terms can be scraped and categorized. The same activity can also be completed for other public platforms, such as Facebook and cancer support groups.

# # How the Tweet Summary Data was Created
# First tweets were scraped using twitterscraper (https://github.com/taspinar/twitterscraper). twitterscraper was used 168 times (4 search terms x 6 cities x 7 weeks) and a .json file containing the tweets and their meta-data were saved for each run whenever tweets that met the search criteria were available. The labelling of the type of user ("general public", "news", etc.) was done manually and required reading the usernames of each unique user. The labelling of the tweet content ("virus concern", "cancer concern", etc.) was done manually and required reading each tweet. The summary data was created for the sole purpose of categorizing the users and tweets. If you're not interested in this level of detail (or are concerned about the labelling bias that comes with this) and only want to see statistics such as number of tweets or number of unique users publishing tweets, the raw tweet (.json) files can be used instead.

# In[ ]:




