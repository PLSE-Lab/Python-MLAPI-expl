#!/usr/bin/env python
# coding: utf-8

# ## Context:
# An NGO wants to track the outbreak of diseases, and believes
# this could be done by analyzing sentiments and keywords of (geotagged)
# non-English twitter messages.

# ## Assignment: 
# The goal is to build a tool that can track the usage of disease-related
# keywords (e.g. stomache ache, headache, fever) in non-English tweets
# (for this case in French).
# 

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Resources:
# - https://pypi.org/project/googletrans/
# - https://stackabuse.com/accessing-the-twitter-api-with-python/ 
# - https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
# - https://pypi.org/project/googletrans/
# - https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0019467&type=printable
# - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4763717/ 
# - https://geopy.readthedocs.io/en/1.16.0/#usage-with-pandas
# 

# ## Code:

# ### Initialisation:
# 
# 
# We suppose that we want to track ebola disease, where poeple talk about it in french. We set up a list of keywords in english and translate it to french using google translate tool.

# In[ ]:


#Set up input list
keywords = ['stomach ache','ebola','fever','vaccin','epidemic','#ebola','#Virus','vomit','virus ebola','symptoms ebola','"diarrhea due to ebola"']
#n = int(input("Enter the number of keywords : ")) 
#for i in range(0,n):
#    keywords.append(input())
print(keywords)


# In[ ]:


#Install and import module for this part
get_ipython().system('pip install googletrans')
from googletrans import Translator


# In[ ]:


#Translate
translator = Translator()
keywords_trad = []
translations = translator.translate(keywords, dest='fr')
for translation in translations:
    keywords_trad.append(translation.text)
    print(translation.origin, ' -> ', translation.text)


# ### Keywords research on Twitter:
# 
# Now that we have our translate keywords, we have to search it on twitter by mean of twython. This module use twitter api thus we signed to a twitter developper account. We save all credentials parameters in a dictionary in json format and load it in our code for the next.

# In[ ]:


#Install and import module for this part
get_ipython().system('pip install twython')
from twython import Twython
import json


# In[ ]:


# Load credentials from json file
with open("/kaggle/input/credentials/twitter_credentials.json", "r") as file:
    creds = json.load(file)


# In[ ]:


# Instantiate an object
python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

# Create our query and search tweets
dict_ = {'user': [], 'date': [], 'text': [], 'location': []}
for i in keywords_trad:
    query = {'q': ' {}  -filter:retweets'.format(i),
             #'geocode':'-4.437584, 15.252361,1000km',
            'result_type': 'mixed',
            'count': 1000,
            'lang': 'fr',
            }
    for status in python_tweets.search(**query)['statuses']:
        dict_['user'].append(status['user']['screen_name'])
        dict_['location'].append(status['user']['location'])
        dict_['date'].append(status['created_at'])
        dict_['text'].append(status['text'])


# ### Classification and cleaning data:
# 
# We classify our data into a dataframe that contain informations for each tweets related to keywords researched. We have user, location, date and text informations. We convert date format into a simple year/month/day format and perform other cleaning task. The free twitter API provide use data for only 7 days (including the day on which we perform the task). In order to get more data, we save an older dataframe for example the day before, add it to the new one and delete all duplicated data. 

# In[ ]:


#Modules
import pandas as pd
from datetime import datetime 
from email.utils import mktime_tz, parsedate_tz

#Function to convert date format
def parse_datetime(value):
    time_tuple = parsedate_tz(value)
    timestamp = mktime_tz(time_tuple)
    return datetime.fromtimestamp(timestamp)

# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)
#convert twitter date to another date time format
for i in range(len(df['date'])):
    df['date'][i] = parse_datetime(df['date'][i])
df['date'] = df['date'].map(lambda x: str(x)[0:10])
df
#If we want to save this dataframe
#df.to_csv('tweets_saved.csv')


# In[ ]:


#Load an older dataframe with older tweets
older_tweets = pd.read_csv("/kaggle/input/oldertweets/tweets2.csv")
older_tweets = older_tweets.drop(older_tweets.columns[[0]], axis=1)
older_tweets['date'] = older_tweets['date'].map(lambda x: str(x)[0:10])
older_tweets


# In[ ]:


#Merge dataframes into a big one and sort it by date
data = [df,older_tweets]
tweets = pd.concat(data)
tweets = tweets.drop_duplicates()
tweets = tweets.sort_values(by='date')
tweets


# ### Tracking keywords:
# We count the occurence of all keywords appearing in tweets and track it over time and position (if it is possible). 
# 

# - Over time:

# In[ ]:


#Module for regular expression matching operations
import re
#Function that found our word in tweets
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return 1
    return 0
#Function to delete emojis
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

data_kw = pd.DataFrame()
for k in keywords_trad:
    data_kw[k] = tweets['text'].apply(lambda tweet: word_in_text(k, tweet))
data_kw['date'] = tweets['date']
data_kw_date = data_kw.groupby('date').sum()
data_kw_date


# We can now apply a cumulative sum to see the evolution of occurence of each keywords over time.
# 

# In[ ]:


#Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

#Count and time series
count = data_kw_date.cumsum()
time = ['day'+'{}'.format(i+1) for i in range(len(data_kw_date.index))]

#Plot
plt.figure(figsize = (7,5))
plt.style.use('seaborn-paper')
col=iter(cm.rainbow(np.linspace(0,1,len(data_kw_date.columns))))
for k in list(data_kw_date.columns):
    c = next(col)
    plt.plot(time,count[k],'o-',color=c, label='{}'.format(k))
plt.xlabel("Time", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.tick_params(axis = 'both', labelsize = 12)
plt.legend(fontsize=12)
plt.show()


# - Most keywords used

# In[ ]:


#Find which keywords are more used and more relevent for our reasearch
data_count = pd.DataFrame({'Count': data_kw_date.sum(axis = 0, skipna = True)}).sort_values(by=['Count'])
data_count.reset_index(level=0, inplace=True)
data_count = data_count.rename(columns={"index": "Keywords"})

#Bar chart
plt.figure(figsize=(10, 10))
data_count.plot.barh(x='Keywords',y='Count',color='deepskyblue')
plt.xlabel('Occurence', fontsize=15)
plt.ylabel('keywords', fontsize=15)
plt.legend(fontsize=12)
plt.tick_params(axis = 'both', labelsize = 12)
plt.title("Count of keywords founded over {} Tweets".format(len(tweets['text'])), fontsize = 20)
plt.show()


# - Over location

# In[ ]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install gmplot')
from gmplot import gmplot
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# In[ ]:


data_kw['location'] = tweets['location']
data_kw_location = data_kw.groupby('location').sum()
for i in range(len(data_kw_location.index.values)):
    data_kw_location.index.values[i] = deEmojify(data_kw_location.index.values[i])
data_kw_location


# In[ ]:


geolocator = Nominatim(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36")
# Go through all tweets and add locations to 'coordinates' dictionary
coordinates = {'latitude': [], 'longitude': []}
#Tracking Ebola keyword
ebola = data_kw_location['ebola']
ebola = ebola[ebola.values > 0]
for count,loc in enumerate(ebola.index):
    try:
        location = geolocator.geocode(loc)
        # If coordinates are found for location
        if location:
            coordinates['latitude'].append(location.latitude)
            coordinates['longitude'].append(location.longitude)
            
    # If too many connection requests
    except:
        pass


# In[ ]:


# Instantiate and center a GoogleMapPlotter object to show our map
gmap = gmplot.GoogleMapPlotter(30, 0, 3)
gmap.heatmap(coordinates['latitude'], coordinates['longitude'], radius=20)
gmap.draw("map_ebola.html")


# In[ ]:


#Heatmap of regon where the keyword 'ebola' was used
from IPython.core.display import Image
Image("../input/map-image-ebola/map.jpg")


# ## Conclusions and Perspectives:
# 
# In conclusion, we build a tool that can track the usage of disease-related keywords, in this case with French tweets about Ebola. Despite the fact that we were limited in the amount of data due to API free version, we found most used keywords, track it over time and localize it. To get more data, we saved a previous version of our dataframe and add it to the new one. However, the professional version of the API can provide more data. 
# 
# Moreover, to improve this process, we can build the same thing in real time by using streaming tool provided by Twython. We notice that we need to apply more filter to target only relevant tweets about ebola. For example, we can searching only place where we talk more about Ebola. We also remark that not all tweets have a location, which reduces the information that can be drawn from the data.
# 
# 
# In order to make it a MVP that can be marketed, we can make a website or mobile app with a dashboard including all charts related to a specific research like in this case. It can be selled as a web service with subscriptions and each keywords can have a price depending on its relevance.

# In[ ]:




