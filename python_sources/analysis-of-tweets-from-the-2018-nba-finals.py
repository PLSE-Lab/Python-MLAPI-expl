#!/usr/bin/env python
# coding: utf-8

# #Introduction
# 
# With the NBA season coming up soon, I wanted to do a fun analysis on NBA Twitter data. This is the first time I can remember working with Twitter data so this kernel contained a huge learning curve for me. This kernel will use Textblob (https://textblob.readthedocs.io/en/dev/), which I am slightly familiar with, in order to measure the polarity and subjectivity of tweets from Game 3 of the 2018 NBA Finals.

# In[ ]:


import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt


# In[ ]:


#https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
tweets = pd.read_csv('../input/TweetsNBA.csv', encoding = "ISO-8859-1")
tweets = tweets.loc[tweets['lang'] == 'en']


# The next two lines of code help to create new columns in our dataset for polarity and subjectivity of each tweet. Polarity is a number measured between -1 and 1, with -1 being very negative, 0 being neutral, and 1 being very positive. Subjectivity is a number measured between 0 and 1 with 0 being "very objective" and 1 being "very subjective."
# 
# More information: https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

# In[ ]:


#https://textblob.readthedocs.io/en/dev/quickstart.html
#https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.en.sentiments.PatternAnalyzer
tweets['polarity'] = tweets['text'].apply(lambda x: TextBlob(x).polarity)


# In[ ]:


tweets['subjectivity'] = tweets['text'].apply(lambda x: TextBlob(x).subjectivity)


# Let's take a look at the difference in subjectivity between verified and unverifeid Twitter accounts. It appears that verified accounts are slightly more subjective on average than unverified accounts.  With all of this being said, we still do not know if this difference is statistically significant. The issue of statistical significance could be solved by using a potential t-test or other forms of hypothesis testing.

# In[ ]:


Unverified = tweets.loc[tweets['verified'] == False]
Verified = tweets.loc[tweets['verified'] == True]
print('Verified Subjectivity: {}'.format(Verified['subjectivity'].mean()))
print('Unverified Subjectivity: {}'.format(Unverified['subjectivity'].mean()))


# Now let's attempt to make a graph that measures how polarity changes over time. To do this we must change the created_at column to datetime (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html).

# In[ ]:


#https://stackoverflow.com/questions/35595710/splitting-timestamp-column-into-seperate-date-and-time-columns
tweets['created_at'] = pd.to_datetime(tweets['created_at'])


# After changing the created_at column to datetime we can now group the data by minute and see how polarity changes. Very early on in the game there seems to be a big increase in the average polarity which could potentially be because of a dunk by Lebron James. 
# 
# Video of Dunk: https://www.youtube.com/watch?v=oDBLGk12zB4 
# 
# Further Reading: https://www.kaggle.com/xvivancos/eda-tweets-during-cavaliers-vs-warriors

# In[ ]:


#https://stackoverflow.com/questions/16266019/python-pandas-group-datetime-column-into-hour-and-minute-aggregations/32066997
Game3 = tweets.groupby(tweets['created_at'].dt.minute)['polarity'].mean()

plt.figure(figsize = (20,10))
plt.suptitle('Total Game 3 Polarity', fontsize = 24)
plt.xlabel('Time', fontsize = 20)
plt.ylabel("Polarity", fontsize = 20)
plt.plot(Game3)
plt.gcf().autofmt_xdate()
plt.show()


# The next two graphs attempt to look at tweets that contain Cavaliers and Warriors specifically. The goal of the code below is to filter out tweets that contain the text "cav" and "warriors" which corresponds with the Twitter handle of each team.

# In[ ]:


tweets['cavs'] = tweets['text'].str.contains('cav',regex = False, case = False)
Cavs_Tweets = tweets.loc[tweets['cavs'] == True]


# In[ ]:


tweets['warriors'] = tweets['text'].str.contains('warriors',regex = False, case = False)
Warriors_Tweets = tweets.loc[tweets['warriors'] == True]


# Below are the average polarities of each team grouped by time. 

# In[ ]:


Game3_Cavs = Cavs_Tweets.groupby(Cavs_Tweets['created_at'].dt.minute)['polarity'].mean()

plt.figure(figsize = (20,10))
plt.suptitle('Cavs Game 3 Polarity', fontsize = 24)
plt.xlabel('Time', fontsize = 20)
plt.ylabel("Polarity", fontsize = 20)
plt.plot(Game3_Cavs)
plt.gcf().autofmt_xdate()
plt.show()


# In[ ]:


Games3_Warriors = Warriors_Tweets.groupby(Warriors_Tweets['created_at'].dt.minute)['polarity'].mean()

plt.figure(figsize = (20,10))
plt.suptitle('Warriors Game 3 Polarity', fontsize = 24)
plt.xlabel('Time', fontsize = 20)
plt.ylabel("Polarity", fontsize = 20)
plt.plot(Games3_Warriors)
plt.gcf().autofmt_xdate()
plt.show()


# Since I am not very familiar with Twitter data, or text data in general, leave comments down below as to how you would improve this analysis, ideas for future analysis, or if there is anything that I missed.
