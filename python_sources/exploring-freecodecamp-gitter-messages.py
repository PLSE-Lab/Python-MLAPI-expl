#!/usr/bin/env python
# coding: utf-8

# ## Exploring freeCodeCamp Gitter Messages for DevNet
# 
# DevNet is the developer evangelist organization for Cisco - and we care about what developers are interested in.  With the open sourcing of all the messages from the past 3 years (nearly 5M messages) from freeCodeCamp's Gitter message board, we thought we'd take a look to garner what is relevant to our core audience and see what trends are taking place in technology that may not yet be public knowledge!  
# 
# Credit to Aleksey Bilogur as this notebook was forked from work consolidating the data released by freeCodeCamp.  
# 
# Let's get started!
# 
# ## Munging the data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
chat = pd.read_csv("../input/freecodecamp_casual_chatroom.csv", index_col=0)


# In[ ]:


fig_kwargs = {'figsize': (12, 6), 'fontsize': 16}


# Let's take a look at the first 3 rows to see what the data in our dataframe will look like.

# In[ ]:


chat = chat.iloc[:, 1:]
chat.head(3)


# For this notebook I will used the consolidated chat log, `freecodecamp_casual_chatroom.csv`. I generated this consolidated `csv` file by merging three distinct JSON files. The root files had slight overlap (a few days each), leading to a few duplicated messages being present in the unified file. Since this dataset handily comes equipped with an extremely accurate timestamp in the `sent` field (down to `ms`), we can see the exact overlapping records using the `pandas.Series.duplicated` method:

# In[ ]:


chat.sent.duplicated().sum() / len(chat)


# About 0.7 percent of messages were duplicated. We'll drop them before going any further.

# In[ ]:


chat = chat.drop_duplicates()


# In[ ]:


chat = chat.loc[~chat.sent.duplicated()]


# With that out of the way, let's get exploring!

# ## Exploration
# 
# First, lets just do a simple filter for any message containing "cisco" (case insensitive) with a simple regex match.

# In[ ]:


import re
regExPattern = re.compile(r'.*\bcisco\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]


# Let's take a look at the results of our filter to confirm our RegEx match was correct (and its helpful to see the messages being exchanged relevant to the search terms - especially for things like "amazon" which end up parsing URL's for shopping links vs cloud platform discussions).

# In[ ]:


pd.set_option('display.max_colwidth', -1)
searchFrame[['fromUser.displayName','sent','text']].tail(10)


# I wonder how many messages have been sent containing the word "Cisco"?

# In[ ]:


len(searchFrame)


# Not bad.  What % of total messages is that?

# In[ ]:


len(searchFrame) / len(chat) * 100


# Ok, so .004% of messages sent in freeCodeCamp are related to Cisco. 

# In[ ]:


len(chat)


# But there are over 5M messages related to learning code concepts - maybe not a large percentage are vendor specific.  Let's see if there is any trend over time of discussing Cisco.

# In[ ]:


(pd.to_datetime(searchFrame.sent)
     .to_frame()
     .set_index('sent')
     .assign(n=0)
     .resample('M')
     .count()
     .plot.line(**fig_kwargs, title="Postings About Cisco over Time"))


# Excellent! We are on a largely positive trend.  Lets confirm this isn't a confirmation bias that it doesn't match a general trend in Gitter usage.

# In[ ]:


(pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(n=0)
     .resample('M')
     .count()
     .plot.line(**fig_kwargs, title="Number of freeCodeCamp Messages over Time"))


# Oops - looks like this is a universal trend across Gitter of increased usage.  We can also see freeCodeCamp is dropping off significantly in usage!  To get a more accurate view of trending, we should normalize this data as % of messages per month that match our criteria.

# In[ ]:


searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
combinedTimeSeries.head(10)


# Ok, so we put the total of all the messages containing our search term in a monthly period in column 'n', and a total of all messages during that period in column 'p'.  We then stored the percent of all messages matching our search term in the 'pcnt' column.  **Now lets take a look at the trend!**

# In[ ]:


(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"Cisco\" over Time"))


# **Great! Now we have some normalized data to evaluate a trend against.**  It looks like "Cisco" as a term has shown up increasingly as larger part of the conversation held on Gitter month over month (even though it is nearly 3 times increase from its baseline in 2015, the percentage is low enough that one or two more messages could affect the total). 
# 
# I wonder what other vendors look like during this same time period?  Let's take a look at AWS trends:

# In[ ]:


regExPattern = re.compile(r'.*\baws\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"AWS\" over Time"))


# Hmm - not much of a strong trend in any specific direction.  I wonder what happened in Aug/Sept of 2017 to more than double the amount of people discussing AWS?  Let's take a peek:

# In[ ]:


searchFrame = searchFrame.assign(timestamp = lambda x: pd.to_datetime(x.sent))
filteredFrames = searchFrame[(searchFrame.timestamp >= '2017-08-01') & (searchFrame.timestamp <= '2017-09-01')]
filteredFrames[['fromUser.displayName','timestamp','text']].head(30)


# In[ ]:


#Number of records in our filtered data
len(filteredFrames)


# In[ ]:


#Confirm matches number of records in our graphed time-series data
combinedTimeSeries.loc['2017-08-31'].n


# Interestingly, there are a couple of things to note.  First, nothing that interesting was discussed (though we didn't extract the context from this duration which we easily could), but AWS is generally used as a term for answering general questions (such as what appears to be a dialog about what is serverless, and AWS Lambda is made reference to) but not necessarily asking for help on how to use/learn AWS services.  Second thing we notice is that  there aren't that many messages in this time frame.  Meaning even in a large data set of over 5 million messages, we're seeing that ~180 messages is a two-fold increase.  Let's see if there are more messages in a monthly time frame for a term like 'javascript'.

# In[ ]:


regExPattern = re.compile(r'.*\bjavascript\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
searchFrame = searchFrame.assign(timestamp = lambda x: pd.to_datetime(x.sent))
filteredFrames = searchFrame[(searchFrame.timestamp >= '2017-08-01') & (searchFrame.timestamp <= '2017-09-01')]
filteredFrames[['fromUser.displayName','timestamp','text']].tail(5)


# And how many messages in the same time-period sampled above for AWS, is Javascript mentioned?

# In[ ]:


len(filteredFrames)


# While we're at it, let's take a look at the Javascript trend.

# In[ ]:


combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"Javascript\" over Time"))


# Wow - Javascript is in decline for new learners?  I'm biased, but maybe that correlates to the decline in activity at freeCodeCamp ;)  
# 
# However, .75 to 1.75% of all messages containing a single term is a pretty huge percentage. Imagine that conversation with anyone not in the tech world, let alone someone learning to code.  I wouldn't want my instructor to say "javascript" 1 out of 50 sentences even when discussing syntax.  Ok, so we've found data about vendors is relatively sparse, but data about languages is pretty rich, and has solid trending behind it.  What about other languages?
# 
# 

# In[ ]:


regExPattern = re.compile(r'.*\bpython\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"Python\" over Time"))


# Python is increasing!  So even though I'm partial to javascript, I'd have to say Python is on the rise with new coders who are starting to learn.  

# 
# ## Conclusion
# 
# Obviously all kinds of trends/terms can be identified from these sets and hopefully you'll expand on these examples to dig deeper.  For targeting developer oriented marketing outcomes, it could also be valuable to understand the network effects taking place - which messages have the most impact, who drives the most conversation, and which topics are most engaging.  Certainly some social graphing is in order!
