#!/usr/bin/env python
# coding: utf-8

# # Executive Summary
# 
# We analyze seven days of data from Twitter regarding customer sentiment for 6 US airlines: American, Delta, Southwest Airlines, United, US Airways, and Virgin America.  From this data we evaluate how customer sentiment evolves over a week.  We also evaluate how customers tweet sentiment by airline, and the efficiency that others retweet that sentiment.  We find that for most airlines, more tweets are posted on Mondays and decrease over the rest of the week.  In contrast, American recieves most tweets on Sundays and Mondays, and relatively few posted on other days.  Additionally, we find that United receives the most negative tweets by far.  Additionally, we find that the liklihood of a tweet being retweeted is dependent on the sentiment and airline.  Negative tweets about United, American, and US Airways are more likely to be retweeted than positive tweets.  In contrast, positive tweets about Southwest, Virgin America, and Delta are more likely to be retweeted than negetive tweets. We hypothesize that public opinion is driving this bias.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as pylab
import seaborn as sb
pylab.rcParams['figure.figsize'] = 10, 8


# # Reading the Data
# 
# The data is read from a file created by Crowdflower and posted to Kaggle.  The data can be found here: https://www.kaggle.com/crowdflower/twitter-airline-sentiment/version/1.  The tweet ID ('tweet_id') is used as dataframe's index.  Afterward, we show a few columns of the created dataframe as a table.

# In[ ]:


data_file = '../input/Tweets.csv'
parser = lambda x: dt.datetime.strptime(x[:-6], '%Y-%m-%d %H:%M:%S')
tweets = pd.read_csv(data_file, index_col = 'tweet_id',
                     parse_dates=[12], date_parser = parser)
pd.options.display.max_rows = 8
tweets[['airline_sentiment','airline', 'retweet_count', 
        'text', 'tweet_created']]


# # Results
# 
# ## Tweets over the week
# 
# Below we show the distribution of tweets by day of week for each airline (color), and sentiment.  From top to bottom the different panels show neutral sentiment, positive sentiment, and negative sentiment.  Day of week 0 is Monday, while day of week 6 is a Sunday.
# 
# For all airlines but American and Southwest, the number of complaints peak on Sunday and slowly decline throughout the week, while the number of positive comments peak on Tuesday.  In contrast, tweets peak on Monday for American Airlines, but the number of tweets experience a sharp decline in the middle of the week.  For Southwest, the number of compaints tend to peak on Tuesdays and Saturdays.

# In[ ]:


tweets['dow'] = tweets.tweet_created.dt.dayofweek

g = sb.FacetGrid(tweets, row = 'airline_sentiment', 
                 hue = 'airline', legend_out = True,
                 aspect = 4, size = 2.5)
g.map(sb.distplot, 'dow', hist = False)
g.add_legend()
g.axes.flat[0].set_xlim(0,6)
g.axes.flat[2].set_xlabel('Day of Week')


# ## Tweet sentiment by airline
# 
# In the following figure, we study how people tweet and retweet about different airlines.  From top to bottom, we show the number of tweets for each airline and sentiment, the number of retweets for each airline and sentiment, and the ratio of retweets to tweets for each airline and sentiment.  Sentiment is colored red, blue, or green to respectively represent negative, neutral, or positive sentiment.  Twitter users love to hate all of the airlines except Virgin America.  United is clearly the most reviled of all airlines.  Interestingly, the retweet efficiency is airline dependent.  Tweets about United, American, and US Airways are more likely to be retweeted if they are negative, while tweets about Southwest, Delta, and Virgin America are more likely to be retweeted if they are positive.  This general trend anecdotally reflects this author's personal opinion of airline quality. We speculate that Twitter users are retweeting comments that they personally agree with.  Consequently, we suggest that the retweet efficiency could be a useful metric for determining public opinion.

# In[ ]:


groups = tweets.groupby([tweets.airline, 
                         tweets.airline_sentiment])

retweet_table = groups.retweet_count.apply(sum)
my_colors = list(islice(cycle(['r', 'b', 'g']), 
                        None, len(retweet_table)))
fig, ax = plt.subplots(3, sharex = True)
groups.count().name.plot(kind = 'bar', color = 
                         my_colors, title = 
                         '# of Tweets', ax = ax[0])

retweet_table.plot(kind = 'bar', color= my_colors, 
                   title = '# of Retweets', ax = ax[1])
(retweet_table/groups.count().name).plot(
    kind = 'bar', color = my_colors, 
    title = 'Retweet Efficiency', ax = ax[2])

