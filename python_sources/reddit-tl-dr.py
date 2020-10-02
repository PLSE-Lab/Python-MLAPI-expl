#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This is my first attempt using Kaggle's notebook implementation. The goal of this notebook
is to experiment with pandas and matplotlib. Typically, I would use R to this type of analysis
and visualization. 

The original goal was to explore the relationship between the length (in words) of a post and 
up, down and total votes recieved by a comment. As you will see below, the down votes were not
populated in this data feed. Instead I will focus on the relationship between length of the 
comments, their scores and the precieved length of the original post/link.
"""
from pprint import pprint as pp
import pandas
import numpy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sqlalchemy import create_engine
from scipy.interpolate import UnivariateSpline as spline

# Connect to the Reddit database and retrieve SQLite version
import sqlite3
reddit_conn = sqlite3.connect("../input/database.sqlite")
with reddit_conn:
    reddit_cursor = reddit_conn.cursor()
    reddit_cursor.execute('select sqlite_version()')
    version = reddit_cursor.fetchone()
    print("SQLite Version {}".format(version[0]))


# In[ ]:


# Pull sample records from dataset
with reddit_conn:
    reddit_cursor = reddit_conn.cursor()
    reddit_cursor.execute('select * from May2015 Limit 3')
    sample_records = reddit_cursor.fetchall()
    for rec in sample_records:
        print(rec)


# In[ ]:


# Switching to sqlalchemy and pandas to import data frame.
reddit_engine = create_engine("sqlite:///input/database.sqlite")
print("SQLalchemy Engine is connected to {}.".format(reddit_engine.url))


# In[ ]:


# Explore the values of the score and the up/down votes.
reddit_votes = pandas.read_sql_query('select '
                                        'sum(ups) as total_ups '
                                        ',sum(downs) as total_downs '
                                        ',sum(score) as total_score '
                                        'from May2015 '
                                        'where score_hidden = 0 '
                                        , reddit_conn)
reddit_votes.head()
# This results shows us that there are no values entered for the down votes as such
# the scores appears to always be equal to the up votes. Either this field was suppressed
# during the data extract or the API call never returns the down votes.


# In[ ]:


# Find most common subreddits
reddit_subreddit_pop = pandas.read_sql_query('select '
                                             'subreddit_id '
                                             ',count(*) as comments '
                                             'from May2015 '
                                             'group by subreddit_id '
                                             , reddit_conn)
reddit_subreddit_pop.head()


# In[ ]:


# How many subreddits were there
reddit_subreddit_pop.shape


# In[ ]:


top_n = 10
top_subreddits = reddit_subreddit_pop.sort_values(['comments'], ascending=False).head(top_n)
print('Total comments in top {} subreddits: {}'.format(top_n, sum(top_subreddits.comments)))
top_subreddit_list = top_subreddits.subreddit_id.tolist()
top_subreddits


# In[ ]:


# Get word count and score for each comment.
sql_query_words = ''.join(['select '
                          ,'subreddit_id '
                          ,',link_id '
                          ,',(length(body) - length(replace(replace(body,"\n",""), " ", "")) + 1) as words '
                          ,',score '
                          ,'from May2015 '
                          ,'where score_hidden = 0 '
                          ,'and subreddit_id in ("{}") '.format('","'.join(top_subreddit_list))
                          ])
print(sql_query_words)
reddit_word_cnt = pandas.read_sql_query(sql_query_words, reddit_conn)
reddit_word_cnt.head()


# In[ ]:


reddit_word_cnt.describe()


# In[ ]:


link_agg = reddit_word_cnt.groupby(['subreddit_id', 'link_id'], as_index=False).mean()
link_agg.head()


# In[ ]:


#Get tl;dr count for each link.
# Get word count and score for each comment.
sql_query_tldr = ''.join(['select '
                           ,'subreddit_id '
                           ,',link_id '
                           ,',count(*) as tldr_cnt '
                           ,'from May2015 '
                           ,'where score_hidden = 0 '
                           ,'and body GLOB "*tl;dr*" '
                           ,'and subreddit_id in ("{}") '.format('","'.join(top_subreddit_list))
                           ,'group by subreddit_id, link_id '
                          ])
print(sql_query_tldr)
reddit_tldr_cnt = pandas.read_sql_query(sql_query_tldr, reddit_conn)
reddit_tldr_cnt.head()


# In[ ]:


reddit_tldr_cnt.describe()


# In[ ]:


link_data = pandas.merge(link_agg
                         ,reddit_tldr_cnt
                         ,on=['subreddit_id', 'link_id']
                         ,how='left'
                         ,indicator=True
                        )
link_data['tldr_cnt'].fillna(0, inplace=True)
link_data['words_log'] = numpy.log(link_data['words'])
link_data.loc[link_data['tldr_cnt']>10, 'tldr_cnt'] = 10
link_data.head()


# In[ ]:


link_data.describe()


# In[ ]:


tldr_agg = link_data.groupby('tldr_cnt', as_index=False).agg([numpy.mean, 'count'])
tldr_agg


# In[ ]:


tldr_score_spline = spline(tldr_agg.index, tldr_agg['score', 'mean'], tldr_agg['score', 'count'], s=5000)
tldr_words_spline = spline(tldr_agg.index, tldr_agg['words', 'mean'], tldr_agg['score', 'count'], s=5000)
spline_linspace = numpy.linspace(0,10,100)
plt.plot(spline_linspace, tldr_score_spline(spline_linspace), lw=2)
plt.plot(spline_linspace, tldr_words_spline(spline_linspace))
plt.show()


# In[ ]:




