#!/usr/bin/env python
# coding: utf-8

# # How to create diverging stacked bar charts for sentiment analysis in Plotly!
# 
# **#Plotly #Python #diverging-stacked-bar-chart #dentiment-analysis**
# 
# **Problem**: I wanted to visualize the results from a sentiment analysis where the data was from a questionaire with a likert scale as the answer.
# 
# **Solution**: I found that in [Plotly](https://plot.ly/python/), you can have make a diverging stacked bar chart, where the sentiment can be plotted as quantities using bars that are based on the categories of the answers.
# 
# In this example I plotted the sentiment from a dataset of [Twitter Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). I visualized which airlines have the largest proportions of negative reviews, but also show the proportions of reviews are negative versus positive.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
Tweets = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")


# In[ ]:


import numpy as np
import plotly.graph_objects as go
import colorlover as cl


# In[ ]:


Tweets.info()


# In[ ]:


Tweets.sample(3)


# In[ ]:


Tweets.airline.value_counts()


# In[ ]:


Tweets.airline_sentiment.value_counts()


# In[ ]:


Tweets[Tweets.airline == 'American'].airline_sentiment.value_counts()


# In[ ]:


category_order = [
    'negative',
    'neutral',
    'positive'
]

# rearrange the data into the format we desire to only show airline and its sentiment proportions
Tweets_airline = pd.pivot_table(
    Tweets,
    index = 'airline',
    columns = 'airline_sentiment',
    values = 'tweet_id',
    aggfunc = 'count'
)

# reorder the columns as desired above
Tweets_airline = Tweets_airline[category_order]

# make specific columns to represent undesired or negative answers
Tweets_airline.negative = Tweets_airline.negative * -1


# In[ ]:


Tweets_airline


# In[ ]:


# sort by desired column
Tweets_airline = Tweets_airline.sort_values(by='negative', ascending = False)

fig = go.Figure()

for column in Tweets_airline.columns:
    fig.add_trace(go.Bar(
        x = Tweets_airline[column],
        y = Tweets_airline.index,
        name = column,
        orientation = 'h',
        marker_color = cl.scales[str(len(category_order))]['div']['RdYlGn'][category_order.index(column)],
    ))

fig.update_layout(
    barmode = 'relative',
    title = 'Twitter Sentiment Analysis of US Airlines'
)
fig.show()


# If you liked my work, please remember to upvote the notebook!
# 
# If you have any questions, feel free to comment down below or [message me on LinkedIn](https://www.linkedin.com/in/jaydeep-mistry/).
