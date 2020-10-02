#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Trying to perdict the price of a certain stock has always been a challenging problem. This kernel tries to preict the behavior of a stock from news headlines. The stock price and related news for any trading company can be taken from [FINVIZ](http://finviz.com). 
# I have used [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for the analysis. 'Beautiful Soup is a python library for pulling data out of HTML and XML files.'

# In[ ]:


# Import libraries
from bs4 import BeautifulSoup
import pandas as pd
import os

html_pages = {}

# The dataset includes webpages from different days for different companies.
for page_name in os.listdir('../input'):
    page_path = f'../input/{page_name}'
    page_file = open(page_path, 'r')
    html_data = BeautifulSoup(page_file)
    # The .html file contains a 'news-table' section that holds all the news headlines
    html_page = html_data.find(id="news-table")
    # Add the page to our dictionary
    html_pages[page_name] = html_page


# You can display the html pages that we have extracted so far.

# In[ ]:


#html_pages


# Now we can explore the material of these headlines. What's inside them!
# You may want to check these links for a better understanding of the elements of a HTML page.
# 
# [a: The Anchor element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a)
# 
# [td: The Table Data Cell element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/td)

# In[ ]:


amzn = html_pages['amzn_24jul.html']
# All the headlines in the .html file are stored as table rows and are tagged as <tr>
amzn_tr = amzn.findAll('tr') #Remember this is a beautiful soup now. So, we can directly find <tr> tag.

for i, tr in enumerate(amzn_tr):
    # <a> in these .html pages contain the news headlines.
    hyperlinks = tr.a.get_text()
    # <td> in these .html page contain the time stamp.
    headlines = tr.td.get_text()
    print(f'{i}:')
    print(hyperlinks)
    print(headlines)
    
    #if i == 5:
    #    break


# Let's try the same with all the webpages in the dataset.

# In[ ]:


news = []

for page_name, news_table in html_pages.items():
    for x in news_table.findAll('tr'):
        text = x.get_text()
        headline = x.a.get_text()
        # Split date-time, because some elements just have time, while others have time and date as well.
        date_td = x.td.text.split()
        # If only time
        if len(date_td) == 1:
            time = date_td[0]
        # If both time and Date
        else:
            date = date_td[0]
            time = date_td[1]
        # Separate the stock symbol    
        stock_symbol = page_name.split('_')[0]
        # Join everything
        news.append([stock_symbol, date, time, headline])
        
news[:5] # print first five news headlines


# I have used VADER to perform sentiment analysis, because it doesn't require any training data. 
# We can add new words to the VADER Lexicon text file which are specific to our problem. In our case, since we are working with stock market, adding words like 'falls', 'trouble', etc. with an associated positive or negative score should be good practice. 

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

new_words = {
    'crushes': 10,
    'beats': 5,
    'misses': -5,
    'trouble': -10,
    'falls': -100
}

analyser = SentimentIntensityAnalyzer()

analyser.lexicon.update(new_words)


# Now we have updated the lexicons in the VADER model, we can try to evaluate the polarity of the news headlines.

# In[ ]:


columns = ['stock_symbol', 'date', 'time', 'headline']
# Load list of the news into a pandas DataFrame
news_headlines = pd.DataFrame(news, columns = columns)
# Go through each headline and evaluate it's polarity. 
senti_scores = [analyser.polarity_scores(headline) for headline in news_headlines.headline.values]
# Convert it into a DataFrame for easy joining.
sentiment = pd.DataFrame(senti_scores)
# Join headline and its sentiment score.
news_headlines = pd.concat([news_headlines, sentiment], axis = 1)
# Date column is converted from string to datetime
news_headlines['date'] = pd.to_datetime(news_headlines.date).dt.date
news_headlines.head(10)


# Try to plot the sentiment scores

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')
# Group by Date and Company
mean_c = news_headlines.groupby(['date', 'stock_symbol']).mean()

mean_c = mean_c.unstack(level=1)
mean_c = mean_c.xs('compound', axis=1)

mean_c.plot.bar();


# Seems like Tesla had some trouble on November 22. Further exploration reveals that there were only 5 headlines on that day. And 2 of them were almost similar.
# We can further check and try to remove similar headlines from the dataset.

# In[ ]:


# Number of news headlines
dirty_news = news_headlines.headline.count()
# Drop duplicates headlines
cleaned_news = news_headlines.drop_duplicates(['stock_symbol', 'headline'])
# Number of headlines after dropping 
clean_news = cleaned_news.headline.count()

print(dirty_news)
print(clean_news)


# To finish the project, lets try to explore just one stock on any given day (just a single day), cause this is what one should be interested in.
# And don't forget to use 'cleaned_news'.

# In[ ]:


single_day = cleaned_news.set_index(['stock_symbol', 'date'])
single_day = single_day.loc['amzn']

# Set day to January of 2019
single_day = single_day.loc['2019-07-24']

# Convert the datetime string to just the time since it is just one day.
single_day['time'] = pd.to_datetime(single_day['time'])
single_day['time'] = single_day.time.dt.time

# Set the index to time and sort by it
single_day.set_index('time', inplace=True)
single_day=single_day.sort_index(ascending=True)
single_day.head()


# We can now visualize the sentiment scores using a plot.

# In[ ]:


# Drop the columns which are useless
plot_day = single_day.drop(['headline', 'compound'], axis=1)
# Give names to the sentiments
plot_day.columns = ['negative', 'positive', 'neutral']
# Plot a stacked bar chart
plot_day.plot.bar(stacked = True, 
                  figsize=(10, 6), 
                  title = "Sentiment Analysis for AMAZON on 2019-07-24", 
                  color = ["red", "blue", "green"])
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.ylabel("scores");


# Seems like news regarding Amazon were mostly 'positive' or 'neutral' on July 24, 2019. 
# Let's check the stock price for the same day.
# ![AMZN.PNG](attachment:AMZN.PNG)
