#!/usr/bin/env python
# coding: utf-8

# # Visualisation and Sentiment Analysis of Times of India's Coverage 

# #### Author: Tanmay Bansal (January 2, 2018), Cornell University, tb444@cornell.edu

# Times of India is the largest circulated English daily in the world, and also India's largest news website. I decided to visualise and ultimately analyse the various trends that a dataset of 2.6 million TOI headlines reveals! 

# **Dataset by**: Rohit Kulkarni (2017), News Headlines of India 2001-2017 [CSV data file], Retrieved from https://www.kaggle.com/therohk/india-news-publishing-trends-and-cities/data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob


# In[ ]:


data = pd.DataFrame.from_csv('../input/india-news-headlines.csv', index_col=None)


# ## Coverage by Cities 

# ### Volume of coverage per city

# In[ ]:


def label_plot(x, y, title):
    """
    Labels the x-axis, y-axis and title of the given plot
    
    This procedure uses the matplotlib library to label the aforementioned elements of the given plot
    
    Parameter x: The value to be set as the x-label
    Precondition: Value passed should be of String type
    
    Parameter y: The value to be set as the y-label
    Precondition: Value passed should be of String type
    
    Parameter title: The value to be set as the title
    Precondition: Value passed should be of String type
    """
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)


# In[ ]:


data_cities = data[data['headline_category'].str.contains('^city\.[a-z]+$', regex=True)]
data_cities['city_name'] = data_cities.headline_category.str[5:]
city_list = data_cities['city_name'].unique().tolist()

#Bar chart of covrage by cities in descending order
grp_city = data_cities.groupby(['city_name'])['headline_text'].count().nlargest(20)
ts = pd.Series(grp_city)
ts.plot(kind='bar', figsize=(20,12))
label_plot('City Name', 'Number of recorded articles', 'Number of articles per city')
plt.show()


# Not so surprisingly, Mumbai and Delhi lead the list. Following these two cities, the decrease is fairly uniform until the sharp drop in coverage from Chennai to Jaipur.
# I wanted to check if the coverage of a particular city has a relationship with its population density, and ended up with the following chart:

# ### Year-wise coverage for cities

# In[ ]:


data['publish_date'] = pd.to_datetime(data['publish_date'])
data['year'] = data['publish_date'].apply(lambda x: (x.microsecond)//10)
data = data.drop('publish_date', axis = 1)
data = data.reset_index(drop=True)
data_cities = data_cities.reset_index(drop=True)
data_cities['year'] = data['year']
data_cities = data_cities.drop('headline_category', axis = 1)


# In[ ]:


data_cities2 = data_cities
data_cities2['headline_text'] = 1


# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
grp = data_cities2.groupby(['year', 'city_name']).sum()['headline_text'].unstack().plot(ax=ax)
label_plot('Year', 'Number of articles', 'Year-wise coverage for cities')
plt.show()


# This is obviously too cumbersome a way to find any pattern at all, and the dataset is clearly not large enough to account for all the cities. I, therefore, limited my graph to the big metropolitan cities: Delhi, Mumbai, Kolkata, Bengaluru, Pune, Hyderabad, and Chennai.

# In[ ]:


data_cities_del = data_cities[data_cities.city_name == 'delhi']
data_cities_mum = data_cities[data_cities.city_name == 'mumbai']
data_cities_kol = data_cities[data_cities.city_name == 'kolkata']
data_cities_ben = data_cities[data_cities.city_name == 'bengaluru']
data_cities_pun = data_cities[data_cities.city_name == 'pune']
data_cities_hyd = data_cities[data_cities.city_name == 'hyderabad']
data_cities_che = data_cities[data_cities.city_name == 'chennai']


# In[ ]:


frames_metros = [data_cities_mum, data_cities_del, data_cities_che, data_cities_ben, data_cities_kol, data_cities_hyd, data_cities_pun]
data_cities_metros = pd.concat(frames_metros)


# In[ ]:


fig=plt.figure()

a = fig.add_subplot(111,label="1")
b = fig.add_subplot(111,label="2", frame_on = False)

#Coverage relevant to cities per year
ts = pd.Series(data_cities.groupby(['year'])['headline_text'].count())
ts.plot(kind='bar', figsize=(20,10), ax=a, color="black", title="Number of articles per year and per city")
a.set_ylabel('Number of Articles (Per year)', size =16)
a.set_xlabel("")

ts2 = data_cities_metros.groupby(['year', 'city_name']).sum()['headline_text'].unstack()
ts2.plot(ax=b)
b.set_xticks([])
b.set_ylabel('Number of Articles (Per city)', size =16)
b.yaxis.tick_right()
b.set_xlabel('Year') 

a.get_yaxis().set_label_coords(-.05,0.5)
b.get_yaxis().set_label_coords(1.05,0.5)
b.get_xaxis().set_label_coords(0.5, -0.07)
plt.tight_layout()
plt.show()


# The graph for the cities was not uniform and hence I decided to overlap on it a graph for the number of city-based articles by year. This way, we can disregard the apparent non-uniformity in city-based reporting over the years. 
# 
# While the plot is open to interpretation, one interesting trend is very conspicous: there is a major spike in reporting for Mumbai in 2009. The possible reason for this is the 26/11 Terrorist Attacks in Mumbai towards the end of 2008 that shook the world. There was a major coverage of the aftermath of the attacks in the remainder of 2008 and throughout the next year (i.e. 2009).

# ## Topic-wise Coverage

# In[ ]:


#Original Author: https://www.kaggle.com/therohk ; Modifications by Tanmay Bansal

data_non_cities = data[~data['headline_category'].str.contains('city', regex=False)]

non_cities = data_non_cities.groupby(['headline_category'])['headline_text'].count()
non_cities = non_cities.drop('unknown')
non_cities = non_cities.drop('removed')
non_cities = non_cities.drop('top-stories')
non_cities['tech'] += non_cities['tech.tech-news']
non_cities = non_cities.drop('tech.tech-news')
non_cities = non_cities.nlargest(30)
ts = pd.Series(non_cities)
ts.plot(kind='bar', figsize=(20,10), color='green')
label_plot('Category', 'Number of articles', 'Top 30 Non-city Categories')

plt.show()


# It's no surprise that news concerning India is the most published since the publication is based in India. Interestingly, Bollywood is the third-most covered topic by TOI. In fact, it's eye-opening how Bollywood is given more coverage than every other topic except for Indian Business.
# 
# Of news concentrated on other countries, news about the US is the most prevalent followed by that about Pakistan. It is also interesting to note that the ICC World Cup 2015 (Cricket) was more talked about than anything related to football in all the years combined.

# ## Sentiment Analysis

# In[ ]:


def analyze_polarity(headline):
    """
    Returns a value based on the polarity of the given text using TextBlob
    
    This function uses the TextBlob library to perform sentiment analysis and analyse the polarity of the
    passed headline. It returns '1' if the sentiment is positive, '-1' if the sentiment is negative, and
    '0' if the sentiment is neutral.
    
    Parameter headline: The text whose sentiment is to be determined
    Precondition: It should be a non-empty value of type String
    """
    result = TextBlob(headline)
    if result.sentiment.polarity > 0:
        return 1
    elif result.sentiment.polarity == 0:
        return 0
    else:
        return -1


# In[ ]:


data['Result'] = np.array([analyze_polarity(headline) for headline in data['headline_text']])


# In[ ]:


positive_headlines = [ head for index, head in enumerate(data['headline_text']) if data['Result'][index] > 0]
unbiased_headlines = [ head for index, head in enumerate(data['headline_text']) if data['Result'][index] == 0]
negative_headlines = [ head for index, head in enumerate(data['headline_text']) if data['Result'][index] < 0]


# ### Overall Sentiment Distribution

# In[ ]:


o_pos = len(positive_headlines)*100/len(data['headline_text'])
o_neg = len(negative_headlines)*100/len(data['headline_text'])
o_un = len(unbiased_headlines)*100/len(data['headline_text'])


# In[ ]:


print('Positive Headlines: ' + str(o_pos) + '\nNegative Headlines: ' + str(o_neg) + '\nUnbiased Headlines: ' + str(o_un))


# As is evident, most of the headlines published by Times of India are unbiased in their sentence structure. The ratio of the percentage of positive to negative headlines is **1.5**. This is not significantly large and can be ignored. 

# ### Sentiment Distribution by Major National Political Parties

# The following are some major national political parties in India:
# 
# 1. Bhartiya Janata Party (BJP)
# 2. Indian National Congress (Congress)
# 3. Bahujan Samaj Party (BSP)

# In[ ]:


def party_data(party_name):
    """
    Returns a dataset that contains data relevant to the given party name
    
    The procedure looks for the given party name in the headlines of the entire dataset
    and filters the dataset accordingly to create a new dataset
    
    Parameter party_name: Name of the party relevant to which the dataset is to be created
    Precondition: Non-empty String
    """
    return data[data['headline_text'].str.contains(party_name)]


# In[ ]:


frames_bjp = [party_data('bjp'), party_data('BJP')]
data_bjp = pd.concat(frames_bjp)

frames_congress = [party_data('congress'), party_data('Congress')]
data_congress = pd.concat(frames_congress)

frames_bsp = [party_data('bsp'), party_data('BSP')]
data_bsp = pd.concat(frames_bsp)


# In[ ]:


def result_ratio(dataset):
    """
    Returns the ratio of positive sentiment with negative sentiment of all the relevant reported headlines
    
    The given dataset is run through and 3 groups are created: 
    (1) - Positive Sentiment
    (0) - Unbiased Sentiment
    (-1) - Negative Sentiment
    The ratio of the number of the reports with a positive sentiment with number of reports with a negative
    sentiment is then returned.
    
    Parameter dataset: The dataset to segregate on the basis of sentiments
    Precondition: Dataset with column 'Result' that may only contain '1', '0' or '-1' values based on the
    aforementioned criteria
    """
    grouped_data = dataset.groupby(['Result'])['headline_text'].count()
    return grouped_data[1]/grouped_data[-1]


# In[ ]:


print('1. ' + str(result_ratio(data_bjp)) + '\n2. ' + str(result_ratio(data_congress)) + '\n3. ' + str(result_ratio(data_bsp)))


# In[ ]:


print('1. ' + str(len(data_bjp)) + '\n2. ' + str(len(data_congress)) + '\n3. ' + str(len(data_bsp)))


# As is evident, the **ratios of positive sentiment to negative sentiment for different major parties are fairly the same**. While the positive sentiment for BJP is a bit higher, the difference is not significant and hence negligible. Another important conclusion made was that **atleast 70.30% of the 2.6 million headlines published by TOI are unbiased in nature (as far as language is concerned**. These are remarkable statistics - kudos to the Times of India team!
# 
# Note: As was pointed out by [Aleksey Bilogur](http://www.kaggle.com/residentmario), large dailies like Times of India are fairly unbiased in language. However, bias, if it exists, is in the form of sources and coverage - which is bit more challenging to evaluate.

# #### Future Steps:
# 
# In future, I plan to compare this percentage to that of other large dailies from across the globe. It would be an interesting comparison to evaluate!

# ### References:

# 1. Rohit Kulkarni's Kernel [https://www.kaggle.com/therohk/india-news-publishing-trends-and-cities]
# 2. Pandas (0.22.0) Documentation [https://pandas.pydata.org/pandas-docs/stable/index.html]
# 3. Matplotlib (2.1.1) Documentation [https://matplotlib.org/index.html]
# 4. TextBlob (0.15.0) Documentation [https://textblob.readthedocs.io/en/dev/]
# 5. Rodolfo Ferro, Sentiment Analysis on Donald Trump's Tweets [https://dev.to/rodolfoferro/sentiment-analysis-on-trumpss-tweets-using-python-]
