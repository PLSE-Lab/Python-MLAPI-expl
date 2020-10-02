#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import missingno as msno
import os


# In[ ]:


netflix_data = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv") # load netflix movies and TV Series data into Pandas Dataframe.
netflix_data.shape


# As we can see, Netflix Data has total of 6234 records with 12 different features.

# Let's draw Bar chart and get idea of data for Null values in the records.

# In[ ]:


msno.matrix(netflix_data)
plt.show()


# From the chart, we can see that there are some fields which have NaN (Null Values). Of all the features, **director** column has the most number of NaN values.

# Let's count the total number of NaN for each column containing NaN values.

# In[ ]:


for column in netflix_data.columns:
    null_count = netflix_data[column].isna().sum()
    if null_count > 0:
        print(f"{column}'s null count: {null_count}")


# From the output, we can see that **director**, **date_added**, **cast** and **country** have significant number of null values.

# Now let's take a closer look at data by printing top 5 rows.

# In[ ]:


netflix_data.head()


# Now take a look at count, frequency and top result from data

# In[ ]:


netflix_data.describe(include='all').head(4)


# After exploring data this much, we will have visual presentation for mostly each of feature. Like what is word that comes frequently in **title**. Which Country has most number of Movies or TV shows. Which year Netlifx added most of movies. What is top genre and many more. Stay tuned !

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize=(14, 14), facecolor=None)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=1000, height=1000, max_words=150).generate(' '.join(netflix_data['title']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Most Popular words in Title", fontsize=25)
plt.show()


# Words like - Love, World, Man, Girl, Story, Life, Live have the most popularity in the Title.

# In[ ]:


plt.subplots(figsize=(9,7))
netflix_data['type'].value_counts().plot(kind='bar', fontsize=12,color='blue')
plt.xlabel('Type',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.title('Type Count',fontsize=12)
plt.ioff()


# There are almost 2 Times Movies than TV shows on Netflix.

# In[ ]:


netflix_data["date_added"] = pd.to_datetime(netflix_data['date_added'])
netflix_data['year_added'] = netflix_data['date_added'].dt.year
netflix_data[netflix_data['year_added']> 2009]['year_added']

plt.figure(figsize=(10,8))
sns.countplot(x='year_added', hue='type', data=netflix_data[netflix_data['year_added']>2000])
# Show the plot
plt.show()


# By this graph, we understand that Netflix is adding more number of movies/TV shows in recent years. Netflix added almost 3 times movies in year 2017 compared to year 2016.

# Let's look at the graph for countries having most movies/TV shows.

# In[ ]:


import squarify
from collections import Counter
country_data = netflix_data['country'].dropna()
country_count_dict = dict(Counter([country_name.strip() for country_name in (','.join(country_data).split(','))]))
country_data_count = pd.Series(country_count_dict).sort_values(ascending=False)
y = country_data_count[:25]
plt.rcParams['figure.figsize'] = (20, 16)
squarify.plot(sizes = y.values, label = y.index, color=sns.color_palette("RdGy", n_colors=25))
plt.title('Top 25 producing countries', fontsize = 25, fontweight="bold")
plt.axis('off')
plt.show()


# Clearly we can see that United States, India and United Kingdom are the top movies/TV shows producing countries.

# Now let's look down at the graph of famous countries and shows released in that country.

# In[ ]:


start_year = 2010
end_year = 2020
def content_over_years(country):
    movie_per_year=[]
    tv_shows_per_year=[]

    for i in range(start_year,end_year):
        h=netflix_data.loc[(netflix_data['type']=='Movie') & (netflix_data.year_added==i) & (netflix_data.country==str(country))] 
        g=netflix_data.loc[(netflix_data['type']=='TV Show') & (netflix_data.year_added==i) &(netflix_data.country==str(country))] 
        movie_per_year.append(len(h))
        tv_shows_per_year.append(len(g))



    trace1 = go.Scatter(x=[i for i in range(start_year, end_year)], y=movie_per_year, mode='lines+markers', name='Movies')

    trace2=go.Scatter(x=[i for i in range(start_year, end_year)], y=tv_shows_per_year, mode='lines+markers', name='TV Shows')

    data=[trace1, trace2]

    layout = go.Layout(title="Content added over the years in "+str(country), legend=dict(x=0.1, y=1.1, orientation="h"))

    fig = go.Figure(data, layout=layout)

    fig.show()

countries=['United States', 'India', "United Kingdom", "France", 'Australia','Turkey','Hong Kong','Thailand', 'Taiwan',"Egypt", 'Spain'
          ,'Mexico','Japan','South Korea','Canada']

for country in countries:
    content_over_years(str(country))


# Almost in every country, Netflix started its growth in 2014.

# In[ ]:


rating_df = netflix_data['rating'].dropna()
rating_df.value_counts()
fig, ax = plt.subplots(figsize=(10,7))
rating_df.value_counts().plot(kind='bar', fontsize=12,color='blue')
plt.xlabel('Rating Type',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.title('Rating Type Count',fontsize=12)
plt.ioff()


# There are highest number of 'TV-MA' and 'TV-14' ratings movies or TV shows on Netflix.

# STILL IN PROGRESS !!
