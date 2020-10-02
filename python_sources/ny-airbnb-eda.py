#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# Any results you write to the current directory are saved as output.


# In[ ]:


airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.head(10)


# In[ ]:


boroughs = airbnb.groupby('neighbourhood_group')['host_id'].agg('count').reset_index().sort_values(by = ['host_id'])
fig, ax1 = plt.subplots(1,1, figsize=(6,6)
                       )
sns.set_palette('Blues')
sns.barplot(x='neighbourhood_group', y='host_id', data=boroughs, ax=ax1)

ax1.set_title('Number of Listings by Borough', fontsize=15)
ax1.set_xlabel('Borough', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.tick_params(axis='both', labelsize=10)


# > As expected, Manhattan has the most number of listings, followed closely by Brooklyn.Staten Island and Bronx do not seem to be very popular on airbnb.Let's check to see if there are any missing values.

# In[ ]:


airbnb.isnull().sum()


# > 'name','host_name','last_review' and 'reviews_per_month' have null values. It makes sense to replace these with 0. Also, I dont think i need the column 'last_review'. So let's drop that.

# In[ ]:


airbnb['name'].fillna(value=0, inplace=True)
airbnb['reviews_per_month'].fillna(value=0, inplace=True)
airbnb['host_name'].fillna(value=0, inplace= True)
airbnb.drop('last_review', axis=1, inplace=True)
airbnb.isnull().sum()


# In[ ]:


room_cat = airbnb.groupby(['room_type','neighbourhood_group'])['id'].agg('count').reset_index().sort_values(by = ['id'] )
fig, ax2 = plt.subplots(1,1, figsize=(10,5))
                       
sns.set_palette('YlOrRd')
sns.barplot( x='room_type',y = 'id',hue='neighbourhood_group', data=room_cat, ax=ax2)

ax2.set_title('Number of Listings by Room Type and Borough', fontsize=15)
ax2.set_xlabel('Borough', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.tick_params(axis='both', labelsize=10)


# > The most popular type of listing is 'Entire home/apt'. Let's see which neighbourhood has the most listings in Manhattan and in Brooklyn.Interestingly, Brooklyn has more private rooms listed than Entire apartments.

# In[ ]:


NY_neigh = airbnb[airbnb['neighbourhood_group'] == 'Manhattan'].groupby('neighbourhood')['id'].agg('count').reset_index().sort_values(by = ['id'],ascending = False).head(10)
fig, ax3 = plt.subplots(1, 1, figsize=(16,6))
sns.barplot(x='neighbourhood', y='id',data=NY_neigh,ax= ax3,palette='RdYlBu')
ax3.set_title('Neighbourhoods with the most listings in Manhattan')
ax3.set_xlabel('neighbourhood',fontsize = 12)
ax3.set_ylabel('Count',fontsize = 12)
ax3.tick_params(axis = 'both',labelsize = 10)


# In[ ]:


Brook_neigh = airbnb[airbnb['neighbourhood_group'] == 'Brooklyn'].groupby('neighbourhood')['id'].agg('count').reset_index().sort_values(by = ['id'],ascending = False).head(10)
fig, ax3 = plt.subplots(1, 1, figsize=(16,6))
sns.barplot(x='neighbourhood', y='id',data=Brook_neigh,ax= ax3,palette='RdYlBu')
ax3.set_title('Neighbourhoods with the most listings in Brooklyn')
ax3.set_xlabel('neighbourhood',fontsize = 12)
ax3.set_ylabel('Count',fontsize = 12)
ax3.tick_params(axis = 'both',labelsize = 10)


# > Brooklyn has the most listings in Williamsburg. Bedford-Stuyvesant is not far behind. Manhattan has the most listings in Harlem. It appears that most of Brooklyn's listings are concentrated in Williamsburg and Bedford-Stuyvesant. Whereas Manhattan's listings seem to be distributed quite evenly amongst the top 5 neighbourhoods.Next, Let's look at a distribution of the price by borough.

# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(data=airbnb, x='neighbourhood_group', y='price', palette='GnBu_d')
plt.title('Density and distribution of prices for each neighbourhood group', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Price")


# > There seem to be outliers in the data.Let's only look at listings where price < 700.

# In[ ]:


plt.figure(figsize=(15,6))
sns.violinplot(data=airbnb[airbnb.price <700], x='neighbourhood_group', y='price', palette='Greens')
plt.title('Density and distribution of prices for each neighbourhood group', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Price")


# > Much better. As expected, Manhattan has the highest range of prices. Queens,Staten Island and the Bronx follow a similar distribution. Most listings in these boroughs are priced between 40 and 70. Manhattan's median price appears to be around 150 and most of the listings are priced atleast around 100 per night. I'm guessing that a majority of these listings are in Harlem.Let's identify Manhattan's and Brooklyn's priciest neighbourhoods.Keep in mind that price varies by the size of the apartment , number of rooms ,amenities etc. So, the average price would not be an exact representation of the landscape. Since outliers can heavily skew the distribution , I've only considered prices below 700 and greater than 0. 

# In[ ]:


Man_price = airbnb[(airbnb['neighbourhood_group'] == 'Manhattan') & (airbnb['room_type'] == 'Entire home/apt') & (airbnb.price < 700) & (airbnb.price > 0)].groupby('neighbourhood')['price'].agg('mean').reset_index().sort_values(by = ['price'],ascending = False).head(20)
plt.figure(figsize=(15,6))
sns.barplot(data=Man_price, y='neighbourhood', x='price', palette='RdGy')
plt.title('Average Price of Entire apartments by Neighbourhood', fontsize=15)
plt.ylabel('Neighbourhoods')
plt.xlabel("Price")


# In[ ]:


Brook_price = airbnb[(airbnb['neighbourhood_group'] == 'Brooklyn') & (airbnb['room_type'] == 'Entire home/apt') & (airbnb.price < 700) & (airbnb.price > 0)].groupby('neighbourhood')['price'].agg('mean').reset_index().sort_values(by = ['price'],ascending = False).head(20)
plt.figure(figsize=(15,6))
sns.barplot(data=Brook_price, y='neighbourhood', x='price', palette='RdGy')
plt.title('Average Price of Entire apartments by Neighbourhood', fontsize=15)
plt.ylabel('Neighbourhoods')
plt.xlabel("Price")


# > The most expensive neighbourhoods in Manhattan and Brooklyn are Tribeca and Vinegar Hill. Vinegar Hill and DUMBO have similar prices. This makes sense as these neighbourhoods are right next to each other.Let's look at the hosts with the most number of listings.

# In[ ]:


top_host = airbnb.host_id.value_counts().head(10)
sns.set(rc={'figure.figsize':(10,8)})
viz_1=top_host.plot(kind='bar')
viz_1.set_title('Hosts with the most listings in NYC')
viz_1.set_ylabel('Listings')
viz_1.set_xlabel('Host IDs')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)


# > The top host has more than 300 listings to his name. Let's try to investigate the reason for his popularity. Let's see if we can extract some insights from the descriptions of the airbnbs. Do they impact revenue ? First, let's standardize all the text by converting it to lower case letters.

# In[ ]:


airbnb['new_name'] = airbnb['name'].str.lower()
text = airbnb.new_name.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# > Quiet ,clean ,home and apt pop out of this word cloud. Let's see what pops out if we only include the pricey airbnbs. First let's create a new column in the dataframe called price_group.How do we determine if a price is a high or low? Let's check the percentile distributions.

# In[ ]:


airbnb[(airbnb.price < 700) & (airbnb.price > 0)].price.describe()


# In[ ]:


def price_group(row):
    if row['price'] < 61:
        val = 'Very Low'
    elif row['price'] < 91:
        val = 'Low'
    elif row['price'] < 131:
        val = 'Moderate'
    elif row['price'] < 201:
        val = 'High'
    else:
        val = 'Very High'
    return val

airbnb['price_group'] = airbnb.apply(price_group, axis=1)


# Let's create a similar grouping for the number_of_reviews variable.

# In[ ]:


airbnb.number_of_reviews.describe()


# In[ ]:


def reviews_group(row):
    if row['number_of_reviews'] < 1:
        val = 'None'
    elif row['number_of_reviews'] < 5:
        val = 'Low'
    elif row['number_of_reviews'] < 24:
        val = 'High'
    else:
        val = 'Very High'
    return val

airbnb['review_group'] = airbnb.apply(reviews_group, axis=1)


# Let's also create a total revenue variable. Total revenue = price X minimum no of nights X number of reviews

# In[ ]:


airbnb['total_revenue'] = airbnb.price * airbnb.number_of_reviews *airbnb.minimum_nights


# Let's look at the word cloud for pricey apartments and for the cheapest apartments. But before we do, let's clean up the data a little bit.

# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update('NEW','YORK','York','room','Bedroom')
text = airbnb[(airbnb.price_group == 'Very High')| (airbnb.price_group =='High')].new_name.values
wordcloud = WordCloud(stopwords = stopwords,max_words = 100,max_font_size = 100,min_font_size = 10,height = 600, width = 1000 ).generate(str(text))
fig = plt.figure()
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# > Castle,Skylit , Midtown,heart,Clean and Village pop out.

# In[ ]:


stopwords = set(STOPWORDS)
stopwords.update('NEW','YORK','York','room','Bedroom')
text = airbnb[(airbnb.price_group == 'Very Low')].new_name.values
wordcloud = WordCloud(stopwords = stopwords,max_words = 100,max_font_size = 100,min_font_size = 10,height = 600, width = 1000 ).generate(str(text))
fig = plt.figure()
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# Lovely ,spacious,furnished and Sunny stand out.Now that we've found the most common unigrams. Let's look at a distribution of the number of characters in the description.

# In[ ]:



fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=airbnb[airbnb['price_group'] == 'Very High']['name'].str.len()
ax1.hist(tweet_len,color='blue')
ax1.set_title('Pricey Listings')
tweet_len=airbnb[airbnb['price_group'] == 'Very Low']['name'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Cheap listings')
fig.suptitle('Characters in the description')
plt.show()


# 30 - 50 characters seems to be the most common in both categories. 
