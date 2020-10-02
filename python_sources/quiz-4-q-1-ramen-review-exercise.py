#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 1a. Import file of ramen ratings and show what the dataset looks like using .head() and reset index if necessary
# 
# *Hint: the index is most likely assigned numbers to each row, reset the index to the review # to make it more useful for data extraction

# In[ ]:


ramen = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
ramen = ramen.set_index('Review #')
ramen.head()


# *The Stars column, which contains the information of what each user rated the ramen in their review based on a 1 to 5 scale (5 being the best, 1 being the worst), was stored as a string. To properly use and extract information from the dataset we need this information in a numeric form, here it will need to be float, so below we will convert the entire column from string to float.*

# In[ ]:


ramen['Stars']=pd.to_numeric(ramen['Stars'],errors='coerce')


# 1b. Show the number of entries in the table using .shape

# In[ ]:


ramen.shape #shows how many rows and how many columns in the data set


# *The information above shows there are 2580 rows meaning there are 2580 reviews of ramen in this dataset. There are 6 columns, in addition to the index review #, which are the attributes each ramen review has: brand, variety, style, country, stars, and Top Ten (if the ramen is considered one of the Top Ten all time ramens on the review website).*

# 2a. Create a New Data Frame called 'Reviews' that has the average rating of each brand and the number of reviews for each brand

# In[ ]:


brands = ramen.groupby('Brand')
reviews = brands.mean().rename(columns = {'Stars': 'Average_Stars'})
reviews['review_count'] = brands.size()
reviews.head()


# 2b. Select brands with the 15 most reviews

# In[ ]:


top_reviews = reviews.review_count.sort_values(ascending = False).iloc[0:16]
top_reviews.head(15)


# *This information tells us which brands are the most popularly known, as they have the most reviews and are therefore consumed by the most people.*

# 2c. Show brands where reviews average to an Average Stars score of 4.5 or higher and they recieved 2 or more reviews (to increase credibility of review, and minimize possibility of one poor experience).

# In[ ]:


high_reviews = reviews[reviews.Average_Stars >= 4.5]
high_reviews = high_reviews[high_reviews.review_count >= 2]
high_reviews.head(10)


# 2d. How many brands have high ratings?

# In[ ]:


high_reviews.shape


# *45 brands have high reviews, as indicated by the 45 rows in the dataframe high reviews that only includes brands with reviews of 4.5 or higher*

# 2e. Show the 10 brands with the lowest average rating.

# In[ ]:


reviews.Average_Stars.sort_values(ascending = False).tail(10)


# 3a. Ramen (regardless of type or style) from which country has the highest average review?

# In[ ]:


country = ramen.groupby('Country')
country_ratings = country.mean().rename(columns = {'Stars': 'Average_Country_Stars'})
max_rating = country_ratings.Average_Country_Stars.max()
max_country = country_ratings.Average_Country_Stars.idxmax()
print('The country with the most reviews is ' + max_country + ' with the average star rating of ' + str(max_rating) + '.')


# *Code: The code above first groups the ramen reviews by Country, then calculates the average (mean) for any numerical information stored. In this case, the only non-index column with a numerical value is the Stars column so the mean for stars is calculated. Next, the maximun star rating for the countries is found and stored as max_rating. This only returns the number, without information of what country the maximun average rating is connected to, so the index (which is now the country name due to the newly created series that is based on group by country) that is connected to the maximum rating is also found and stored as max_country.*
# 
# 
# *This information is useful because it shows which country has the highest rated, and therefore best ramen, according to this dataset.*

# 3b. Which country has the most amount of reviews?

# In[ ]:


country_ratings['total_reviews'] = country.size()
most_reviews = country_ratings.total_reviews.sort_values(ascending = False).reset_index().iloc[0]
print(most_reviews)
country_ratings.loc[most_reviews.Country] 


# *The second line tells us that Japan has the most amount of reviews for Ramen out of all countries, with 352 total reviews in the dataset. This information tells us that the most reviews for Ramen from any country are from Japan, indicating that Ramen is popular in Japan and many people review their Ramen. The third line seeks out the information for Japan, showing both the total reviews and the average star score, which is 3.98.*

# 4. Plot a pie chart of Ramen Styles

# In[ ]:


style = ramen.groupby('Style').size()
style.plot.pie()


# *This pie chart shows the different styles Ramen is served in, and it shows that Ramen that comes in Packs are reviewed a majority of the times.*
