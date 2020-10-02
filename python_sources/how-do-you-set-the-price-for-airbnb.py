#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


# In[ ]:


cal = pd.read_csv('/kaggle/input/boston/calendar.csv')
lis = pd.read_csv('/kaggle/input/boston/listings.csv')
rev = pd.read_csv('/kaggle/input/boston/reviews.csv')


# In[ ]:


len(cal)


# In[ ]:


len(lis)


# In[ ]:


len(rev)


# In[ ]:


rev.isnull().sum()


# In[ ]:


cal.head()


# In[ ]:


lis.head()


# In[ ]:


lis.columns


# In[ ]:


# extract columns I expected to relate with prices
lis[['room_type', 'bathrooms', 'bedrooms', 'beds','square_feet', 'price', 'weekly_price', 'monthly_price']].head(10)


# In[ ]:


# the number of None data
lis[['accommodates', 'room_type', 'bathrooms', 'bedrooms', 'beds', 'square_feet', 'price', 'weekly_price', 'monthly_price']].isnull().sum()


# In[ ]:


max(lis.beds)


# In[ ]:


rev.head()


# # Questions
# ### 1. What changes the price?
# **We need to check correlation**
# - largeness?
# - location?
# 
# ### 2. How can we get more money from AirBnB business?
#  - distributions of properties price
#  - 1) What makes more booked listings? or less booked listings?
#      - cheap? good environment? good reviews?
#  - 2) What makes good review scores by customers?
#  - 3) Owner's attributes
#      - 1, Do owners who have more propaties earn more money than owners who have less propaties? or Do owners with experiences for longer time earn more money than 
#       - (This is based on the idea that more experiences of sharing rooms/houses increases profits.)
#      - 2, Owner's reliability/hospitality

# In[ ]:


# check the type
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # show all contents in a seriese
    print(lis.dtypes)


# **BE CAREFUL**
# 
# **Price is object so it needs to be int or float.**

# In[ ]:


# change the type for price
lis.price = lis.price.str.replace('$', '').str.replace(',', '').astype(float)


# In[ ]:


# check correlation between numerical variables
lis.describe()


# In[ ]:


# check  the distribution for numerical variables
lis.hist(bins=20, xlabelsize=20, ylabelsize=20, figsize=(40,40));


# In[ ]:


# select columns for conditions of price
selected_lis = lis[['accommodates', 'host_listings_count', 'host_total_listings_count', 'bathrooms', 'bedrooms', 
                   'beds', 'number_of_reviews', 'review_scores_accuracy', 'price', 'square_feet']]


# In[ ]:


# check correlation between extracted variables
plt.figure(figsize=(10,10))
sns.heatmap(selected_lis.corr(), annot=True, fmt='.2f');


# ## About correlation between numerical variables
# - `price` has higher coefficient of correlation with `accommodates`, `bedrooms` and `square_feet`.
# - However, `square_feet` has higher coefficient of correlation with `bedrooms`, and `accommodates`. `square_feet` has more `Nan` than `bedrooms` and `accommodates`.
# - `accommodates` and `bedrooms` are highly related so we need to choose one of them. `accommodates` doesn't have `Nan` so we will chose `accommodates`.

# In[ ]:


# plot the price with scatter
# all properties
plt.figure(figsize=(15, 10));
sns.scatterplot(x='price', y='accommodates', hue='room_type', data=lis);
plt.xticks(rotation='vertical');
plt.title('Distribution of price')
plt.savefig('dist_price.jpg')


# ## About the distribution of price for each accomodates and room type
# - `price` varies even if conditions about the room type and accommodates are the same.
# 
# - Therefore, **`accommodates` and `room_type` are not enough to decide the price.**

# # What is the difference instead of `accommodates` and `room_type` ? 
# - how about location of the property?

# In[ ]:


# take a look at when accommodates == 2 and room_type == 'Entire home/apt'
tmp_df = lis[(lis.accommodates == 2) & (lis.room_type == 'Entire home/apt')]


# In[ ]:


# descriptive statistics for price
tmp_df.price.describe()


# In[ ]:


# check the counts for each city
tmp_df.city.value_counts()


# In[ ]:


# show average and sd as bar
plt.figure(figsize=(6,4));
sns.barplot(x='city', y='price', data=tmp_df[['city', 'price']].sort_values(by='price', ascending=False), 
            ci="sd", color='orange');
plt.xticks(rotation='vertical');
plt.title('Price variation in cities');
plt.savefig('price_var_cities.jpg')


# ## About city and price relationships
# - standard deviation is different in each city and `Boston` has big standard deviation which means its price varies in wide range.
# - `Cambridge` has the highest average price.
# 
# ## So city can differ price! Check the effect by difference of city for price

# In[ ]:


for i in lis.accommodates.unique():
    print('accommodates = ' + str(i), '\n')
    if len(lis[lis.accommodates == i]) >= 4:
        figsize = (30, 20)
    else:
        figsize = (15, 15)
    lis[lis.accommodates == i].hist(column='price', by=['city', 'room_type'], grid=True, figsize=figsize, bins=20,
                                   xlabelsize=10, ylabelsize=10);
    plt.show();
    plt.close();


# ## Price varies in cities under the condition that accommodates are the same!
# - **The combination of cities and room type changes the price.**

# # Summary
# - Question 1 ) `city`, `room_type` and `accommodates` effect the change of prices.

# ## * cal.price is the price when the properties are not booked.

# # Question 2. How can we get more money from AirBnB business?
# ## distributions of properties price
#  - 2-1) What makes more booked listings? or less booked listings?
#      - cheap?  good reviews?
#  - 2-2) What makes good review scores by customers?
#  - 2-3) Owner's attributes
#      - 1, Do owners who have more propaties earn more money than owners who have less propaties? or Do owners with experiences for longer time earn more money than 
#       - (This is based on the idea that more experiences of sharing rooms/houses increases profits.)
#      - 2, Owner's reliability/hospitality

# In[ ]:


# check the number of `available` for each listing
booked = cal[cal.available == 'f'].groupby('listing_id').available.count()
non_booked = cal[cal.available == 't'].groupby('listing_id').available.count()
booking_ratio = booked / (booked + non_booked)


# In[ ]:


more_booked = booking_ratio[booking_ratio >= 0.7].sort_values(ascending=False)
more_booked.head()


# In[ ]:


more_booked.describe()


# In[ ]:


less_booked = booking_ratio[booking_ratio <= 0.3].sort_values(ascending=False)
less_booked.head()


# In[ ]:


less_booked.describe()


# In[ ]:


more_booked_id = more_booked.index.to_list()


# In[ ]:


less_booked_id = less_booked.index.to_list()


# ##  2-1) What makes more booked listings? or less booked listings? 
# - check the price for each
# - check how positive/negative in reviews

# In[ ]:


def make_series_df(ids_list, target_column, df_return=True):
    '''
    by listing data, make dataframe for price with selected ids
    
    ids_list : list :  list for ids
    target_column : strings : the name of column
    df : bool : True->make and return a dataframe
                False-> make and return a series
    '''
    target_column_list = []
    df = pd.DataFrame()
    for sel_id in ids_list:
        if len(lis[lis.id == sel_id][target_column]) >= 1:
            target_column_list.append(lis[lis.id == sel_id][target_column].values[0])
        
    # when df_return = True    
    if df_return > 0:
        df[target_column] = target_column_list
        return df

    else:
        return pd.Series(target_column_list)


# In[ ]:


# make a df with more booked id
more_df = make_series_df(more_booked_id,'price', df_return=True)
more_df.head()


# In[ ]:


# make a df with less booked id
less_df = make_series_df(less_booked_id, 'price', df_return=True)
less_df.head()


# In[ ]:


more_df.describe()


# In[ ]:


less_df.describe()


# In[ ]:


sns.distplot(a=more_df['price'], bins=40, hist=False, color='orange', label='more 70 %');
sns.distplot(a=less_df['price'], bins=40, hist=False, color='gray', label='less 30 %');
plt.title('Price distribution')
plt.savefig('price_dist.jpg')


# In[ ]:


sns.distplot(a=more_df['price'], bins=40, hist=False, color='orange', label='more 70 %');
sns.distplot(a=less_df['price'], bins=40, hist=False, color='gray', label='less 30 %');
plt.xlim(200, 800);
plt.axvline(x=305, color='green', linestyle='--');
plt.title('Zoom up price distribution')
plt.savefig('intersection_price.jpg')


# ## Summary for price distribution
# - `less 30 %` properties tend to price higher. The intersection of 2 KDE is around 300 dollars which means `more 70 %` has more properties under around 300 dollars and `less 30 %` has more properties above 300 dolalrs.
# 
# - Therefore, it can be said properties with under 300 dollars are easier booked. 

# ## 2-2) What makes good review scores by customers?
# - check the review scores

# In[ ]:


review_scores = lis[['id', 'review_scores_accuracy', 'review_scores_rating',
    'review_scores_checkin', 'review_scores_cleanliness',
    'review_scores_communication', 'review_scores_location',
    'review_scores_value']].dropna()


# In[ ]:


review_scores.head()


# In[ ]:


review_scores.describe()


# **Weird point**
# - Every score's minmum value is 20 % of max. Is this kind of spam? No one didn't rate as 1 or 0 ?

# In[ ]:


# check the rating as sum
sns.distplot(a = review_scores.review_scores_rating.dropna(),kde=False, bins = 50);
plt.title('review scores for all data')
plt.savefig('all_scores.jpg')


# **About distribution of review_scores_rating**
# - It looks like there are 5 groups in all. So we need to try separate for each group.

# ## This is the time to separately analyze data into well booked and opposite

# In[ ]:


# check well booked (more 70 % booked)
more_review_scores = make_series_df(more_booked_id, 'review_scores_rating', df_return=False)
more_review_scores.head()


# In[ ]:


more_review_scores.describe()


# In[ ]:


sns.distplot(a = more_review_scores, kde = False, bins = 50);
plt.title('review score for well booked');
plt.savefig('well_booked_scores.jpg');


# In[ ]:


# check well booked (less 30 % booked)
less_review_scores = make_series_df(less_booked_id, 'review_scores_rating', df_return=False)
less_review_scores.head()


# In[ ]:


less_review_scores.describe()


# In[ ]:


sns.distplot(a = less_review_scores, kde = False, bins = 50);
plt.title('review score for less booked');
plt.savefig('less_booked_scores.jpg');


# In[ ]:


def ratio_rev_scores(review_scores, thresh=90):
    '''
    return the ratio of scores with some threshold (numeral)
    
    review_scores : Series : Series of review scores
    thresh : numeral : default is 90
    
    '''
    return (review_scores >= thresh).sum() / len(review_scores)


# In[ ]:


# the ratio of over 90 scores in reviews for less booked 
round(ratio_rev_scores(less_review_scores) * 100)


# In[ ]:


# the ratio of over 90 scores in reviews for well booked 
round(ratio_rev_scores(more_review_scores) * 100)


# ## Result for review comparing
# - Well booked and not well booked properties seems to have similar reviews....
#  - A difference is about 2 points in mean
#  - Minimum scores are the same
#  - Standard deviation's difference is about 0.1 points
# 
# 
# - The ratio of scores over 90 is different from well booked and less booked.
#  - about 67% of reviews for well booked is over 90
#  - about 59% of reviews for less booked is over 90 
# 
# 
# - It might suggest that you can't judge how well the properties can be booked by just reviews.

# # Summary for all questions
# ## Question 1 : What changes the price?
# 
# - location(city), room_type and accommodates effect the change of prices.
# 
# 
# ## Question 2 : How can we get more money from AirBnB business?
#  - distributions of properties price
#  - **2-1) What makes more booked listings? or less booked listings?**
#      - Can be said properties with under 300 dollars are easier booked
#  - **2-2) What makes good review scores by customers?**
#   - Even not well booked had as good scores as well booked. It seems like no link between how much booked and how good the review is. If you believe the review scores, you can say how much booked doesn't effect on the review scores.

# In[ ]:


# rev


# In[ ]:


# rev.dropna(subset=['comments'], how='any', inplace=True)


# In[ ]:


# rev.isnull().sum()


# In[ ]:


# make words lower
# rev['low_comments'] = rev['comments'].str.lower()
# rev['low_comments'].head()


# In[ ]:


# split each comments with space
# rev['low_nospace_com'] = rev['low_comments'].str.split(' ')
# rev['low_nospace_com'] .head()


# In[ ]:


# comments from reviews of more booked properties
# more_df_rev = []
# for more_id in more_booked_id:
# #     print(more_id)
#     more_df_rev.append(rev[rev.listing_id == more_id].low_nospace_com.values)


# In[ ]:


# more_df_rev[0][0]


# In[ ]:


# import re
# text_lis = []
# for one_id in more_df_rev:
#     for comments in one_id:
# #         print(comments)
#         text = [re.sub(r'[0-9]*', '', word) for word in comments if str(word) != 'nan']
#         text_lis.append(text)
# #         words = [x for x in comments if x]
# #             print(text)


# In[ ]:


# text_lis[0]


# In[ ]:


# remove stop words
# from nltk.corpus import stopwords
# stopwords_en = stopwords.words('english')


# In[ ]:


# nonstop_text = [x for one_com in text_lis for x in one_com if x not in stopwords_en]


# In[ ]:


# nonstop_text

