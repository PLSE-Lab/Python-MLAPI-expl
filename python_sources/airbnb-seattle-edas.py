#!/usr/bin/env python
# coding: utf-8

# **The goal of this kernel is to analyze the dataset AirBNB of Seattle. It contains information about the places which are able to rent and also about the hosts who are taking care of them.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium import plugins
from folium.plugins import HeatMap
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.offline as py
import plotly.graph_objs as go
pd.set_option('display.max_columns', 500)
sns.set_style('darkgrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Later on I will take care about only the **listings** and **calendar**. Review are for the further analysis.
# 
# File **listings** contains information about places able to rent. One observation is per every unique listing on AIRBNB. We've got 3818 unique listings.
# 
# On the other hand file **calendar** has 1 393 570 observations which means there is data for every listing for every day - 365*3818 = 1 393 570

# In[ ]:


df1 = pd.read_csv("../input/listings.csv")
df2 = pd.read_csv("../input/calendar.csv")
df3 = pd.read_csv("../input/reviews.csv")


# In[ ]:


df1.head(3)


# In[ ]:


drop = ['listing_url','scrape_id','last_scraped','name','summary', 'space', 'description',
        'experiences_offered','neighborhood_overview', 'notes', 'transit', 'thumbnail_url', 'medium_url',
        'picture_url', 'xl_picture_url', 'host_url', 'host_about', 'host_thumbnail_url',
        'host_picture_url', 'street', 'license', 'host_name', 'host_location',
        'host_neighbourhood', 'neighbourhood','neighbourhood_cleansed',
        'neighbourhood_group_cleansed', 'city', 'state', 'zipcode',
        'market', 'experiences_offered', 'smart_location', 'host_acceptance_rate', 'country',
        'country_code', 'has_availability', 'calendar_last_scraped', 'requires_license',
        'jurisdiction_names', 'square_feet', 'weekly_price', 'monthly_price', 'security_deposit',
        'cleaning_fee', 'host_listings_count']
df = df1.drop(columns=drop) 


# There are a lot of columns which I'm not going to use so I'm just gonna drop them. Most of them are text values, however a few of them are dropped beacuse of reasons posted below.

# 1. experiences_offered = 1 category
# 2. host_acceptance_rate = 2 categories
# 3. smart_location = 1 category
# 4. has_availability = only true category
# 5. calendar_last_scraped = only one date
# 6. requires_license = only false category
# 7. jurisdiction_names = only WASHINGTON
# 8. square_feet = too much NaN
# 9. weekly_price = too much NaN
# 10. montly_price = too much NaN
# 11. security_deposit = too much NaN
# 12. cleaning_fee = too much NaN
# 13. host_listings_count = correlation 1 with total listings

# In[ ]:


df.info()


# In[ ]:


df.head(2)


# In[ ]:


def correct_number(df_value):
        try:
            value = float(df_value[1:])
        
        except ValueError:
            value = np.NaN
        except TypeError:
            value = np.NaN
        return value
    
def correct_number1(df_value):
        try:
            value = float(df_value[:-1])
        
        except TypeError:
            value = np.NaN
        return value


# Before moving on there is a little bit of feature engineering. 

# In[ ]:


df['host_since'] = pd.to_datetime(df['host_since'], format='%Y-%m-%d')
df['first_review'] = pd.to_datetime(df['first_review'], format='%Y-%m-%d')
df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')

df['host_verifications_count'] = df['host_verifications'].apply(lambda x: x.count(' ') + 1)
df['amenities_count'] = df['amenities'].apply(lambda x: x.count(' ') + 1) 
df['property_type_new'] = df['property_type'] .replace(['Cabin', 'Camper/RV', 'Bungalow'], 'Category 1')
df['property_type_new'] = df['property_type_new'] .replace(['Condominium', 'Townhouse', 'Loft', 'Bed & Breakfast'], 'Category 2')
df['bed_type_new'] = df['bed_type'].replace(['Futon',' Pull-out Sofa', 'Airbed', 'Couch'], 'Other')

df['price_normal'] = df['price'].apply(correct_number)
df['extra_people_normal'] = df['extra_people'].apply(correct_number)
df['host_response_rate_normal'] = df['host_response_rate'].apply(correct_number1)

df = df.drop(columns=['host_verifications', 'amenities', 'property_type', 
                      'bed_type', 'extra_people', 'host_response_rate'])

df2['price_calendar'] = df2['price'].apply(correct_number)


# In[ ]:


object_c = df.select_dtypes(include='object')
numeric_c = df.select_dtypes(include='number')
for c in object_c.columns:
    l = len(object_c[c][object_c[c].notnull()].unique())
    print('Column {} has {} unique values'.format(c, l))
print('\n'*3,'NUMERIC')
for n in numeric_c.columns:
    print('Column', n)


# In[ ]:


print(df.isnull().sum().sort_values(ascending=False))
df.isnull().any(axis=1).value_counts()


# Here we've got a problem. There are missing values mostly in review scorings and host response time. At the beggining I thought that those lacks of data are correlated with each observation and corresponds to time of being host(new hosts have not reviews yet) but if I would like to drop every missing data there would be 984 observations to drop instead of 650~. That's 25% of the data so I'm not going to drop them but also not going to fill with any kind of method.

# **The very first look**

# In[ ]:


f, (ax1,ax2) = plt.subplots(1,2, figsize=(14,8))
g = df2['price_calendar'].groupby(df2['date']).mean()
h = df2['available'][df2['available'] == 't'].groupby(df2['date']).count()
ax1.plot(g)
ax1.set_xticks(g.index[::30])
ax1.tick_params('x', labelrotation=90)
ax2.plot(h)
ax2.set_xticks(h.index[::30])
ax2.tick_params('x', labelrotation=90)
f.suptitle('Price and availability over the year', fontsize=20)
ax1.set_title('Mean price')
ax2.set_title('Availability')
plt.show()


# In[ ]:


a = df['host_id'].value_counts().reset_index()
a.rename(columns={'index': 'host_id', 'host_id': 'num_of_listings'}, inplace=True)
b = df['review_scores_rating'].groupby(df['host_id']).mean().reset_index()
c = a.merge(b, how='left', on='host_id')
#c = c[['index', 'review_scores_rating']]
c.sort_values(by='num_of_listings', ascending=False).head(10)

plt.figure(figsize=(14,8))
sns.scatterplot(x=c['num_of_listings'],y=c['review_scores_rating'], alpha=0.5)
plt.ylabel('Review rating')
plt.xlabel('Number of listings per host')
plt.title('Host involvement', fontsize=20)


# * The problem with the 1st graph is that the mean price is only counted from the listings that are available on specific date. That means at the beginning of the year mean is from less than 1800 observations while at the very end of the year there about 2900 observations which are included in mean price.
# * Beside that there is seasonality in price series with peak in a summer time.
# * If we talk about availabiity during year there is strange situation in the winter time. In January we've got less than 1800 listings available, however in December the same year there is almost 3000 listings which are able to be rent. Was January some kind of special time that year?
# * Most of the hosts have less than 5 places to take care off. However there are people who I'm pretty sure are taking it to real business with record of 46 listings. Even though they stay at the top of mean review ratings.

# **MAPS**

# Maps below are interactive and represents prices and density in geographical locations over the city of Seattle. 

# In[ ]:


map_hooray = folium.Map(location=[47.60, -122.24], zoom_start = 11) 

for i in range(0, df.shape[0]):
    folium.Circle(
    location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],
    popup=df.iloc[i]['price'],
    radius=df.iloc[i]['price_normal']/3,
    color='red',
    fill=True,
    fill_color='black').add_to(map_hooray)


map_hooray


# In[ ]:



#points = sns.scatterplot('latitude', 'longitude', data=df1, hue='price_normal')
df_high = df[df['price_normal'] > 400]
plt.figure(figsize=(14,8))
points = plt.scatter(df['latitude'], df['longitude'], c=df["price_normal"], s=20, cmap="viridis") #set style options
#add a color bar
plt.colorbar(points)


# In[ ]:


map_hooray = folium.Map(location=[47.60, -122.24], zoom_start = 11)

heat_data = [[row['latitude'],row['longitude']] for index, row in
             df[['latitude', 'longitude']].iterrows()]

hh =  HeatMap(heat_data).add_to(map_hooray)

map_hooray


# Highest prices can be found in the centre of the city, aswell as density there of AirBNB houses is higher than in the coastline.

# **Hosts and listings**

# In[ ]:


def cross(c1,c2, xlabel, title):
    p = pd.crosstab(df[c1], df[c2])
    p.plot.bar(stacked=True, figsize=(14,8))
    plt.xlabel(xlabel)
    plt.suptitle(title, fontsize=20)
    plt.show()


# In[ ]:


cross('host_is_superhost', 'host_response_time', 'Superhost status', 'Status vs response time')


# In[ ]:


d = df['review_scores_rating'].groupby(df['host_response_time']).mean()
d = d.sort_values(ascending=False)
f, ax = plt.subplots(1,1, figsize=(14,8))
ax.plot(d, 'o-')
ax.set_xticklabels(d.index)
ax.tick_params('x', labelrotation=45)
plt.ylabel('Review rating')
plt.xlabel('Respone time')
plt.title('Mean rating of response time', fontsize=20)
plt.show()


# In[ ]:


plt.figure(figsize=(14,8))
sns.boxplot(x='host_is_superhost', y='review_scores_rating', data=df)
plt.ylabel('Review rating')
plt.xlabel('Superhost status')
plt.show()


# In[ ]:


cross('room_type', 'property_type_new', 'Room type', 'Types of rooms and properties')


# In[ ]:


cross('cancellation_policy', 'instant_bookable', 'Cancellation policy', 'Booking and cancellation')


# * There is less than twice hosts with status of *superhost*. They are generally evaluated with higher rating. Also looks like fast response is one of the key aspect to get high review rating
# * Most common type of room is the entire place followed by private room and shared room at the end. Almost all the time it's apartament or house then we've got category 2 which is something like hostel.
# * Only 15% of listings are instant bookable, I suppose that there is possibility to negotiate the price with the host.

# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(df.drop(columns=['id','host_id', 'latitude', 'longitude']).corr(),cmap='Blues', annot=True, fmt='.1f', linewidths=0.5)


# * The heatmap shows 4 significant correlated squares.
# The first one applies to equipment of the house. Next one is about availability of long term rent. And of course third one shows that different categories of rating are often correlated wich each other. Last one is most important for customers and shows that price is highly correlated with numbers of accomodates, bed, bathrooms and guests included

# In[ ]:


sns.pairplot(x_vars=['review_scores_rating', 'review_scores_accuracy',
                     'review_scores_checkin', 'review_scores_cleanliness',
                     'review_scores_communication', 'review_scores_location'],
             y_vars=['review_scores_rating', 'review_scores_accuracy',
                     'review_scores_checkin', 'review_scores_cleanliness',
                     'review_scores_communication', 'review_scores_location'],
            data=df, kind='reg', diag_kind='hist')


# In[ ]:


plt.figure(figsize=(14,8))
plt.hist(df['review_scores_rating'], bins=15, histtype='stepfilled', label='review_scores_rating', alpha=0.5)
for p in ['review_scores_accuracy',
          'review_scores_checkin', 'review_scores_cleanliness',
          'review_scores_communication', 'review_scores_location']:
    plt.hist(df[p]*10, bins=15, histtype='stepfilled', label=p, alpha=0.3)
    plt.legend(loc='upper left')
plt.suptitle('Histogram of ratings', fontsize=20)
plt.show()


# In[ ]:


x=df['beds']
y=df['bedrooms']
z=df['accommodates']


trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=df['price_normal'],               
        colorscale='Viridis',   
        opacity=0.8,
        colorbar=dict(
                title='Price'
            ),
        
    ),text=df['price']
)

data = [trace1]
layout = go.Layout(
    scene=dict(
    xaxis=dict(
        title='beds'),
    yaxis=dict(
        title='bedrooms'),
    zaxis=dict(
        title='accommodates')),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')


# Most of the ratings on AirBNB is pretty high which is quite normal situation. Different kind of ratings are correlated with each other but not always.
# Interesting fact which can't be seen on 3D graph is that the most expensive place on the data with the price of 999$ has only 1 bathroom, 1 bedroom and 1 bed.
