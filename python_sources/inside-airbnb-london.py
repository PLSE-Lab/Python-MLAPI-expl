#!/usr/bin/env python
# coding: utf-8

# # Exploring guest review scores and their relationships with properties of Airbnb hosts/listings/neighbourhoods in London

# This analysis has also corresponding Medium article, where I summarized my general findings about guest reviews:  
# [Exploring Airbnb Guest Reviews in London](https://medium.com/@labdmitriy/exploring-airbnb-guest-reviews-in-london-682b45aba34e)

# ## Table of Contents
# [Business Understanding](#business-understanding)  
# * [Determine business objectives](#business-objectives)
# * [Assess situation](#assess-situation)
# * [Determine data mining goals](#data-mining-goals)
# * [Produce project plan](#project-plan)
# 
# [Data Understanding](#data-understanding)
# * [Collect initial data](#collect-data)
# * [Describe data](#describe-data)
# * [Explore data](#explore-data)
# 
# [Data Preparation](#data-preparation)
# * [Select data](#select-data)
# * [Clean data](#clean-data)
# * [Construct data](#construct-data)
# * [Integrate data](#integrate-data)
# 
# [Data Understanding](#data-understanding-2)
# * [Which neighbourhood is preferable for renting based on guest review scores?](#question-1)
# * [Are there any relationships between different types of guest review scores?](#question-2)
# * [How host/listings properties are related with guest review scores?](#question-3)
# 
# [Evaluation](#evaluation)
# * [Evaluate results](#evaluate-results)
# * [Review process](#review-process)
# * [Determine next steps](#next-steps)

# In[ ]:


from pathlib import Path

import requests
from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import altair as alt


# In[ ]:


sns.set_style('white')

alt.themes.enable('default')
alt.renderers.enable('kaggle')
alt.renderers.set_embed_options(actions=False)
alt.data_transformers.enable('json')


# ## Business Understanding <a class="anchor" id="business-understanding"></a>

# ### Determine business objectives <a class="anchor" id="business-objectives"></a>
# **Background**  
# Airbnb was founded in 2008 and has already become a very popular service for travellers around the world.  
# 
# The number of Airbnb listings in London has grown significantly in the past few years, especially after it was legalised in 2015  (https://www.travelgumbo.com/blog/london-legalizes-airbnb-similar-rentals).  
# 
# The reputation of the service can be disrupted by fraud (https://www.vice.com/en_us/article/43k7z3/nationwide-fake-host-scam-on-airbnb) or can be increased by different improvements.
# 
# One hypothesis that we have is that guest review scores can be reliable factor to evaluate current status of attractiveness of the service in general, and increasing hosts reputation can be the reason for service reputation improvement.
# 
# **Business objectives**
# - How can Airbnb can improve its reputation for prospective users in London?
# 
# **Business success criteria**
# - Give useful insights what the overall impression of the Airbnb users depends on and probably how it can be improved

# ### Assess situation <a class="anchor" id="assess-situation"></a>
# **Inventory of resources**
# - Data
#     - Inside Airbnb project (http://insideairbnb.com/)
# - Hardware
#     - Google Cloud Platform
# - Software
#     - Python data science platform (Anaconda)
#     
# **Requirements, assumptions and constraints**
# - Guest review scores can be reliable factors for assessing "attractiveness" of hosts/listings/neighbourhoods
# - Better guest reviews will entail service reputation increasing and additional bookings and users

# ### Determine data mining goals <a class="anchor" id="data-mining-goals"></a>
# **Data mining goals**
# - Which neighbourhood is preferable for renting based on guest review scores? 
# - Are there any relationships between different types of guest review scores?
# - How host/listings properties are related with guest review scores?
# 
# **Data mining success criteria**
# - Define features of hosts/listings/neighbourhoods that have strong relationships with guest review score types

# ### Produce project plan <a class="anchor" id="project-plan"></a>
# **Project plan**
# - Gather
#     - Data source: Airbnb data for London from Inside Airbnb project
# - Assess
#     - Describe data
#     - Explore data for useful features
# - Clean
#     - Select required subset of data
#     - Preprocess categorical data
#     - Preprocess missing data
#     - Generate additional features based on selected ones
# - Analyze
#     - Analyze data for answering relevant questions
# - Visualize
#     - Visualize findings
# - Results
#     - Formulate the answers for the business/data mining questions
# 
# **Initial assessment of tools and techniques**
# - Programming language  
#   - Python
# - Packages
#   - Data loading
#       - requests, pandas, geopandas
#   - Data analysis
#       - numpy, pandas, geopandas
#   - Visualization
#       - matplotlib, seaborn, altair

# ## Data Understanding <a class="anchor" id="data-understanding"></a>

# ### Collect initial data <a class="anchor" id="collect-data"></a>

# In[ ]:


DATA_PATH = Path('../input/airbnb/')
listings_df = pd.read_csv(DATA_PATH/'listings_summary.csv',
                          parse_dates=['last_review'])
listings_detail_df = pd.read_csv(DATA_PATH/'listings.csv', low_memory=False,
                                 parse_dates=['host_since', 
                                              'last_scraped', 'calendar_last_scraped',
                                              'first_review', 'last_review'])

reviews_df = pd.read_csv(DATA_PATH/'reviews_summary.csv', parse_dates=['date'])
reviews_detail_df = pd.read_csv(DATA_PATH/'reviews.csv', parse_dates=['date'])

calendar_df = pd.read_csv(DATA_PATH/'calendar.csv', parse_dates=['date'])

neighbourhoods_df = pd.read_csv(DATA_PATH/'neighbourhoods.csv')
gdf = gpd.read_file(DATA_PATH/'neighbourhoods.geojson')


# **Initial data collection report**
# - Data source
#     - http://insideairbnb.com/get-the-data.html
#     - Section "London, England, United Kingdom"  
# 
# 
# - Data files
#     - Detailed Listings data for London  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/data/listings.csv.gz
#     
#     - Detailed Calendar Data for listings in London  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/data/calendar.csv.gz
# 
#     - Detailed Review Data for listings in London  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/data/reviews.csv.gz
# 
#     - Summary information and metrics for listings in London (good for visualisations)  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/visualisations/listings.csv
# 
#     - Summary Review data and Listing ID (to facilitate time based analytics and visualisations linked to a listing)  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/visualisations/reviews.csv
# 
#     - Neighbourhood list for geo filter. Sourced from city or open source GIS files  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/visualisations/neighbourhoods.csv
# 
#     - GeoJSON file of neighbourhoods of the city  
#     http://data.insideairbnb.com/united-kingdom/england/london/2019-11-05/visualisations/neighbourhoods.geojson
#     
#     
# - Data access 
#     - Data is refreshed periodically on the site, but the links for previous versions of data are still available

# ### Describe data <a class="anchor" id="describe-data"></a>

# **Summary information and metrics for listings in London**

# In[ ]:


listings_df.info()


# **Detailed Listings data for London**

# In[ ]:


listings_detail_df.info()


# In[ ]:


print(listings_detail_df.columns.tolist())


# **Detailed Calendar Data for listings in London**

# In[ ]:


calendar_df.info(null_counts=True)


# **Summary Review data and Listing ID**

# In[ ]:


reviews_df.info()


# **Detailed Review Data for listings in London**

# In[ ]:


reviews_detail_df.info()


# **Neighbourhood list for geo filter**

# In[ ]:


neighbourhoods_df.info()


# **GeoJSON file of neighbourhoods of the city**

# In[ ]:


gdf.plot();


# **Data description report**  
# - Data summary statistics described above are self-explanatory
# - Additional data considerations are available on the Inside Airbnb project page:  
# http://insideairbnb.com/about.html

# ### Explore data <a class="anchor" id="explore-data"></a>

# **Summary information and metrics for listings in London**

# Sample of data

# In[ ]:


listings_df.head(1)


# Features with non-zero number of missing values

# In[ ]:


print(listings_df.shape)
listings_df.loc[:, listings_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Features with zero variance

# In[ ]:


listings_df.loc[:, listings_df.nunique() <= 1].nunique().sort_values()


# Summary information for datetime features

# In[ ]:


listings_df.describe(include='datetime')


# Summary information for string features

# In[ ]:


listings_df.describe(include=['object'])


# Number of listings by neighbourhood

# In[ ]:


listings_df['neighbourhood'].value_counts().sort_values().plot.barh(figsize=(10, 10));
sns.despine()
plt.title('Number of listings by neighbourhood', fontsize=14);


# Number of listings by room type

# In[ ]:


listings_df['room_type'].value_counts(dropna=False).sort_values().plot.barh()
sns.despine()
plt.title('Number of listings by room type', fontsize=14);


# Numeric features distribution

# In[ ]:


listings_df.hist(figsize=(12, 10), bins=20, grid=False)
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


# **Detailed Listings data for London**

# Sample of data

# In[ ]:


listings_detail_df.head(1)


# Features with non-zero number of missing values

# In[ ]:


print(listings_detail_df.shape)
listings_detail_df.loc[:, listings_detail_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Features with zero variance

# In[ ]:


listings_detail_df.loc[:, listings_detail_df.nunique() <= 1].nunique().sort_values()


# Missing values count distribution for review scores by row

# In[ ]:


listings_detail_df.filter(regex='review_scores').notnull().sum(axis=1).value_counts(normalize=True)


# Summary information for datetime features

# In[ ]:


listings_detail_df.describe(include='datetime')


# Summary information for string features

# In[ ]:


listings_detail_df.describe(include='object').T


# Country codes distribution

# In[ ]:


print(listings_detail_df['country_code'].value_counts())
listings_detail_df.query('country_code != "GB"')


# Numeric features distribution

# In[ ]:


listings_detail_df.hist(figsize=(12, 30), bins=20, grid=False, layout=(15, 3))
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


# **Detailed Calendar Data for listings in London**

# Sample of data

# In[ ]:


calendar_df.head(1)


# Features with non-zero number of missing values

# In[ ]:


print(calendar_df.shape)
calendar_df.loc[:, calendar_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Summary information for datetime features

# In[ ]:


calendar_df.describe(include='datetime')


# Summary information for string features

# In[ ]:


calendar_df.describe(include='object')


# **Summary Review data and Listing ID**

# Sample of data

# In[ ]:


reviews_df.head(1)


# Features with non-zero number of misssing values

# In[ ]:


print(reviews_df.shape)
reviews_df.loc[:, reviews_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Summary information for datetime features

# In[ ]:


reviews_df.describe(include='datetime')


# Numeric features distribution

# In[ ]:


reviews_df.hist(bins=20, grid=False)
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


# **Detailed Review Data for listings in London**

# Sample of data

# In[ ]:


reviews_detail_df.head(1)


# Features with non-zero number of missing values

# In[ ]:


print(reviews_detail_df.shape)
reviews_detail_df.loc[:, reviews_detail_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Summary information for datetime features

# In[ ]:


reviews_detail_df.describe(include='datetime')


# Summary information for string features

# In[ ]:


reviews_detail_df.describe(include='object')


# Numeric features distribution

# In[ ]:


reviews_detail_df.hist(figsize=(8, 6), bins=20, grid=False)
sns.despine()
plt.suptitle('Numeric features distribution', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


# **Neighbourhood list for geo filter**

# Sample of data

# In[ ]:


neighbourhoods_df.head(1)


# Features with non-zero number of missing values

# In[ ]:


print(neighbourhoods_df.shape)
neighbourhoods_df.loc[:, neighbourhoods_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Summary information for string features

# In[ ]:


neighbourhoods_df.describe(include='object')


# **GeoJSON file of neighbourhoods of the city**

# Sample of data

# In[ ]:


gdf.head(1)


# Features with non-zero number of missing values

# In[ ]:


print(gdf.shape)
gdf.loc[:, gdf.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)


# Summary information for string features

# In[ ]:


gdf.describe(include='object')


# **Data exploration report**  
# Listings data
# - Contains data for host which are registered from 2008-09-03 00:00:00 to 2019-11-04 00:00:00
# - Features
#     - There are a lot of higly skewed numeric features
#     - There are many text features with different descriptions
#     - Data has 3 listings with non-GB country code (2 - France, 1 - Spain) and strange location (according to their information pages)
# - Missing values    
#     - There are features with all null values
#     - There are features which are related to the same entity with the same number of missing values 
#         - There are 12 hosts with most empty host-related columns
#         - There are more than 25% of records with any empty review scores
#     - A lot of features have missing values        
#     
# Calendar data 
# - Contains data for availability and price from 2019-11-05 00:00:00 to 2020-11-04 00:00:00
# - Features
#     - This data is not reliable corresponding to Inside Airbnb project data description (http://insideairbnb.com/about.html)
# - Missing values
#     - There are few missing values for price and renting period information   
#     
# Reviews data
# - Contains data for reviews created from 2009-12-21 00:00:00 to 2019-11-06 00:00:00
# - Features
#     - Data is related to only reviewers and their reviews on specific listings
# - Missing values
#     - There are few missing values only for reviews
#     
# Geo data 
# - Contains geo information for all 33 neighbourhoods of London with their names
# 
# The following information can be useful for the further analysis:  
# - Multiple types of guest review scores
# - Neighbourhood properties
# - Host properties
# - Listing properties
# 
# For further analysis and answering questions we will use only the following datasets:
# - Detailed listings data for London
# - GeoJSON file of neighbourhoods of the city

# ## Data preparation <a class="anchor" id="data-preparation"></a>

# ### Select data <a class="anchor" id="select-data"></a>

# Select necessary columns and rows for further analysis

# In[ ]:


review_cols = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
               'review_scores_communication', 'review_scores_location', 'review_scores_value']
host_cols = ['host_since', 'host_response_time',
             'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']

listing_detail_cols = ['id', 'instant_bookable', 'neighbourhood_cleansed', 'room_type'] + review_cols + host_cols 


res_listings_detail_df = listings_detail_df.query('country_code == "GB"')
res_listings_detail_df = res_listings_detail_df[res_listings_detail_df['host_name'].notnull()]
res_listings_detail_df = res_listings_detail_df[res_listings_detail_df.filter(regex='review_scores').notnull().all(axis=1)]
res_listings_detail_df = res_listings_detail_df[listing_detail_cols].rename({'neighbourhood_cleansed': 'neighbourhood'}, axis=1)
res_listings_detail_df.head()


# Summary information for selected subset of data

# In[ ]:


res_listings_detail_df.info()


# **GeoJSON file of neighbourhoods of the city**

# Select non-null geo data for further analysis

# In[ ]:


geo_cols = ['neighbourhood', 'geometry']
res_gdf = gdf.loc[:, geo_cols]
res_gdf.head()


# **Rationale for inclusion/exclusion**  
# Guest review scores
# - Reason to include
#     - Main features for analysis
# 
# - Features
#     - review_scores_accuracy 
#         - How accurately did the listing page represent the space?
#     - review_scores_cleanliness
#         - Did guests feel that the space was clean and tidy?
#     - review_scores_checkin
#         - How smoothly did check-in go?
#     - review_scores_communication
#         - How well did you communicate before and during the stay?
#     - review_scores_location
#         - How did guests feel about the neighborhood?
#     - review_scores_value
#         - Did the guest feel that the listing provided good value for the price?
#         
# Neighbourhood properties
# - Reasons to include 
#     - Necessary information for calculating and comparing values by different neighbourhoods
# - Features
#     - neighbourhood
#         - Neighbourhood name
#         
# Host properties
# - Reasons to include
#     - We can assume that host properties (related with status, security, responsiveness etc) have relationship with guest review scores
# - Features
#     - host_since
#         - Registration date of the host
#     - host_response_time
#         - Average time of response by host
#     - host_is_superhost
#         - Whether host is superhost
#     - host_has_profile_pic
#         - Whether host has picture in profile
#     - host_identity_verified
#         - Is host's identity verified
#         
# Listing properties
# - Reasons to include 
#     - Different guests can be comfortable with different type of housing to live or process of booking
# - Features
#     - listing_id 
#         - Listing unique identificator
#     - room_type
#         - Type of housing
#     - instant_bookable
#         - Whether additional approval by host is required for booking 
# 
# - 3 listings with non-GB country codes are deleted
# - 12 listings with most empty host-related columns are deleted
# - All listings with any empty review score are deleted
# 
# Geo data
# - Reasons to include 
#     - Necessary information for plotting and merging geo data to listings data later
# - Features
#     - neigbourhood
#         - Neighbourhood name
#     - geometry
#         - Geo data for neighbourhoods (polygons)

# ### Clean data <a class="anchor" id="clean-data"></a>

# **Summary information and metrics for listings in London**

# No cleaning

# **Detailed Listings data for London**

# Convert columns to required types, clean data, and calculate non-aggregated features

# In[ ]:


binary_cols = ['instant_bookable', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']
binary_map = {'f': False, 't': True}
res_listings_detail_df[binary_cols] = res_listings_detail_df[binary_cols].apply(lambda x: x.map(binary_map)).astype(bool)

cat_type = pd.api.types.CategoricalDtype(['not specified', 'within an hour', 'within a few hours', 'within a day', 'a few days or more'])
res_listings_detail_df['host_response_time'] = res_listings_detail_df['host_response_time'].fillna('not specified').astype(cat_type)


# Summary information after cleaning

# In[ ]:


res_listings_detail_df.info()


# **Data cleaning report**  
# 
# Summary information and metrics for listings in London
# - No cleaning
# 
# Detailed Listings data for London
# - Binary columns 
#     - instant_bookable, host_is_superhost, host_has_profile_pic, host_identity_verified
#     - Convert binary columns to boolean type
# - host_response_rate
#     - Clean
#         - Remove "%" symbol
#         - Convert to float type
# - host_response_time
#      - Convert to ordered categorical type
#      - All empty values impute with unique value ("not specified")
#         - Assumption that lack of information is useful for analyzing guest review relationships 

# ### Construct data <a class="anchor" id="construct-data"></a>

# Calculate non-aggregated features based on geodata

# In[ ]:


res_gdf['area_sq_km'] = (res_gdf['geometry'].to_crs({'init': 'epsg:3395'})
                                    .map(lambda p: p.area / 10**6))

res_listings_detail_df['age'] = (pd.Timestamp('now') - pd.to_datetime(res_listings_detail_df['host_since'])).dt.days.div(365.25).round(2)


# **Derived attributes**
# 
# - area_sq_km
#     - Area of the neighbourhood in squared kilometers
# - age
#     - How long host has been registered on Airbnb site

# ### Integrate Data <a class="anchor" id="integrate-data"></a>

# Merge data and calculate aggregated features

# In[ ]:


geo_listings_df = res_gdf.merge(res_listings_detail_df, how='inner', on='neighbourhood')

geo_listings_df['listings_count'] = geo_listings_df.groupby('neighbourhood')['id'].transform('count')
geo_listings_df['listings_density'] = geo_listings_df.groupby('neighbourhood')['area_sq_km'].transform(lambda x: len(x) / x)

geo_listings_df['mean_review_scores_accuracy'] = geo_listings_df.groupby('neighbourhood')['review_scores_accuracy'].transform('mean')
geo_listings_df['mean_review_scores_cleanliness'] = geo_listings_df.groupby('neighbourhood')['review_scores_cleanliness'].transform('mean')
geo_listings_df['mean_review_scores_checkin'] = geo_listings_df.groupby('neighbourhood')['review_scores_checkin'].transform('mean')
geo_listings_df['mean_review_scores_communication'] = geo_listings_df.groupby('neighbourhood')['review_scores_communication'].transform('mean')
geo_listings_df['mean_review_scores_location'] = geo_listings_df.groupby('neighbourhood')['review_scores_location'].transform('mean')
geo_listings_df['mean_review_scores_value'] = geo_listings_df.groupby('neighbourhood')['review_scores_value'].transform('mean')

geo_listings_df['mean_review_scores_all'] = geo_listings_df.filter(like='mean_review_scores').mean(axis=1)


# Summary information after data integration

# In[ ]:


geo_listings_df.info()


# **Merged data**
# - Merge
#     - Type
#         - Inner join
#     - Merge key
#         - Neighborhood name
# - Datasets 
#     - Listings detailed info
#     - Neighbourhoods geo data
# - Aggregates (by neighborhood)
#     - listings_count
#         - Count of listings
#     - listings_density
#         - Density of listings (per square kilometer)
#     - mean_review_scores_*
#         - Mean review scores (overall and by neighbourhood)

# ## Data Understanding <a class="anchor" id="data-understanding-2"></a>

# ### Which neighbourhood is preferable for renting based on guest review scores? <a class="anchor" id="question-1"></a>
# - Different neighbourhoods can be more attractive based on review scores of specific type than the other ones
# - This information can probably be used as indicator whether certain neighbourhood is underestimated or overestimated

# In[ ]:


review_cols = ['mean_review_scores_accuracy', 'mean_review_scores_cleanliness', 'mean_review_scores_checkin',
               'mean_review_scores_communication', 'mean_review_scores_location', 'mean_review_scores_value']
review_titles = ['Accuracy', 'Cleanliness', 'Check-in',
                 'Communication', 'Location', 'Value']
review_map = {col: title for col, title in zip(review_cols, review_titles)}

result_df = geo_listings_df[['geometry', 'neighbourhood', 'mean_review_scores_all'] + review_cols].drop_duplicates()

def gen_map_chart(df, review_col, review_title):
    '''Generate choropleth map
    
    Generate choropleth map based on scores of specific review types
    
    :param df: DataFrame with necessary geo data and review scores for different neighbourhood
    :type df: DataFrame
    :param review_col: name of review scores type
    :type review_col: str
    :param review_title: title of review scores type
    :type review_title: str
    :return: Altair Chart for displaying 
    :rtype: Chart
    '''
    chart = alt.Chart(
        df,
        title=review_title
    ).mark_geoshape().encode(
        color=f'{review_col}:Q',
        tooltip=['neighbourhood:N', f'{review_col}:Q']
    ).properties(
        width=250, 
        height=250
    )
    
    return chart

charts = []

for review_col, review_title in zip(review_cols, review_titles):
    charts.append(gen_map_chart(result_df, review_col, review_title))

overall_map_chart = gen_map_chart(result_df, 'mean_review_scores_all', 'Overall')

((alt.vconcat(alt.concat(*charts, columns=3), overall_map_chart, 
              title='Average review scores by neighbourhood', 
              center=True)
     .configure_view(strokeWidth=0)
     .configure_title(fontSize=18)
     .configure_legend(title=None, orient='top',  labelFontSize=12)))


# **Conclusions**
# - Neighbourhoods in the center of London are more attractive by location for guests, especially on the north side of the Thames (average review score by location):
#     - Kensington and Chelsea (~9.76)
#     - Westminster (~9.73)
#     - Camden (~9.73)
# - Center regions are not so good for guests in other aspects.   
#   It seems like central regions either are too expensive or hosts in the center don't pay such attention to different aspects like other hosts
# - There is probably a bias between different types of reviews, so several review types have much more average score (e.g. Check-in and Communication) than the other ones (e.g. Cleanliness)
# - The best regions in general (corresponding to overall mean guest review scores) are located on the Southwest of London (overall average review score):
#     - Kingston Upon Thames (~9.70)
#     - Richmond Upon Thames (~9.69)
# - There is the region which is worst for most of the review types and in general (overall average review score):
#     - Bexley (~9.2)

# ### Are there any relationships between different types of guest review scores? <a class="anchor" id="question-2"></a>
# - Different review types can be related with the other ones while another reviews can be independent
# - This information can be used for score prediction based on known values of another guest review types

# In[ ]:


result_df = (geo_listings_df[review_cols].rename(review_map, axis=1)
                                         .corr()
                                         .reset_index()
                                         .melt(id_vars='index')
                                         .rename({'value': 'correlation'}, axis=1))

base = alt.Chart(
    result_df,
    title='Average Review Scores Relationship'
).properties(
    width=600, 
    height=600
)

heatmap = base.mark_rect().encode(
    x=alt.X('index:N', title=None),
    y=alt.Y('variable:N', title=None),
    color='correlation:Q'
)

text = base.mark_text(baseline='middle').encode(
    x=alt.X('index:N', title=None),
    y=alt.Y('variable:N', title=None),
    text=alt.Text('correlation:Q', format='.2f'),
    color=alt.condition(
        alt.datum.correlation < 0,
        alt.value('black'),
        alt.value('white')
    )
)

(heatmap + text).configure_axis(
    labelAngle=0,
    labelFontSize=14
).configure_legend(
    orient='top',
    titleFontSize=14,    
).configure_title(
    fontSize=18,
    offset=15,
    anchor='start',
    frame='group'
)


# **Conclusions**
# - There is a strong positive relationship between review scores except for Location review scores
#     - If the guests give high scores for Accuracy, Check-in, Cleanliness, Communication or Value, it is much likely that other 4 review scores will be also high
# - There is a negative relationship between location review scores and the other ones
#     - If the guest give high score to location of the listing, it is much likely that other aspects of the listing will not be such attractive
# - It seems like cleanliness review scores are most independent from other reviews scores

# ### How host/listings properties are related with guest review scores? <a class="anchor" id="question-3"></a>
# - Different aspects of listings and its owners (hosts) can have influence on guest review scores
# - This information can be used to determine possible host and listing features that can be improved

# In[ ]:


def gen_parallel_chart(df, class_col, class_title):
    '''Generate parallel coordinates chart
    
    Generate parallel coordinates chart based on specific class column by different review score types
    
    :param df: DataFrame with necessary data for class column calculation
    :type df: DataFrame
    :param class_col: name of class column 
    :type class_col: str
    :param class_title: title of review scores type
    :type class_title: str
    :return: Altair Chart for displaying 
    :rtype: Chart
    '''
    result_df = (df.groupby(class_col)[review_cols]
                   .mean()
                   .reset_index()
                   .melt(id_vars=class_col))
    result_df['variable'] = result_df['variable'].map(review_map)

    chart = alt.Chart(
        result_df,
        title = f'{class_title}'
    ).mark_line().encode(
        x=alt.X('variable:N',
                title=None),
        y=alt.Y('value:Q',
                scale=alt.Scale(zero=False),
                axis=None),
        color=f'{class_col}:N'
    ).properties(
        width=750, 
        height=300
    )
    
    return chart

class_cols = ['room_type', 'instant_bookable', 'host_is_superhost']
class_titles = ['Room Type', 'Listing is Instant Bookable', 'Host is Superhost']

charts = []

for class_col, class_title in zip(class_cols, class_titles):
    charts.append(gen_parallel_chart(geo_listings_df, class_col, class_title))
    
(alt.concat(*charts, columns=1, title='Average Review Scores by Host/Listing Properties')
    .configure_view(strokeWidth=0)
    .configure_legend(
        title=None, 
        orient='top', 
        columns=0,
        labelFontSize=14)
    .configure_axis(
        labelAngle=0,
        grid=False,
        labelFontSize=14)
    .configure_title(
        anchor='start',
        fontSize=18,
        offset=15)
    .resolve_scale(color='independent')
)


# **Conclusions**
# - Guest reviews for private and shared rooms are consistently greater than for other room types (except for Location)
# - For Location review scores, the best room types are entire home/apartments and hotel rooms
# - If additional approval from host is not required for booking (instant bookable listings), then guest reviews are consistently better (except for Location)
# - If host has the status of Superhost, then guest reviews are consistently better (except for Location)

# ## Evaluation <a class="anchor" id="evaluation"></a>

# ### Evaluate results <a class="anchor" id="evaluate-results"></a>

# **Assessment of data mining results with respect to business success criteria**
# - Different review scores have strong relationship between themselves and with different host/listing/neighbourhood properties
# - These relationships can be used for further analysis how to improve hosts reputation and Airbnb reputation in general

# ### Review process <a class="anchor" id="review-process"></a>

# **Review of process**
# 
# - Data collected from Airbnb site was used to explore relationship of guest reviews with different aspects of hosts and listings
# - Only subset of Airbnb listings with all non-empty review score types were selected for analysis
# - Only small subset of features was selected for analyis based on assumptions which features can have most strong relationship with guest review scores
# - There was no filtering of listings based on availability/price
# - Only descriptive statistics were used for analysis

# ### Determine next steps <a class="anchor" id="next-steps"></a>

# **List of possible actions**
# - Find possible usage of already found relationship to motivate hosts to improve guest experience
# - Continue data understanding
#     - Use another features from listings dataset
#         - Different host/listing properties
#     - Use another files from dataset for further exploration and analysis
#         - Reviews
#             - Review texts
#         - Calendar data
#             - Availabitily
#             - Price
