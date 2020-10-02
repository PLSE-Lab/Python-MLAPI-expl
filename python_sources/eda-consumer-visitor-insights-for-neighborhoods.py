#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel aims to explore this dataset on Visitor information to Census Block Groups.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
print(os.listdir("../input"))


# In[ ]:


dtype = {'census_block_group': 'object', 'date_range_start': 'int', 'date_range_end':'int',
       'raw_visit_count':'float', 'raw_visitor_count':'float', 'visitor_home_cbgs':'object',
       'visitor_work_cbgs':'object', 'distance_from_home':'float', 'related_same_day_brand':'object',
       'related_same_month_brand':'object', 'top_brands':'object', 'popularity_by_hour':'object',
       'popularity_by_day':'object'}
data = pd.read_csv('../input/visit-patterns-by-census-block-group/cbg_patterns.csv', dtype=dtype)
data.info()


# There are some missing values for some columns.

# A list of description of data attributes:
# * census_block_group: The unique 12-digit FIPS code for the Census Block Group. Please note that some CBGs have leading zeroes.
# * date_range_start: Start time for measurement period as a timestamp in UTC seconds.
# * date_range_end: End time for measurement period as a timestamp in UTC seconds.
# * raw_visit_count: Number of visits seen by our panel to this CBG during the date range.
# * raw_visitor_count: Number of unique visitors seen by our panel to this POI during the date range.
# * visitor_home_cbgs: This column lists all the origin home CBGs for devices that visited a destination in the CBG listed in the column census_block_group (the destination CBG). The number mapped to each home CBG indicates the number of visitors observed from this home CBG that visited census_block_group during this time period. Home CBGs with less than 50 visitors to census_block_group are not included.
# * visitor_work_cbgs: This column lists all the work-location CBGs for devices that visited a destination in the CBG listed in the column census_block_group (the destination CBG). The number mapped to each work CBG indicates the number of visitors observed with this work CBG that visited census_block_group during this time period. Work CBGs with less than 50 visitors to census_block_group are not included.
# * distance_from_home: Median distance from home traveled to CBG by visitors (of visitors whose home we have identified) in meters.
# * related_same_day_brand: Brands that the visitors to this CBG visited on the same day as their visit to the CBG where customer overlap differs by at least 5% from the SafeGraph national average to these brands. Order by strength of difference and limited to top ten brands.
# * related_same_month_brand: Brands that the visitors to this CBG visited on the same month as their visit to the CBG where customer overlap differs by at least 5% from the SafeGraph national average. Order by strength of difference and limited to top ten brands.
# * top_brands: A list of the the top brands visited in the CBG during the time period. Limited to top 10 brands.
# * popularity_by_hour: A mapping of hour of the day to the number of visits in each hour over the course of the date range in local time.
# * popularity_by_day: A mapping of day of week to the number of visits on each day (local time) in the course of the date range.

# In[ ]:


data.head()


# There is one NA CBG. This row will be removed.

# In[ ]:


data = data.dropna(subset=['census_block_group'])
data.info()


# NA values for raw_visitor_count will also be removed.

# In[ ]:


data.loc[data.raw_visit_count.isna()].head()


# In[ ]:


data = data.dropna(subset=['raw_visitor_count'])
data.info()


# # Additional Features

# In[ ]:


data['duration'] = data['date_range_end'] - data['date_range_start']
data['duration_days'] = data['duration'] / 86400
data.head()


# In[ ]:


print(data.duration_days.describe())


# All rows have the same observation period of 31 days.

# Geographical information is added.

# In[ ]:


dtype = {'census_block_group': 'object', 'amount_land': 'float', 'amount_water': 'float', 'latitude': 'float', 'longitude': 'float'}
geo = pd.read_csv('../input/census-block-group-american-community-survey-data/safegraph_open_census_data/safegraph_open_census_data/metadata/cbg_geographic_data.csv',dtype=dtype)
geo.info()


# In[ ]:


geo = geo.set_index('census_block_group')


# In[ ]:


data = data.join(geo, on='census_block_group')
data.info()


# NA rows are removed.

# In[ ]:


data = data.dropna(subset=['amount_land'])
data.info()


# In[ ]:


data.sort_values(by='longitude',ascending=False).head(3)


# In[ ]:


adj_val = data.loc[data.longitude==data.longitude.max()].longitude - 360


# Longitude of one CBG was adjusted for better visualisation.

# In[ ]:


data.loc[data.longitude==data.longitude.max(),'longitude'] = adj_val


# In[ ]:


def plot_map(data, column, alpha=0.7, colormap='coolwarm'):
    plt.figure(figsize=(30,30))

    m = Basemap(projection='lcc', 
                resolution='h',
            lat_0=38,
            lon_0=-101,
                llcrnrlon=-125, llcrnrlat=20,
                urcrnrlon=-64, urcrnrlat=47)

    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=2, color='blue')
    m.drawstates()
    m.drawmapboundary()
    
    lons = data["longitude"].values.tolist()
    lats = data["latitude"].values.tolist()

    # Draw scatter plot with all CBGs
    x,y = m(lons, lats)
    m.scatter(x, y, c=data[column], alpha=alpha, cmap=colormap, s=4)
    m.colorbar(location="bottom", pad="4%")
    
    plt.show()


# In[ ]:


plot_map(data,'raw_visit_count')


# In[ ]:


cluster_data = data.loc[:,['latitude','longitude']]


# Each CBG will be grouped under one of 12 clusters. This is firstly done by sampling 20% of the data. This is due to insufficient memory. After which a random forest classifier is built to classify all CBGs into the 12 clusters.

# In[ ]:


c_sample = cluster_data.sample(frac=0.2, random_state=100)
HC = AgglomerativeClustering(n_clusters=12,linkage='ward')
c_sample_label = HC.fit_predict(c_sample[['latitude','longitude']])


# In[ ]:


c_sample['label_HC'] = c_sample_label
c_sample['label_HC'] = c_sample['label_HC'].astype(int)
c_sample.plot(x='longitude',y='latitude', c='label_HC', kind='scatter', colormap='Paired', s=3, figsize=(20,10))
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RF = RandomForestClassifier(random_state=100)
RF.fit(c_sample[['latitude','longitude']], c_sample['label_HC'])


# In[ ]:


cluster_data['label'] = RF.predict(cluster_data)


# In[ ]:


cluster_data.plot(x='longitude',y='latitude', c='label', kind='scatter', colormap='Paired', s=3, figsize=(20,10))
plt.show()


# In[ ]:


data = pd.concat([data, cluster_data['label']], axis=1)


# In[ ]:


plot_map(data,'label', colormap='Paired', alpha=0.9)


# ## Questions
# 
# ### What are the most popular brands in a neighborhood? Are there regional preferences for some brands over others?

# In[ ]:


data['top_brands'] = data.top_brands.map(lambda x:eval(x))


# In[ ]:


brand_dict = {}
for x in data.top_brands:
    for y in x:
        if y.lower() in brand_dict.keys():
            brand_dict[y.lower()] += 1
        else:
            brand_dict[y.lower()] = 1


# In[ ]:


top_10_pop = pd.Series(brand_dict).sort_values().tail(10)
top_10_pop.plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Brands across all CBGs')
plt.show()


# In[ ]:


(top_10_pop/pd.Series(brand_dict).sum()).plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Brands across all CBGs')
plt.xlabel('Percent of total count')
plt.show()


# By counting the occurrence of each brands in the top brand column, I can find out what are the most popular top brands. The most popular brands are:
# 1. United States Postal Service
# 2. Subway
# 3. Dollar General
# 4. McDonald's
# 5. Shell Oil
# 6. Cricket Wireless
# 7. Starbucks
# 8. Family Dollar Stores
# 9. The American Legion
# 10. Walgreens

# In[ ]:


brand_cluster = {}
for label in data.label.unique():
    brand_cluster[label] = {}
    for x in data.loc[data.label==label].top_brands:
        for y in x:
            if y.lower() in brand_cluster[label].keys():
                brand_cluster[label][y.lower()] += 1
            else:
                brand_cluster[label][y.lower()] = 1


# Next, I consider the top brands in each of the regional cluster.

# In[ ]:


labels = [0, 1, 2, 3, 7, 8, 11]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,20), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# For cluster 0, 1, 2, 3, 7, 8 and 11, their most popular brand is USPS. They also have similar top brands (Subway, McDonald's)

# In[ ]:


labels = [10]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,2.5), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster[label]).sort_values().tail(5)
    
    ax.barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster[label]).sum(), color='darkblue', alpha=0.5)
    
    ax.set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    ax.set_xlabel('Percent of total occurrence')

plt.show()


# Cluster 10's most popular top brand is also USPS. One interesting observation is that for this cluster, USPS accounts for more than 8% of the total occurrence. This is more than twice of USPS's percentage of total occurrence at the nationa level.

# In[ ]:


labels = [4, 5, 6, 9]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# For cluster 4, 5, 6, and 9, their most popular brand is not USPS. Only cluster 9 has USPS among its top brands. Cluster 6 has more than 17.5% of its total occurrence coming from department of veteran affairs. This probably indicates there is higher proportion of its residents are veteran.

# ## Conclusion: Top Brands
# 
# There are differences in brands preference across the USA, as shown by the different top brands in each cluster. USPS is the most popular top brands when looking at the aggregated data. It is also the top brand across a number of regional clusters. However, Subway can also be considered the most consistent top brand as it is the only brand that is among the top 5 brands in all regional clusters.

# We can further substantiate our observation of brands preference by analysing the related_same_day_brand and related_same_month_brand columns. These columns provide a more accurate depiction of brand preference as it indicates in each CBG, what brands are visited more than the national average. 

# In[ ]:


data['related_same_day_brand'] = data.related_same_day_brand.map(lambda x:eval(x))


# In[ ]:


brand_day_dict = {}
for x in data.related_same_day_brand:
    for y in x:
        if y.lower() in brand_day_dict.keys():
            brand_day_dict[y.lower()] += 1
        else:
            brand_day_dict[y.lower()] = 1


# In[ ]:


top_10_pop_day = pd.Series(brand_day_dict).sort_values().tail(10)
top_10_pop_day.plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Related Day Brands across all CBGs')
plt.show()


# In[ ]:


(top_10_pop_day/pd.Series(brand_day_dict).sum()).plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Related Day Brands across all CBGs')
plt.xlabel('Percent of total count')
plt.show()


# We can see that the top 10 related same day brands are quite different from the top 10 top brands we seen earlier.
# 
# The top 10 related same day brands are:
# 1. McDonald's (Also in top brands)
# 2. Walmart
# 3. Dunkin' Donuts
# 4. Dollar General (Also in top brands)
# 5. Starbucks (Also in top brands)
# 6. Shell Oil (Also in top brands)
# 7. Sonic
# 8. Quiktrip
# 9. 7-Eleven 
# 10. Casey's General Stores
# 
# Next, I will analyse by regional clusters.

# In[ ]:


brand_cluster_day = {}
for label in data.label.unique():
    brand_cluster_day[label] = {}
    for x in data.loc[data.label==label].related_same_day_brand:
        for y in x:
            if y.lower() in brand_cluster_day[label].keys():
                brand_cluster_day[label][y.lower()] += 1
            else:
                brand_cluster_day[label][y.lower()] = 1


# In[ ]:


labels = [0,1,11]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# Only three regional clusters ( 0 ,1 and 11) have McDonalds as the top related same day brand.

# In[ ]:


labels = [4,8,10]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# Clusters 4, 8 and 10 have Starbucks as their top related same day brand. It would seem like Starbucks is more popular in the west coast as compared to the rest of the USA.

# In[ ]:


labels = [3,7,9]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# Walmart is the top related same day brand for clusters 3, 7 and 9.

# In[ ]:


labels = [2,5,6]
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,10), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_day[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# Cluster 2, 5 and 6 have other brands as their top related same day brand.
# 
# The same analysis was also done for the related same month brand, and the results are quite similar for each clusters. Below are the results.

# In[ ]:


data['related_same_month_brand'] = data.related_same_month_brand.map(lambda x:eval(x))

brand_month_dict = {}
for x in data.related_same_month_brand:
    for y in x:
        if y.lower() in brand_month_dict.keys():
            brand_month_dict[y.lower()] += 1
        else:
            brand_month_dict[y.lower()] = 1


# In[ ]:


top_10_pop_month = pd.Series(brand_month_dict).sort_values().tail(10)
top_10_pop_month.plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Related Month Brands across all CBGs')
plt.show()


# In[ ]:


(top_10_pop_month/pd.Series(brand_month_dict).sum()).plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Related Month Brands across all CBGs')
plt.xlabel('Percent of total count')
plt.show()


# In[ ]:


brand_cluster_month = {}
for label in data.label.unique():
    brand_cluster_month[label] = {}
    for x in data.loc[data.label==label].related_same_month_brand:
        for y in x:
            if y.lower() in brand_cluster_month[label].keys():
                brand_cluster_month[label][y.lower()] += 1
            else:
                brand_cluster_month[label][y.lower()] = 1


# In[ ]:


labels = np.arange(12)
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,30), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_month[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_month[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# ## Conclusion Related Brands
# 
# Earlier, we see that USPS and Subways are popular brands across the USA, meaning there is not much regional  brand preference towards these two brands. By analysing the related brands, we can see that there are more obvious regional brand preferences. 

# In[ ]:


labels=np.arange(12)
top_day = {}
for i, label in enumerate(labels):
    
    top_day[label] = [key for key, value in brand_cluster_day[label].items() if value == max(brand_cluster_day[label].values())][0]

top_month = {}
for i, label in enumerate(labels):
    
    top_month[label] = [key for key, value in brand_cluster_month[label].items() if value == max(brand_cluster_month[label].values())][0]


# In[ ]:


data['top_related_day'] = data.label.map(lambda x: top_day[x])
data['top_related_month'] = data.label.map(lambda x: top_month[x])


# In[ ]:


OE = OrdinalEncoder()
data['top_day_encoded'] = OE.fit_transform(np.array(data['top_related_day']).reshape(-1,1))
data['top_month_encoded'] = OE.fit_transform(np.array(data['top_related_month']).reshape(-1,1))


# In[ ]:


plot_map(data, 'top_day_encoded', colormap='Paired')


# In[ ]:


plot_map(data, 'top_month_encoded', colormap='Paired')


# Now, I want to find out if there are any brands that are both popular in terms of visitor traffic and regional preference i.e. the brand appears in both top brand list and related brand list.

# In[ ]:


data['number_of_top_brand'] = data.top_brands.map(len)
data['number_of_related_day_brand'] = data.related_same_day_brand.map(len)
data['number_of_related_month_brand'] = data.related_same_month_brand.map(len)


# In[ ]:


data['intersection_day_top_brand'] = data.apply(lambda x: list(set(x.related_same_day_brand)&set(x.top_brands)), axis=1)


# In[ ]:


data['intersection_month_top_brand'] = data.apply(lambda x: list(set(x.related_same_month_brand)&set(x.top_brands)), axis=1)


# In[ ]:


brand_day_dict_intersect = {}
for x in data.intersection_day_top_brand:
    for y in x:
        if y.lower() in brand_day_dict_intersect.keys():
            brand_day_dict_intersect[y.lower()] += 1
        else:
            brand_day_dict_intersect[y.lower()] = 1
            
brand_month_dict_intersect = {}
for x in data.intersection_month_top_brand:
    for y in x:
        if y.lower() in brand_month_dict_intersect.keys():
            brand_month_dict_intersect[y.lower()] += 1
        else:
            brand_month_dict_intersect[y.lower()] = 1


# In[ ]:


top_10_pop_day_int = pd.Series(brand_day_dict_intersect).sort_values().tail(10)
top_10_pop_day_int.plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Popular and Related Day Brands across all CBGs')
plt.show()


# In[ ]:


(top_10_pop_day_int/pd.Series(brand_day_dict_intersect).sum()).plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Popular and Related Day Brands across all CBGs')
plt.xlabel('Percent of total count')
plt.show()


# In[ ]:


brand_cluster_day_int = {}
for label in data.label.unique():
    brand_cluster_day_int[label] = {}
    for x in data.loc[data.label==label].intersection_day_top_brand:
        for y in x:
            if y.lower() in brand_cluster_day_int[label].keys():
                brand_cluster_day_int[label][y.lower()] += 1
            else:
                brand_cluster_day_int[label][y.lower()] = 1


# In[ ]:


labels = np.arange(12)
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,30), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_day_int[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_day_int[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# In[ ]:


top_10_pop_month_int = pd.Series(brand_month_dict_intersect).sort_values().tail(10)
top_10_pop_month_int.plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Popular and Related Month Brands across all CBGs')
plt.show()


# In[ ]:


(top_10_pop_month_int/pd.Series(brand_month_dict_intersect).sum()).plot(kind='barh', color='darkblue', alpha=0.5)
plt.title('Top 10 Popular and Related Month Brands across all CBGs')
plt.xlabel('Percent of total count')
plt.show()


# In[ ]:


brand_cluster_month_int = {}
for label in data.label.unique():
    brand_cluster_month_int[label] = {}
    for x in data.loc[data.label==label].intersection_month_top_brand:
        for y in x:
            if y.lower() in brand_cluster_month_int[label].keys():
                brand_cluster_month_int[label][y.lower()] += 1
            else:
                brand_cluster_month_int[label][y.lower()] = 1


# In[ ]:


labels = np.arange(12)
fig, ax = plt.subplots(nrows=len(labels), figsize=(5,30), sharex='all')

for i, label in enumerate(labels):
    
    top_5_pop = pd.Series(brand_cluster_month_int[label]).sort_values().tail(5)
    
    ax[i].barh(top_5_pop.index, top_5_pop.values/pd.Series(brand_cluster_month_int[label]).sum(), color='darkblue', alpha=0.5)
    
    ax[i].set_title('Top 5 brands for Cluster {}'.format(label), loc='left')
    
    if i == len(labels)-1:
        ax[i].set_xlabel('Percent of total occurrence')

plt.show()


# ### How do people travel between neighborhoods? How does distance traveled compare for suburban and urban communities?

# In[ ]:


plt.figure(figsize=(10,5))
data.distance_from_home.plot(kind='hist',bins=50)
plt.show()


# In[ ]:


data['distance_log'] = np.log(data['distance_from_home'])


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='raw_visit_count',y='distance_from_home', data=data)
plt.show()


# ### What times do people visit certain census block groups (ex. Manhattan during the day vs. night)?

# In[ ]:


data['popularity_by_day'] = data.popularity_by_day.map(lambda x:eval(x))


# In[ ]:


data = pd.concat([data, pd.DataFrame.from_dict(data['popularity_by_day'].to_dict(), orient='index')],axis=1)
data.head()


# In[ ]:


data[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].agg('median').plot(kind='bar')
plt.show()


# Sunday has the lowest median number of visitors. From the above plot, there are less visitors during the weekend than on weekdays. This could suggest that many people "visit" other CBGs for work during the weekdays, and many people choose to stay at home/own CBGs during the weekend. Thursday has the lowest median number of visitors. Note that the data only relates to a single month, hence, a public holiday on a Thursday might have lower the number of visitors on Thursday.

# In[ ]:


data['day_sum'] = data['Monday']+data['Tuesday']+data['Wednesday']+data['Thursday']+data['Friday']+data['Saturday']+data['Sunday']


# In[ ]:


for x in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
    data[x+'_prop'] = data[x]/data['day_sum']


# In[ ]:


plt.figure(figsize=(10,8))
sns.violinplot(x='variable', y='value', data=pd.melt(data[['census_block_group','Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']], id_vars=['census_block_group']))
plt.xticks(rotation=60)
plt.show()


# The above plot shows that there are big variations for each day of the week. We can take a look at some of the outliers.

# In[ ]:


data.loc[data[['Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']].agg('max', axis=1)>0.5].sort_values(by='day_sum',ascending=False).head()


# In[ ]:


data.loc[data[['Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']].agg('max', axis=1)>0.5].sort_values(by='day_sum',ascending=False).head(1)[['Monday_prop','Tuesday_prop','Wednesday_prop','Thursday_prop','Friday_prop','Saturday_prop','Sunday_prop']].plot(kind='bar')
plt.show()


# In[ ]:


data['popularity_by_hour'] = data.popularity_by_hour.map(lambda x:eval(x))


# In[ ]:


data = pd.concat([data, pd.DataFrame.from_records(data['popularity_by_hour'].to_dict()).transpose()], axis=1)
data.head()


# In[ ]:


plt.figure(figsize=(10,5))
data[np.arange(24)].agg('median').plot(kind='bar', color='grey')
plt.xticks(rotation=0)
plt.show()


# This plot shows the expected trend of number of visitors throughout a single day. At the peak hour of 0700 and 1700, there are the most number of visitors. This is in line with the usual business working hours.

# In[ ]:


data['hour_sum'] = data[np.arange(24)].sum(axis=1)


# In[ ]:


for x in np.arange(24):
    data[str(x)+'_prop'] = data[x]/data['hour_sum']


# In[ ]:


plt.figure(figsize=(20,10))
sns.violinplot(x='variable', y='value', data=pd.melt(data[['census_block_group']+[str(x)+'_prop' for x in np.arange(24)]], id_vars=['census_block_group']))
plt.xticks(rotation=90)
plt.show()


# In[ ]:


data.loc[data[[str(x)+'_prop' for x in np.arange(24)]].agg('max', axis=1)>0.2].sort_values(by='hour_sum',ascending=False).head()


# In[ ]:


data.loc[data[[str(x)+'_prop' for x in np.arange(24)]].agg('max', axis=1)>0.2].sort_values(by='hour_sum',ascending=False).head(1)[[str(x)+'_prop' for x in np.arange(24)]].plot(kind='bar')
plt.legend().remove()
plt.show()


# ### Which neighborhoods are the most mobile? Which neighborhoods receive the most outside visitors?

# In[ ]:


data.raw_visit_count.plot(kind='hist',bins=20)
plt.show()


# In[ ]:


data.raw_visitor_count.plot(kind='hist',bins=20)
plt.show()


# In[ ]:


data['percent_unique_visitor'] = data['raw_visitor_count']/data['raw_visit_count']
data['percent_unique_visitor'].plot(kind='hist',bins=50)
plt.show()


# A high percent of unique visitors indicate that this CBG is most likely a tourist attraction, where there are no regular visitors.

# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='raw_visit_count',y='raw_visitor_count', data=data)
plt.show()


# This is an interesting plot as you can see there are almost two separate relationships, indicating there may be two broad categories of CBGs.

# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(y='raw_visit_count',x='percent_unique_visitor', data=data)
plt.show()


# In[ ]:


data['raw_visit_count_log'] = np.log(data['raw_visit_count'])
data['raw_visitor_count_log'] = np.log(data['raw_visitor_count'])


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='raw_visit_count_log',y='percent_unique_visitor', data=data)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(y='distance_from_home',x='percent_unique_visitor', data=data)
plt.show()


# In[ ]:


data['visitor_home_cbgs'] = data.visitor_home_cbgs.map(lambda x:eval(x))
data['visitor_work_cbgs'] = data.visitor_work_cbgs.map(lambda x:eval(x))


# In[ ]:


data['number_of_home_cbgs'] = data.visitor_home_cbgs.map(len)
data['number_of_work_cbgs'] = data.visitor_work_cbgs.map(len)


# In[ ]:


home_cbgs = {}
for x in data.visitor_home_cbgs:
    for key, value in x.items():
        if key in home_cbgs.keys():
            home_cbgs[key] += np.array([1, value])
        else:
            home_cbgs[key] = np.array([1, value])


# In[ ]:


home_cbgs_df = pd.DataFrame.from_dict(home_cbgs, orient='index',columns=['number_of_occurrence_home','total_visitors_count_home'])


# In[ ]:


work_cbgs = {}
for x in data.visitor_work_cbgs:
    for key, value in x.items():
        if key in work_cbgs.keys():
            work_cbgs[key] += np.array([1, value])
        else:
            work_cbgs[key] = np.array([1, value])
            
work_cbgs_df = pd.DataFrame.from_dict(work_cbgs, orient='index',columns=['number_of_occurrence_work','total_visitors_count_work'])


# In[ ]:


total = pd.concat([home_cbgs_df,work_cbgs_df],axis=1, sort=True)
total.fillna(0, inplace=True)
total.head()


# In[ ]:


total['log_home'] = np.log(total.total_visitors_count_home+1)
total['log_work'] = np.log(total.total_visitors_count_work+1)


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='total_visitors_count_home',y='total_visitors_count_work', data=total)
plt.show()


# In[ ]:


total.log_home.plot(kind='hist',bins=60)


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='log_home',y='log_work', data=total)
plt.show()


# In[ ]:


data = data.join(total, on='census_block_group')


# In[ ]:


data.loc[:,total.columns] = data.loc[:,total.columns].fillna(0)


# In[ ]:


data.loc[(data.number_of_occurrence_home!=0)&(data.number_of_home_cbgs!=0)]


# In[ ]:


plt.figure(figsize=(10,10))
sns.regplot(x='number_of_work_cbgs',y='number_of_occurrence_work', data=data.loc[(data.number_of_occurrence_work!=0)&(data.number_of_work_cbgs!=0)])
plt.show()


# In[ ]:




