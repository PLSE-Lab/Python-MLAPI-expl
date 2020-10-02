#!/usr/bin/env python
# coding: utf-8

# # Contents
# 
# #### 1. Foreword
# #### 2. Data Load and Libraries Import
# #### 3. Data Cleaning and Validity
# *  3.1 Nulls and Empty
# *    3.2 Valididty Check by SQL
# *    3.3 Georaphical Validity
# 
# #### 4. Exploratory Data Analysis
# *  4.1 SQL Script for Data Preparation
# *  *  4.1.1 Dumping Data in NY_2
# *  4.2 Simple data Analysis
# *  *  4.2.1 Neighbourhood Groups
# *  *  4.2.2 Neighbourhoods Analysis -- Applying Pareto Principle (80/20 rule) 
# *  4.3 Inference
# 
# #### 5. Data Analysis by availability
# *  5.1 Inference
# 
# #### 6. Data Analysis by Price
# *  6.1 Inference
# 
# #### 7. Geographic Analysis
# *  7.1 inference

# # 1. Foreword
# 
# This Notebook is created for learning purpose for beginners specially for those who have very little knowledge of Python but have nice experience with other programming languages for example c#, java, c++, SQL. I will be using lot od SQL in there for data wrangling instead of Pandas or any other library.
# 
# In addition to that I have created a small utility to load data from/to CSV/SQL while I will upload once it gets stabalized.

# # 2. Data Load and Libraries Import

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


# In[ ]:


data=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_org.csv')
data.head(8)


# # 3. Data Cleaning and Validity

# ## 3.1 Nulls and Empty

# In[ ]:


data.isnull().sum()


# Since I have to transafer data to SQL name and host name can be considered as Empty but I am replacing last_review with *__'1/1/1900'__* and reviews_per_month is replaced with __0__. Although SQL can consider NULL values its just that my utility is not that much evolved yet.
# 
# So after doing so following are the results for same isnull function:
# 

# In[ ]:


data=pd.read_csv('../input/ny_airbnb/AB_NYC_2019.csv')
data.isnull().sum()


# ## 3.2 Validity check by SQL
# 
# I am running a few queries and listing down its results to discard rows tat looks in valid or not usable

# In[ ]:


# SELECT COUNT(*) FROM NY1 WHERE Price <= 0
# Result: 11 --- will remove those 11 rows

# SELECT COUNT(*) FROM NY1 WHERE Price <= 0
# Result: 0 --- good to go

# SELECT COUNT(*) FROM NY1 WHERE availability_365 <= 0
# Result: 17533 --- thats is shocking, more than one third of property listed has never been available


# Removing this might not be a good idea since it is atleast giving us the data that these property might be available in future.
# 
# We can utilize its longitude and latitude but we should not use this data for price comparison, since they have never been rented.
# 
# We can add a flag/column in our dataframe depicting if the property is avilable or not

# ## 3.3 Geographical Validity
# 
# Here is the image of NewYork map with neighbourhood_group borders and do a scatter plot with neighbourhood_group as hue, that will easily point out outliers

# In[ ]:


#Courtesy Dgmonov, I didn't know earler how to use image while plotting

plt.figure(figsize=(20,16))
nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')
#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group, ax=ax)


# Cant find any outlier atleast not with respect to neighbhourhood_group, so moving forward to Simple dta analysis.

# ## 4. Exploratory Data Analysis
# 
# Now lets start with simple Explorartory Data Analysis. For this as the data is already transferred to SQL so there are three things that we would like to do with data
# 
# ##### 1) Remove the rows with 0 price
# 
# ##### 2) Categorize data with respect to Prices
# 
# ##### 3) Categorize data wit respect to Availability

# ## 4.1 SQL Script for data preperation
# 
# We will be keeping the originial table (NY1) as it is while creating new table name NY_2

# In[ ]:


# CREATE TABLE [dbo].[NY_2](

# 	[id] [int] NULL,
#   [name] [varchar](max) NULL,
#   [host_id] [int] NULL,
# 	[host_name] [varchar](max) NULL,
# 	[neighbourhood_group] [varchar](max) NULL,
# 	[neighbourhood] [varchar](max) NULL,
# 	[latitude] [decimal](18, 9) NULL,
# 	[longitude] [decimal](18, 9) NULL,
# 	[room_type] [varchar](max) NULL,
# 	[price] [int] NULL,
# 	[minimum_nights] [int] NULL,
# 	[number_of_reviews] [int] NULL,
# 	[last_review] [datetime] NULL,
# 	[reviews_per_month] [decimal](18, 9) NULL,
# 	[calculated_host_listings_count] [int] NULL,
# 	[availability_365] [int] NULL,
# 	[Price_Range_100] [varchar](max) NULL,
# 	[Price_Range_Sequence] INT,
# 	[Availability_Range] [varchar](max) NULL,
# 	[Availibility_Range_Sequence] INT
# ) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]


# As you can see here that 4 columns are increased, that is 2 for each category, one depicts name of Category while other depicts its sequence. 
# 
# Now lets fill the table first and we shall update the newly created column afterwards

# ### 4.1.1 Dumping data in NY_2

# In[ ]:


# INSERT INTO NY_2(id,name,host_id,host_name,neighbourhood_group,neighbourhood,latitude,longitude,room_type,price,minimum_nights,number_of_reviews,last_review,reviews_per_month,calculated_host_listings_count,availability_365)
# SELECT * FROM NY1
# WHERE Price > 0

##### Updating Category columns

# UPDATE NY_2
# SET [Price_Range_100] = CASE 

# 	WHEN price BETWEEN 0 AND 100 THEN '0-100'
# 	WHEN price BETWEEN 101 AND 200 THEN '101-200'
# 	WHEN price BETWEEN 201 AND 300 THEN '201-300'
# 	WHEN price BETWEEN 301 AND 400 THEN '301-400'
# 	WHEN price BETWEEN 401 AND 500 THEN '401-500'
# 	WHEN price BETWEEN 501 AND 600 THEN '501-600'
# 	WHEN price BETWEEN 601 AND 700 THEN '601-700'
# 	WHEN price BETWEEN 701 AND 800 THEN '701-800'
# 	WHEN price BETWEEN 801 AND 900 THEN '801-900'
# 	WHEN price BETWEEN 901 AND 1000 THEN '901-1000'
# 	ELSE '> 1000' END 
        
# , [Price_Range_Sequence] = CASE 

# 	WHEN price BETWEEN 0 AND 100 THEN 1
# 	WHEN price BETWEEN 101 AND 200 THEN 2
# 	WHEN price BETWEEN 201 AND 300 THEN 3
# 	WHEN price BETWEEN 301 AND 400 THEN 4
# 	WHEN price BETWEEN 401 AND 500 THEN 5
# 	WHEN price BETWEEN 501 AND 600 THEN 6
# 	WHEN price BETWEEN 601 AND 700 THEN 7
# 	WHEN price BETWEEN 701 AND 800 THEN 8
# 	WHEN price BETWEEN 801 AND 900 THEN 8
# 	WHEN price BETWEEN 901 AND 1000 THEN 10
# 	ELSE 11 END 
#         
# , [Availability_Range] =  CASE 
# 
# 	WHEN [availability_365] = 0 THEN 'No Availability'
# 	WHEN [availability_365] BETWEEN 1 AND 100 THEN 'Low Avalability'
# 	WHEN [availability_365] BETWEEN 101 AND 200 THEN 'Medium Availablity'
# 	WHEN [availability_365] BETWEEN 201 AND 300 THEN 'High Availablity'
# 	WHEN [availability_365] BETWEEN 301 AND 365 THEN 'Nearly Always Availability'
# 	ELSE 'Invalid' END
#        
# , [Availibility_Range_Sequence] = CASE 
# 
# 	WHEN [availability_365] = 0 THEN 0
# 	WHEN [availability_365] BETWEEN 1 AND 100 THEN 1
# 	WHEN [availability_365] BETWEEN 101 AND 200 THEN 2
# 	WHEN [availability_365] BETWEEN 201 AND 300 THEN 3
# 	WHEN [availability_365] BETWEEN 301 AND 365 THEN 4
# 	ELSE -1 END
#         
# 	FROM NY_2


# In[ ]:


### Lets Have a look at the Data we have prepared so far
data_cat=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_4.csv')
data_cat.head()


# ## 4.2 Simple Data Analysis

# ### 4.2.1 Neighbourhood Groups

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
title = plt.title('Number of listings by Naighbourhood Groups', fontsize=20)
title.set_position([-0.2, 1.15])
patches, texts = plt.pie(data['neighbourhood_group'].value_counts(), startangle=90, pctdistance=0.85)
plt.legend(patches, data['neighbourhood_group'], loc="best")

centre_circle = plt.Circle((0,0),0.4,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.tight_layout()

sns.countplot('neighbourhood_group',data=data,ax=ax[0])
ax[0].set_xlabel('Neighbourhood Groups')
ax[0].set_ylabel('Number of Listings')
plt.show()


# ### 4.2.2 Neighbourhoods Analysis -- Applying Pareto Principle (80/20 rule) 

# Total Number of Neigbourhoods: **221**
# **20% of  221 is 44**
# 
# Picking up __*Top 35 neighbourhoods*__  since it loks like presenting 44 neighbourhoods in a chart will not be a good idea, so further reducing number of neighbourhoods that is with follwing SQL Query

# In[ ]:


# SELECT TOP 35 neighbourhood, count(*) FROM NY_2
# GROUP BY neighbourhood
# ORDER BY COUNT(*) DESC

#Resulted in following
data_neighbourhood=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_7.csv')
data_neighbourhood


# ##### Information:
# 
# Number of listings in top 35 Neighbourhoods: 38795 (79.34%)
# Percentage of Neighbouroods: 15.84%
# 
# <font color=red> __*Under 16% of Neighbourhoods contains about 80% of data*__ </font>

# In[ ]:


#Lets run pie on this
plt.figure(figsize=(15,11))
plt.title('Number of listings by Top 35 Neighbourhoods', fontsize=20)
patches, texts = plt.pie(data_neighbourhood['COUNT'], startangle=90, pctdistance=0.85)
plt.legend(patches, data_neighbourhood['neighbourhood'], loc="upper right")

centre_circle = plt.Circle((0,0),0.4,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.tight_layout()
plt.show()


# #### Neighbourhood with more than 500 listings
# 
# Further narrowing our data, that is selecting only those neighbourhoods that have more than 500 listings

# In[ ]:


# SELECT TOP 35 neighbourhood, count(*) AS [COUNT] FROM NY_2
# GROUP BY neighbourhood
# HAVING COUNT(*) > 500
# ORDER BY COUNT(*) DESC

#Resulted in following
data_neighbourhood_500=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_8.csv')
data_neighbourhood_500


# ##### Information:
# 
# Number of listings in top 23 Neighbourhoods: 33775 (69.03%)
# Percentage of Neighbouroods: 10.04%
# 
# <font color=red> __*So just 10% of Neighbourhoods contains about 70% of data*__ </font>

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,10))
title = plt.title('Number of listings by Naighbourhood having more than 500 records', fontsize=20)
title.set_position([0.0, 1.15])
patches, texts = plt.pie(data_neighbourhood_500['COUNT'], startangle=90, pctdistance=0.85)
plt.legend(patches, data_neighbourhood_500['neighbourhood'], loc="upper left")

centre_circle = plt.Circle((0,0),0.4,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.tight_layout()

chart = sns.barplot(data=data_neighbourhood_500,x='neighbourhood',y='COUNT',ax=ax[0])
ax[0].set_xlabel('Neighbourhoods')
ax[0].set_ylabel('Number of Listings')
ax[0].tick_params(axis='x', labelrotation=45)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Number of listings by Room Type', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.countplot(x="room_type", data=data_cat)
ax.set_xlabel('Room Type')
ax.set_ylabel('Listings')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right')


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Number of Reviews by Room Type', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.barplot(x="room_type", y="number_of_reviews", data=data_cat, ci=None)
ax.set_xlabel('Room Type')
ax.set_ylabel('Reviews')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Distribution of Reviews by Neighbourhood Groups', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.violinplot(x="neighbourhood_group", y="number_of_reviews", data=data_cat)
ax.set_xlabel('Neighbourhood Group')
ax.set_ylabel('Reviews')
ax.set(ylim=(0, 200))
c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Distribution of Reviews by Room Type', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.violinplot(x="room_type", y="number_of_reviews", data=data_cat)
ax.set_xlabel('Room Types')
ax.set_ylabel('Reviews')
ax.set(ylim=(0, 200))
c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


heatmap_data_roomtype_ng_price = pd.pivot_table(data_cat, values='price', 
                     index=['room_type'], 
                     columns='neighbourhood_group')

plt.figure(figsize=(20,6))
title = plt.title('Room Type, Neighbourhood Groups and Price Relation', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.heatmap(heatmap_data_roomtype_ng_price, cmap="YlGnBu", cbar_kws={'label': 'Price'})
ax.set_xlabel('Neighbourhood Groups')
ax.set_ylabel('Room Type')
a = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# ## 4.3 Inference
# 
# With the simple data analysis we can infer that:
# 
# 1. Data is avaialable for 5 Neighbourood Groups
# 
# 2.  There are 221 Neighbourhoods listed in data
# 
# 3.  Pareto rule is quite applicable 16% of Neighbourhoods have 80% data while top 10% Neighbourhoods have 70% of data
#  *  We will surely apply thatwhile doing Data analysis by categories
# 
# 4.  There were 11 records that had 0 price so they are removed
# 
# 5.  There are 17533 records with no avilability, that will be further discussed in data analysis by Categories
# 
# 6. Though Entire home/appartments have higher number of listings but Private room have more number of reviews
#  *  We shall look into relation of Reviews and prices later
#         
# 7. The two least listed Neighbourhood groups have greater number of reviews per property
# 
# 8. Manhattan is ofcourse the costliest in case of any type of propety.
#  *  Staten Island in case of Entire Appartment has more price than any other Neighbourhood Group ofcourse except Manhattan 

# ## Data Analysis by Categories
# 
# As we have aready added four new columns to categorize our data in two ways
# 
# ##### 1) Prince Range
# Categorized in 11 categories with the interval of 100 starting with 0, which reated 10 categories till 1000 while for price more than 1000 just one category is created
# 
# ##### 2) Availability
# Categorized in 5 categories:
# 
#     1. Not Available - Property has never been up for rent in a year
#     2. Low Availablity - Property avilable for less than or just for 100 days in a year
#     3. Medium Availability - Property available for 101 to 200 days in a year
#     4. High Availability - Property available for more than 200 but only upto 300 days in a year
#     5. Nearly Always Available - Property available for more than 300 days in a year
# 
# Following are the SQL queries and their results

# In[ ]:


#SELECT Price_Range_100, COUNT(*) AS [COUNT]
#FROM NY_2
#GROUP BY Price_Range_100, Price_Range_Sequence
#ORDER BY Price_Range_Sequence
data_price_count=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_5.csv')
data_price_count


# In[ ]:


#SELECT [Availability_Range], COUNT(*) AS [COUNT]
#FROM NY_2
#GROUP BY [Availability_Range], [Availibility_Range_Sequence]
#ORDER BY [Availibility_Range_Sequence]
data_availability_count=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_6.csv')
data_availability_count


# ## 5 Data Analysis by Availability:

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
plt.title('Share of listings by Availability', fontsize=20)
wedges, patches, texts = ax[1].pie(data_availability_count['COUNT'], startangle=90, labels=data_availability_count['Availability_Range'], pctdistance=0.6,autopct='%1.1f%%')

centre_circle = plt.Circle((0,0),0.4,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.tight_layout()

chart = sns.barplot(data=data_availability_count,x='Availability_Range',y='COUNT',ax=ax[0])
ax[0].set_title('Share of listings by Availability', fontsize=20)
ax[0].set_xlabel('Availability')
ax[0].set_ylabel('Number of Listings')
chart.set_xticklabels(chart.get_xticklabels(), rotation=0, horizontalalignment='center')
plt.show()


# #### Removing Zero Availability
# 
# Removing records with no avilability and drawing again:

# In[ ]:


data_availability_count_nonzero=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_9.csv')
f,ax=plt.subplots(1,2,figsize=(20,10))
plt.title('Share of listings by Availability with non zero availability', fontsize=20)
wedges, patches, texts = plt.pie(data_availability_count_nonzero['COUNT'], startangle=90, labels=data_availability_count_nonzero['Availability_Range'], pctdistance=0.6,autopct='%1.1f%%')

centre_circle = plt.Circle((0,0),0.4,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.tight_layout()

chart = sns.barplot(data=data_availability_count_nonzero,x='Availability_Range',y='COUNT',ax=ax[0])
ax[0].set_title('Share of listings by Availability with non zero availability', fontsize=20)
ax[0].set_xlabel('Availability')
ax[0].set_ylabel('Number of Listings')
chart.set_xticklabels(chart.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


#Including more Parameters
plt.figure(figsize=(10,10))
title = plt.title('Number of listings by Availability and Neighbourhood Groups', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.countplot(x="Availability_Range", data=data_cat, hue='neighbourhood_group')
ax.set_xlabel('Availability')
ax.set_ylabel('Number of Listings')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
ax.titlesize = 'large'   


# In[ ]:


#### Removing Zero Availability

# Removing records with zero availability from *data_cat* and loading into *data_cat_nonzero* by following __SQL__

# SELECT * FROM NY_2
# WHERE availability_365 > 0
# Resulted in 31354 records

data_cat_nonzero=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_10.csv')


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Number of listings by Availability and Neighbourhood Groups', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.countplot(x="Availability_Range", data=data_cat_nonzero, hue='neighbourhood_group')
ax.set_xlabel('Availability')
ax.set_ylabel('Number of Listings')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Number of listings by Availability and Room Type', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.countplot(x="Availability_Range", data=data_cat_nonzero, hue='room_type')
ax.set_xlabel('Availability')
ax.set_ylabel('Number of Listings')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')


# In[ ]:


f,ax=plt.subplots(1,1,figsize=(20,10))

chart1 = sns.barplot(x="Availability_Range", y="number_of_reviews", data=data_cat, ci=None)
title = ax.set_title('Number of Reviews by Availability', fontsize=20)
title.set_position([0.5, 1.15])
ax.set_xlabel('Availability')
ax.set_ylabel('Number Reviews')
chart1.set_xticklabels(chart1.get_xticklabels(), rotation=0, horizontalalignment='center')

plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Number of Reviews by Availability and Appartment Type', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.barplot(x="Availability_Range", y="number_of_reviews", data=data_cat_nonzero, hue='room_type', ci=None)
ax.set_xlabel('Availability')
ax.set_ylabel('Reviews')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')


# In[ ]:


#Another incoming from SQL
# SELECT Availibility_Range_Sequence, neighbourhood_group, price
# FROM NY_2
# ORDER BY Availibility_Range_Sequence

data_av_heatmap=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_11.csv')
heatmap_data = pd.pivot_table(data_av_heatmap, values='price', 
                     index=['Availibility_Range_Sequence'], 
                     columns='neighbourhood_group')

plt.figure(figsize=(10,6))
title = plt.title('Availability, Neighbour Group, and Price comparison', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Price'})
ax.set_xlabel('Neighbourhood Groups')
ax.set_ylabel('Availability')
y_label_list = ['No Availability', 'Low Availablity', 'Medium Availablity', 'High Availablity', 'Nearly Always Availability']
a = ax.set_yticklabels(y_label_list, rotation=0)
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


# Heatmap with respect to Top 10 Neighbourhoods
# SELECT Availibility_Range_Sequence, neighbourhood, price FROM NY_2
# WHERE neighbourhood IN
# (
# 	SELECT TOP 10 neighbourhood
# 	FROM NY_2
# 	GROUP BY neighbourhood
# 	ORDER BY COUNT(*) DESC
#)

data_av_heatmap=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_12.csv')
heatmap_data = pd.pivot_table(data_av_heatmap, values='price', 
                     index=['Availibility_Range_Sequence'], 
                     columns='neighbourhood')

plt.figure(figsize=(20,6))
title = plt.title('Availability, Top 10 Neighbourhoods, and Price comparison', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Price'})
ax.set_xlabel('Neighbourhoods')
ax.set_ylabel('Availability')
y_label_list = ['No Availability', 'Low Availablity', 'Medium Availablity', 'High Availablity', 'Nearly Always Availability']
a = ax.set_yticklabels(y_label_list, rotation=0)
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


# In[ ]:


# SELECT Availibility_Range_Sequence, room_type, price
# FROM NY_2
# ORDER BY Availibility_Range_Sequence

data_av_heatmap_3=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_13.csv')
heatmap_data = pd.pivot_table(data_av_heatmap_3, values='price', 
                     index=['Availibility_Range_Sequence'], 
                     columns='room_type')

plt.figure(figsize=(10,6))
title = plt.title('Availability, Room Type, and Price comparison', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Price'})
ax.set_xlabel('Room Type')
ax.set_ylabel('Availability')
y_label_list = ['No Availability', 'Low Availablity', 'Medium Availablity', 'High Availablity', 'Nearly Always Availability']
a = ax.set_yticklabels(y_label_list, rotation=0)
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# ## 5.1 Inference
# 
# 1) One third of the data have zero availability, means they are nothing more than addresses, and should not be considered r considered separately when applying availability with other parameters for example Price which we will see next.
# 
# 2) Manhattan has the hoghest number of listings in all sorts of avilability except for one that is Low Availability.
# 
# 3) Strangely properties that are always available have lower number of reviews
# 
# 4) Property that is not available also reviews, means it was available before the data is extracted.
# 
# 5) Number of reviews is still higher in case of private room compared with any other type except in case of high availability 
# 
# 6) Price is directly proportional to Availability, in both cases of Neighbourhood Groups and Top 10 Neighbourhoods.
# 

# # 6. Data Analysis by Price

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,10))
title = plt.title('Number of listings by Price', fontsize=20)
title.set_position([0.0, 1.15])
patches, texts = plt.pie(data_price_count['COUNT'], startangle=90, pctdistance=0.6)
plt.legend(patches, data_price_count['Price_Range_100'], loc="upper right")
ax[1].set_title('Number of listings by Price', fontsize=20)

centre_circle = plt.Circle((0,0),0.4,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')  
plt.tight_layout()

chart = sns.barplot(data=data_price_count,x='Price_Range_100',y='COUNT',ax=ax[0])
ax[0].set_xlabel('Price')
ax[0].set_ylabel('Number of Listings')
chart.set_xticklabels(chart.get_xticklabels(), rotation=35, horizontalalignment='right')
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Price comparison by Room Type', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.violinplot(x="room_type", y="price", data=data_cat_nonzero)
ax.set_xlabel('Room Type')
ax.set_ylabel('Price')
ax.set(ylim=(0, 600))
c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


plt.figure(figsize=(10,6))
title = plt.title('Price comparison by Neighbourhood Groups', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.violinplot(x="neighbourhood_group", y="price", data=data_cat_nonzero)
ax.set_xlabel('Neighbourhood Groups')
ax.set_ylabel('Price')
ax.set(ylim=(0, 600))
c = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


# SELECT * FROM NY_2
# WHERE neighbourhood IN
# (
# 	SELECT TOP 20 neighbourhood
# 	FROM NY_2
# 	WHERE availability_365 > 0
# 	GROUP BY neighbourhood
# 	ORDER BY COUNT(*) DESC
# )
# AND availability_365 > 0
data_top_10_neighbourhoods=pd.read_csv('../input/ny_airbnb/AB_NYC_2019_14.csv')

plt.figure(figsize=(25,6))
plt.title('Price comparison by Top 10 Neighbourhoods', fontsize=20)
ax = sns.violinplot(x="neighbourhood", y="price", data=data_top_10_neighbourhoods)
ax.set_xlabel('Neighbourhoods')
ax.set_ylabel('Price')
ax.set(ylim=(0, 500))
c = ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


# In[ ]:


ax = sns.jointplot(y="price", x="availability_365", data=data_cat_nonzero,xlim=(0, 365), ylim=(0, 500), height=8, ratio=5)
l = ax.set_axis_labels("Availaility", "Price")
plt.subplots_adjust(top=0.9)
plt.suptitle('Price and availability distribution', fontsize = 16)
plt.show()


# In[ ]:


heatmap_data_ng_price_reviews = pd.pivot_table(data_cat_nonzero, values='number_of_reviews', 
                     index=['neighbourhood_group'], 
                     columns='Price_Range_100')

plt.figure(figsize=(20,6))
title = plt.title('Reviews by Neighbourhood and Price Range', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.heatmap(heatmap_data_ng_price_reviews, cmap="YlGnBu", cbar_kws={'label': 'Reviews'})
ax.set_xlabel('Price Range')
ax.set_ylabel('Neighbourhood Groups')
a = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# In[ ]:


heatmap_data_rt_price_reviews = pd.pivot_table(data_cat_nonzero, values='number_of_reviews', 
                     index=['room_type'], 
                     columns='Price_Range_100')

plt.figure(figsize=(20,6))
title = plt.title('Reviews by Room Type and Price Range', fontsize=20)
title.set_position([0.5, 1.15])
ax = sns.heatmap(heatmap_data_rt_price_reviews, cmap="YlGnBu", cbar_kws={'label': 'Reviews'})
ax.set_xlabel('Price Range')
ax.set_ylabel('Room Type')
a = ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
a = ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')


# ## 6.1 Inference
# 
# 1. Prices thhat fall in lower prices category like 0-100 and 101-200 has more records than all the oter records combined
# 
# 2. Manhattan and Broklyn has more records in between 100 to 200 than any other borough
# 
# 3. Financial District and West Village has more records with prices more than 100
# 
# 4. Though number of reviews mostly decline with increase in price but for price between 801-900 has more reviews for boroughs Queens and Manhattan **&** for Private Room and Entire Appartment

# # 7. Geograhical Analysis

# In[ ]:


import urllib
f,ax=plt.subplots(1,2,figsize=(25,10))
nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')
#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot
ax[0].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
map1 = sns.scatterplot(data_cat.longitude,data_cat.latitude,hue=data_cat.neighbourhood_group, ax=ax[0])
title = ax[0].set_title('Data with 0 Availability', fontsize=20)
title.set_position([0.5, 1.1])

ax[1].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
map2 = sns.scatterplot(data_cat_nonzero.longitude,data_cat_nonzero.latitude,hue=data_cat_nonzero.neighbourhood_group, ax=ax[1])
title = ax[1].set_title('Data without 0 Availability', fontsize=20)
title.set_position([0.5, 1.1])
plt.show()


# In[ ]:


ax=plt.figure(figsize=(20,8))
nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
sns.scatterplot(data=data_cat[data_cat['Availability_Range']=='No Availability'], x='longitude',y='latitude',hue='Availability_Range', ax=ax)
title = ax.set_title('Data for No Availability', fontsize=20)
title.set_position([0.5, 1.1])


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,10))
nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')
#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot
ax[0].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

map1 = sns.scatterplot(data = data_cat_nonzero,x='longitude',y='latitude',hue='Price_Range_100', ax=ax[0], palette='RdBu')
title = ax[0].set_title('Distribution by Price', fontsize=20)
title.set_position([0.5, 1.1])

ax[1].imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
map2 = sns.scatterplot(data = data_cat_nonzero,x='longitude',y='latitude',hue='Availability_Range', ax=ax[1])
title = ax[1].set_title('Distribution by Availability', fontsize=20)
title.set_position([0.5, 1.1])
plt.show()


# In[ ]:


ax=plt.figure(figsize=(20,8))
nyc_img=plt.imread('../input/ny_airbnb_img/Neighbourhoods_New_York_City_Map.PNG')
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
sns.scatterplot(data=data_cat_nonzero, x='longitude',y='latitude',hue='room_type', ax=ax)
title = ax.set_title('Distribution by Room Type', fontsize=20)
title.set_position([0.5, 1.1])


# ## 7.1 Inference
# 
# 1. Listings with No Availability are scattered throughout the city
# 
# 2. In Manhattan there are more records for Low Availability while in Staten Island there are more records for Always availability
# 
# 3. in Manhattan there are more records for Entire home/apt
