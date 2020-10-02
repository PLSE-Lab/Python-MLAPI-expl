#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
#For geoplots
import geopandas as gpd
from shapely.geometry import Point, Polygon
import descartes


# In[ ]:


#Reading the dataset.
data = pd.read_csv(r'/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# Let us explore the top 5 observation in the dataset to get a feel about what the data looks like and what are the columns we are dealing with.

# In[ ]:


#Exploring the top 5 observations.
data.head()


# We can see that there are some NaN values in the last_review and reviews_per_month columns. We will look into the null values in deatils later.

# In[ ]:


#describing the datasets.
data.describe()


# We can see that the price column has some irregularity as the minimum price of the apartment is 0 i.e Free stay?. We will have look deeper into this later.

# In[ ]:


#null or missing values in the dataset.
data.isnull().sum()


# Missing values are present in the name, host_name, last_reviews and reviews_per_month columns.<br>
# In the above exploration part we can see that if the number_of_reviews is 0 then it does not make sense to have last_review and reviews_per_month and are marked as NaN.<br>
# Hence the missing values in the data is following a pattern and will be treated accordingly.
# 
# Let us check if the assumption made above holds true.

# In[ ]:


#checking the assumption -> 0 reviews will have missing values in last_review and reviews_per_month columns.
assumption_test = data.loc[(data.last_review.isnull()) & (data.reviews_per_month.isnull()), ['number_of_reviews',  'last_reviews', 'reviews_per_month']]
assumption_test.head()


# As we can see our assumption holds true. Let us check the shape of the created dataframe and the number of null values.

# In[ ]:


assumption_test.shape


# The exact amount of null values present in both the columns. It proves that the assumption madde was clear.<br>
# We will substitute 0 for the missing values present in reviews_per_month column. <br>
# 
# As for the last_review column we know that it is a datetime object of the pandas and substituting 0 won't make sense here.<br>
# We will have to leave the nul values of last_reviews as it is for now.

# In[ ]:


#filling the missing values in reviews_per_month with 0.
data.reviews_per_month.fillna(0, inplace=True)


# In[ ]:


#Checking if the changes made are reflected.
data.isnull().sum()


# That been done, we will also leave the null values present in the host name and name columns as they are not required for our EDA as of now.<br>

# # Exploratory Data Analysis

# Lets begin with our exploratory data analysis.

# In[ ]:


#checking the to 5 neighborhood where the properties are listed most.
top_5_neighborhoods = data.neighbourhood.value_counts().head(5)
print(top_5_neighborhoods)

#plotting 
plt.figure(figsize=(8,5))
top_5_neighborhoods.plot.bar()
plt.xlabel('Neighborhoods')
plt.ylabel('Listed Property Count')
plt.title('Count of properties in a neighborhood')
plt.show() #optional


# The top 5 neighborhoods whcih have the highest number of properties listed are shown above.<br>
# We can see that Williamsburg has the highest number of properties listed (3920) followed by Bedford-Stuyvesant (3714).
# 
# As Williamsburg has the highest number of properties listed then the Brooklyn neighborhood group must also have the highest number of properties listed as williamburg comes under brooklyn neighborhood group.
# 
# Lets check if this is correct or not.

# In[ ]:


#checking the to 5 neighborhood groups where the properties are listed most.
top_5_neighborhood_group = data.neighbourhood_group.value_counts()
print(top_5_neighborhood_group)

#plotting 
plt.figure(figsize=(8,5))
top_5_neighborhood_group.plot.bar()
plt.xlabel('Neighborhood Groups')
plt.ylabel('Listed Property Count')
plt.title('Count of properties in a neighborhood group')
plt.show() #optional


# Our assumption was wrong. As it turned out that Manhattan has the higghest number of properties listed although Williamsburg town in Brooklyn had the highest number of properties amongh the neighbor towns.
# 
# This infers that there are many other towns in Manhattan that have properties listed and that is why Manhattan neighborhood as a whole has the highest number of properties.
# 
# We can also see that other neighborhood groups such as Queens, Bronx and Staten Island contribute less compared to Manhattan and Brooklyn.

# In[ ]:


#number of rooms_type provided by the hosts
print(data.room_type.value_counts())
sns.countplot(data.room_type)


# There are 3 room type provided by the host. Most of the rooms provided are private rooms and Entire home or apartments type.
# 
# Share rooms are listed very few, as it make sense that people travelling with family will prefer the top 2 room types rather than sharing.

# In[ ]:


#Lets check the distribution of the price of the properties.
sns.distplot(data.price, bins=50)


# The distribution of price is heavily left skewed. Meaning the most of the properties price are between 0 - 2000 and some minority of the properties are having prices grator than that making the data to be skewed.
# 
# Also we saw that some properties have a price as 0 and that could not be possible here as no one will be giving their property on rent for free! That will be absurd.

# In[ ]:


#Looking into the properties having 0 Price
free_properties = data.loc[data.price <= 0]
print('Shape of the data:', free_properties.shape)
free_properties.head()


# 11 properties have are having 0 price. Assuming this to be a mistake or error from the Airbnb side, we will have to impute the prices according. 
# 
# One way to impute will be by taking the mean, but as we saw earlier the price distribution is highly skewed and hence that will affect the mean of the price.
# 
# Presence of outliers or extreme values in the dataset effect the mean of the data and is not a good option to impute. Other method is to impute the data with mediian as median is less affected by outliers/extreme values.
# 
# The other effective way will be to see the affect of price or the relation of price on various other factors in the dataset and come up with a formula or a model that will do the imputation for us.<br>
# We will look into this later.

# In[ ]:


#minimum number of nights allowed by the host.
sns.distplot(data.minimum_nights, bins=10)


# The minimum nights goes from 1 to 1200+. Only 1 host provide minimum_nights to be 1200+. We come to know from the distribution that the data is skewed.

# Now let's check which top 5 properties have recieved the highest number of reviewes.

# In[ ]:


#properties recieving highest reviews.
highest_reviews = data.sort_values(by='number_of_reviews', ascending=False)
highest_reviews.head()


# The above table shows the top 5 properties which have recieved the highest number of reviews. Out of the 5, three properties are from the Manhattan neighborhood group in Harlem. 
# 
# The top property which has recieved the most reviews is from Queens in the neighborhood of Jamica. The property having 5th highest review is also from Queens.
# 
# Let us look at these properties and try to come up with some hypothesis on why these properties have the highest reviews.
# 1. These properties are the most popular properties among the others and that is why they may be getting more bookings and hence more reviews.
# 2. They come from the same host i.e Dona and JJ. May be they are a good and popular hosts that is why are recieving good amount of bookings and reviews.
# 3. The Dona host has her room near the JFK i.e nearer to the international airport and that's why the high amount of bookings.
# 4. All of them share the similar room type, i.e Private rooms. So we can assume that the private rooms are more popular than any other rooms.
# 5. The price is also almost similar of all the properties approx 50.
# 6. They all offer minimum 1 night stay which most of the people prefer as it is very flexible.
# 7. The availability of the rooms is also high with the top 4 having an availability rate of approx 300 days and +.

# In[ ]:


#host having highest amount of properties listed.
highest_props_host = data.groupby(['host_id', 'host_name'])['host_id'].count().sort_values(ascending=False)[:10]
highest_props_host.plot.bar(figsize=(10,5))
plt.xlabel('Hosts')
plt.ylabel('Properties Listed')
plt.title('Hosts having highest amount of properties listed');


# We can see that Sonder(NYC) has the highest number of properties that are listed but his property was not in the top 5 highest reviews table we saw earlier.
# 
# This means that the number of properties listed on the Airbnb does not mean that the number of customers you will have will be more.

# # Bivariate Analysis

# In[ ]:


#neighborhood group based on the latitude and longitude
plt.figure(figsize=(10,8))
sns.scatterplot(data.latitude,data.longitude, hue='neighbourhood_group', data=data)


# The above resemble the map of NYC and shows the various neighbourhoods and the properties listed in each neighbourhood.

# In[ ]:


# #Properties in the neighbourhood with most reviews. 
# plt.figure(figsize=(10,8))
# sns.scatterplot('latitude', 'longitude', hue='neighbourhood_group', data=highest_reviews.head(10))


# # **Conclusion:**

# So far we have done some basic exploring of the dataset and have gain few insights from it such as:
# 1. Came to know that the price column had some irregularities such as the minimum price was 0$ which is not possible.
# 2. Missing values in the last_review column and reviews_per_month column were following a pattern. If number_of_reviews was 0 then these two columns had null values.
# 3. From the univariate analysis we came to know the top 5 neighbourhood whcih had the highest number of properties listed.
# 4. We also came to know the top neighbourhood groups which had the highest number of properties listed.
# 5. We came to know the top 5 properties which had the highest review and factors contibuting to their reviews.
# 6. We came to know that properties closer to the airport had a good number of reviews.
# 7. We saw the host which had the highest number of properties listed on Airbnb.
# 8. We saw the distribution of the properties based on the neighbourhood_group on a scatter plot.
