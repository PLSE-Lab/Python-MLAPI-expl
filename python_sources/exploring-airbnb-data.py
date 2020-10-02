#!/usr/bin/env python
# coding: utf-8

# # Airbnb Open Data Analysis

# According to [Wikipedia](https://en.wikipedia.org/wiki/Airbnb) , Airbnb, Inc. is an online marketplace for arranging or offering lodging, primarily homestays, or tourism experiences. The company does not own any of the real estate listings, nor does it host events; it acts as a broker, receiving commissions from each booking.
# 
# Airbnb's name comes from AirBreadaNdBreakfast after its founders put bed matresses in their living room turning their room into bed and breakfast lodging inorder to offset higher house rent cost at San Franscisco.
# 
# What started our as a small bed mattresses in a appartment floor now has listings of nearly 6 million worldwide.Atleast 2 million people stay worldwide in Airbnb's properties.
# 
# The dataset includes the Airbnb listing in New York city and includes information like properties found in the neighbourhood,price and the availability to name a few.
# 
# Lets begin our analysis and uncover some interesting insights.

# # Loading required libraries

# In[ ]:


## Loading the required libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# # Loading the data

# In[ ]:


kaggle=1

if kaggle==1:
    data=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
else:
    data=pd.read_csv("../data/AB_NYC_2019.csv")


# In[ ]:


## Examining the first 5 rows to understand the type of data:
data.head()


# In[ ]:


data.shape


# The dataset has 48k rows and 16 columns.

# In[ ]:


data.columns


# In[ ]:


## Check if any of the columns has null value:
data.isna().sum()


# While there are 16 rows with missing name , there are 21 rows where host name is missing.There are 10k rows here review is missing.My guess is that these are new listings and hence we dont have a review yet.

# # Neighbourhood Analysis

# ## Number of unique listings:

# In[ ]:


## Number of unique listings:
print(f'There are {data.id.nunique()} unique listings in the neighbourhood')


# Each row represents a unique listing.

# ## Location of the listings where maximum listings are present:

# In[ ]:


## Check the location of the listings where maximum listings:
data.neighbourhood_group.value_counts()


# There are 5 locations and Mahhattan has the highest listings followed closely by Brooklyn.Staten Island has lowest listing.Lets check the top 5 areas in each location which has maximum listing.

# ## Area where the number of listings are higher:

# In[ ]:


# Check the area where the number of listings are higher:
neighbourhood_group=data.neighbourhood_group.unique()
for n in neighbourhood_group:
    print(f'Top 5 neighbourhood in neighbourhood group {n}')
    print(data.loc[data['neighbourhood_group']==n]['neighbourhood'].value_counts()[:5])
    print()


# ## Room Type % listing:

# In[ ]:


## Room Type % listing:
(data['room_type'].value_counts()/data.shape[0])*100


# 51 % of the rooms listed have rented their entire property whereas 45 % have listed their private room.2% have shared room facility listed in Airbnb.

# # Price Analysis

# ## Distribution of price:

# In[ ]:


## Check the distribution of price:
plt.figure(figsize=(8,8))
sns.distplot(data['price'],bins=50,kde=True)
plt.title("Distribution of price(in USD)")


# The distibution of price is right skewed with the property prices less than 1000 USD.We also have properties with USD 10,000 .Lets check the summary statistics & plot a separate histogram to understand the distribution between price range 0 -1000 USD.

# In[ ]:


data['price'].describe()


# In[ ]:


plt.figure(figsize=(8,8))
sns.distplot(data[(data['price']>0) & (data['price']<1000)]['price'],bins=50,kde=True)
plt.title("Distribution of price(in USD)")


# * From the summary statistics,we understand that the average value of the property is USD 152 while the median value is USD * 106.The maximum property value is 10,000 USD .How many properties are there in this price? 
# * The histogram plot for price between 0-1000 USD is more interpetable and we get to know that the distribution is skewed towars right .We have more properties in the price range 0-200 USD.

# In[ ]:


data.loc[data.price==10000]


# There are three properties in the price of 10000 USD and they are availale in Queens ,Brooklyn and Manhattan.While the property at Queens and Brooklyn are not available for booking ,Manhattan property is available to book for 83 days in a year and the minimum number of nights required to stay is for a month.This property has not got any reviews so far.

# ## Average property value for each neighbourhood group

# In[ ]:


## Check the average property value for each neighbourhood group:
plt.figure(figsize=(8,8))
sns.boxplot(x=data['neighbourhood_group'],y=data['price'],palette=sns.color_palette('Set2'))
plt.title("Boxplot of price for each neighbourhood group",fontsize=15)
plt.xlabel("Neighbourhood group",fontsize=12)
plt.ylabel("Price",fontsize=12)
plt.show()


# Ploting boxplot for all the neighbourhood group, we see that there many outliers in each group and there is no clear understanding of price range.Lets groupby neighbourhood group and understand the median price.

# In[ ]:


data.groupby('neighbourhood_group')['price'].agg(['median','mean']).sort_values('median',ascending=False)


# * We see that Manhattan has higest median property price followed by Brooklyn.Queens and Staten Island have equal median price value.
# * The average price at Staten Island is third higest whereas its median property value is same as Queens.

# ## Minimum nights and price:

# In[ ]:


## Minimum nights and price:
plt.figure(figsize=(8,8))
sns.scatterplot(x='price',y='minimum_nights',data=data[(data.price>0) & (data.minimum_nights>0)])
plt.title("Price Vs Number of Nights",fontsize=15)
plt.xlabel("Price",fontsize=12)
plt.ylabel("Number of minimum nights",fontsize=12)
plt.show()


# There is no clear relationship between the price and the number of minumum nights.There are various options to choose when it comes to number of minimum nights and the price charged.There are properties where the minimum nights requirement is higher but the price is less.

# In[ ]:


data.groupby('neighbourhood_group').agg({'price':'median','minimum_nights':'median'}).sort_values("price",ascending=False)


# Manhattan and Brooklyn have properties with higher median price and the median number of minimum nights required to stay are 3 days while the rest of the neighbourhood group the minimum number of nights required are 2 nights.

# ## Room Type and Price

# In[ ]:


## Room Type and Price:
data.groupby('room_type')['price'].median()


# 51 % of the rooms listed are entire home/apt .The median price of such rooms are relatively higher compared to private rooms.

# ## Price and reviews

# In[ ]:


### Price and reviews:
plt.figure(figsize=(8,8))
sns.scatterplot(x='price',y='number_of_reviews',data=data[data.price<1000])
plt.title("Relation between price and number of reviews(For properties less than 1000 USD)",fontsize=15)
plt.xlabel("Price",fontsize=12)
plt.ylabel("Number of review",fontsize=12)
plt.show()


# From the above plot it is seen that the number of reviews are higher for those properties in the price range 0-400 USD.The maximum number of reviews is 200 as the price for the property increases.

# # Host Analysis

# In[ ]:


print(f'There are {data.host_id.nunique()} unique hosts in the dataset')


# ## Host listing count

# In[ ]:


## Host listing count:
plt.figure(figsize=(8,8))
sns.distplot(data.calculated_host_listings_count,bins=20,kde=False)
plt.title("Distribution of Number of properties listed by host",fontsize=15)
plt.xlabel("Number of properties by host",fontsize=12)
plt.show()


# In[ ]:


(data[data.calculated_host_listings_count<50]['host_id'].nunique()/data.shape[0])*100


# In[ ]:


data.calculated_host_listings_count.describe()


# The distribution is skewed towards right.On an average , each host has listed 7 properties.We see that the data has lot of outliers with a single host having listed 327 properties.Lets check the type of rooms they have listed.

# ## Host and Room Type

# In[ ]:


plt.figure(figsize=(8,8))
sns.boxplot(x='room_type',y='calculated_host_listings_count',data=data)
plt.title("Boxplot of properties listed by each host with room type",fontsize=15)
plt.xlabel("Room Type",fontsize=10)
plt.ylabel("Number of properties listed by each host",fontsize=10)
plt.show()


# * As mentioned earlier,the variable is dominated by outliers.Rooms of type private and entire home has more outliers than Shared rooms.We understood from the previous analysis that there are only 2% of shared room listings.
# * While this analysis provides an overall scenario of the property type,lets check the number of host who have all three property types listed in Airbnb.

# ## Host with all three room type

# In[ ]:


variety=data[data.calculated_host_listings_count>1].groupby('host_id')['room_type'].nunique().reset_index().sort_values('room_type',ascending=False)
print(f'Number of hosts with all three room types listed in Airbnb {len(variety[variety.room_type==3].host_id)}')


# * There are 23 hosts with all three room types listed.Lets check whether they have listed in same or in different neighbourhood.

# In[ ]:


variety_data=data[data.host_id.isin(variety[variety.room_type==3].host_id)]
variety_data.groupby('host_id')['neighbourhood_group'].nunique().sort_values(ascending=False)


# In[ ]:


set(variety_data[variety_data.host_id==213781715]['neighbourhood_group'])


# Of the 23 hosts,only one host has listed the properties in 3 neighbourhood groups - Brooklyn,Manhattan and Queens whereas the rest of them have listed their property in the same neighbourhood.

# ## Maximum Listing by host

# From our earlier analysis , we understood that the maximum number of listing by a single host is 327.Lets check the details for this host.

# In[ ]:


max_host=data[data.calculated_host_listings_count==327]


# In[ ]:


print(f'Name of host:{list(max_host.host_name.unique())}')
print(f'Neighborhood groups listed:{list(max_host.neighbourhood_group.unique())}')
print(f'Neighbourhoods listed:{list(max_host.neighbourhood.unique())}')
print(f'Room type listed:{list(max_host.room_type.unique())}')
print(f'Maximum price listed:{max(max_host.price)} USD Located in neighbourhood {max_host[max_host.price==max(max_host.price)].neighbourhood.unique()}')
print(f'Minimum price listed:{min(max_host.price)} USD Located in neighbourhood {max_host[max_host.price==min(max_host.price)].neighbourhood.unique()}')


# The property is hosted at Manhattan and its neighbourhoods.The rooms are either listed as entire home or private room.The maximum property price is listed at Theater District and Financial Disrict whereas the minimum price is listed in Financial district.

# For the upcoming analysis,we consider only the host who have listed more than 1 properties in Airbnb.

# In[ ]:


## Top 5 Host with maximum median price and median nights for those holding more than 1 property:
data[data.calculated_host_listings_count>1].groupby('host_id').agg({'price':'median','minimum_nights':'median'}).sort_values('price',ascending=False)[:5]


# The above analysis shows the median price for the property hosted by a person with more than 1 listings.The median price for such host is 3780 USD and the minimum night requirement is 1.For host 16105313,though the price is not too high ,the minimum nights requirement is very high.

# # Reviews

# ## Review Count

# In[ ]:


## Create date columns from last review date:
data['last_review']=pd.to_datetime(data['last_review'])
data['year']=data['last_review'].dt.year
data['month']=data['last_review'].dt.month
data['day']=data['last_review'].dt.day
data['day_name']=data['last_review'].dt.day_name()


# In[ ]:


data.head()


# ## Owner and Review

# Since we only have the last review count , we wont be able to do any interesting analytics on it . If we would have had the time stamp of review ,we would have known about the pattern in which certain rooms are getting booked , see whether there were any seasonal pattern in room booking , whether any property is busy only on weekdays /weekends etc . But here we only have the last review date.Hence we restrict ourselves to check whether the hosts with all the properties have received equal reviews etc.

# In[ ]:


multi_host=data[data.calculated_host_listings_count>1]


# In[ ]:


multi_host['review_min']=multi_host.groupby('host_id')['number_of_reviews'].min()
multi_host['review_max']=multi_host.groupby('host_id')['number_of_reviews'].max()
multi_host['review_median']=multi_host.groupby('host_id')['number_of_reviews'].median()
multi_host['review_diff']=multi_host.groupby('host_id')['number_of_reviews'].max()-multi_host.groupby('host_id')['number_of_reviews'].min()


# In[ ]:


multi_host.describe()


# In[ ]:


def _gen_histogram(df,column):
    plt.figure(figsize=(8,8))
    sns.distplot(df[column].dropna(),bins=10)
    plt.xlabel(r"{}".format(column))
    plt.ylabel("Density")
    plt.title(r"Distribution of {}".format(column))


# In[ ]:


columns=['review_min','review_max','review_median','review_diff']

for c in columns:
    _gen_histogram(multi_host,c)


# Couple of observations:
# 
# * From all the plots,it is understood that from the host having more than 1 property,it is observed that most of them did no receive any review.
#     
# * The minimum review varies in the range 0-100 with most of the host receiving anywhere between 0-10.There are spikes in between which indicates that some properties have been extremely popular among guests.
#     
# * The maximum review also lies in the similar range 0-20.There is a split in 90-100,200-210 and ~250-260 range.
#     
# * The median review count indicates that the distribution is sligthly right skewed with the mean somewhere between 40-50.This also suggests that most of the hosts receive reviews across their properties anywhere between 0-50 range.
#     

# **Work in progress.If you like my kernel,pls upvote and comment.**
