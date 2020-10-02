#!/usr/bin/env python
# coding: utf-8

# # Data visualization for Linear Regression
# 
# In this kernel we are focusing on data preprocessing and data visualisation of New York City Airbnb Open Data
# Airbnb listings and metrics in NYC, NY, USA (2019)for linear regression.
# 
# ### Data
# 
# Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019. This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.<br>
# 
# This data contains 16 columns, 47905 unique values(samples). Imported all necessary files and libraries, We removed unnecessary data from the datset like last review, reviews per month and host name as they donot support the data required. We filled the null values with zero constant and did the visualization using seaborn, pyplot, matplotlib.<br>
# 
# #### Variables
# id: listing ID<br>
# name: name of the listing<br>
# host_id: host ID<br>
# host_name: name of the host<br>
# neighbourhood_group: location<br>
# neighbourhood: area<br>
# latitude: latitude coordinateslatitude: latitude coordinates<br>
# longitude: longitude coordinates<br>
# room_type: listing space type<br>
# price: price in dollars<br>
# minimum_nights: amount of nights minimum<br>
# number_of_reviews: number of reviews<br>
# last_review: latest review<br>
# reviews_per_month: number of reviews per month<br>
# calculated_host_listings_count: amount of listing per host<br>
# availability_365: number of days when listing is available for booking<br>
# 
# We will perform data visualization for linear Regression on this dataset.
# 
# 
# 

# In[ ]:


#import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#read the file 'NYC_2019.csv' from the file
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


#obtain information about the dataframe
df.info()


# In[ ]:


df


# In[ ]:


#find no of columns and no of rows
df.shape


# In[ ]:


#obtaining the description of the dataframe
df.describe()


# In[ ]:


#finding out if there are any null or empty values
df.isnull().sum()


# We leave the null values as is, as it doesn't effect the visualization.

# In[ ]:


#knowing how many neighbourhood groups are there and count of them
df.neighbourhood_group.value_counts()


# In[ ]:


fig,ax=plt.subplots(figsize=(10,8))
sub_df = df[df.price < 1000]
plot_2=sns.violinplot(data=sub_df, x='neighbourhood_group', y='price')
plot_2.set_title('Density and distribution of prices for each neighberhood_group')


# In[ ]:


sub_df = df[df.price < 1000]
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'room_type', y = 'price',  data = sub_df)
#regression


# In[ ]:


top10_freq_neighbourhood=df.neighbourhood.value_counts().head(10)
print(top10_freq_neighbourhood)


# In[ ]:


top10_freq_neighbourhood_data=df[df['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',
'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]
top10_freq_neighbourhood_data


# In[ ]:


t=sns.catplot(x="neighbourhood", y="price", col="room_type", data=top10_freq_neighbourhood_data)
t.set_xticklabels(rotation=45)


# In[ ]:


df.fillna('0',inplace=True)
df


# In[ ]:


fig,ax=plt.subplots(figsize=(10,8))
sns.distplot(np.log1p(df['number_of_reviews']))


# In[ ]:


fig,ax=plt.subplots(figsize=(10,8))
sns.countplot(df['neighbourhood_group'])


# Dividing the data into costly,medium, reasonable, cheap, very cheap.

# In[ ]:


df['Cat'] = df['price'].apply(lambda x: 'costly' if x > 3000
                                                    else ('medium' if x >= 1000 and x < 3000
                                                    else ('reasonable' if x >= 500 and x < 1000
                                                     else ('cheap' if x >= 100 and x <500
                                                          else'very cheap'))))


# In[ ]:


plt.figure(figsize=(10,8))

sns.scatterplot(df.latitude,df.longitude, hue='Cat', data=df)


# ### Observation
# 
# We can obeserve that many people skipped giving reviews which affected the data. We can observe that maximum room prices are around 1000 usd only. Also there are lot of rooms in Manhattan which is costlier than any other place. Williamsburgh in Manhattan has many rooms there. The price of single room is very less compared to a whole apartment. The second highest place containing the rooms is brooklyn. Shared rooms are cheaper but are less available in Manhattan and Brooklyn. Highest no of reviews were given in manhattan and it looks like people like manhattan than any other place. The availability of single rooms is higher and is ever ready. But the entire suits are available only in particular days. The latitude and longitude gives the distribution of price over NYC.
# 

# In[ ]:




