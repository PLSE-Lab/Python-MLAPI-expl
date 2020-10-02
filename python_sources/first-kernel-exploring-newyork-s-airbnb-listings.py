#!/usr/bin/env python
# coding: utf-8

# # **About the data set**
# This public dataset is part of Airbnb and it includes information to find out more about hosts, geographical availability, neighborhood and reviews. Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.
# 
# **The inspiration for my analysis are the questions below**
# * Which are the famous neighborhoods and neighborhood groups among the listings?
# * What is the price range of the listings?
# * What are the mst common words used in the description?
# * Which room types are usually busy ? Is the trend different in different neighborhoods?
# * How are the reviews distributed?
# 
# 
# 

# # Importing the required libraries

# In[ ]:


#importing the required libraries
import pandas as pd
import numpy as np
import datetime

#viz Libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight') # setting style for the plots

#warnings
import warnings
warnings.filterwarnings("ignore")

#word cloud
from wordcloud import WordCloud, ImageColorGenerator


# # Reading the data - Bringing it in

# In[ ]:


df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',index_col='id')


# # Cleaning the data - Lets make it better

# In[ ]:


df.shape # shape of data


# In[ ]:


df.head() # first 5 rows of data - 5 by default


# In[ ]:


df.info()


# In[ ]:


df.describe()


# **Zero price** -- Free Rooms?<br>
# **Zero Availability** -- Looks like those listings are not available any more and the data is not updated.

# # Checking for missing data

# In[ ]:


df.isna().sum()


# Missing data is one of the most common problem in data cleaning/exploring. There is no standard method of replacing the missing values and the method of imputating them highly depends on the type of problem and the data we are dealing with.
# 
# Sometimes, counting the missing data using "*isna()*" function can be misleading because missing data does not always mean NaN values. For example, lets consider the variables 'price' and 'availability_365'. The values for these variables cannot be zeroes practically. But we have such values in this  data.

# In[ ]:


#dropping the rows with price = 0 or availability_365 = 0
indexNames = df[ (df['price'] == 0) | (df['availability_365'] == 0) ].index
df.drop(indexNames , inplace=True)


# # Dropping the duplicate data if any

# In[ ]:


#dropping the duplicates
df = df.drop_duplicates()


# In[ ]:


#dropping the columns which may not add extra value to the analysis
df = df.drop(['host_id','latitude','longitude'],axis=1)


# # Checking for inconsistent data types

# In[ ]:


df.dtypes


# In[ ]:


#converting "last_review" to "date_time" data type
df['last_review'] = pd.to_datetime(df['last_review'])

#converting categorical variables into "categorical" data type
cat_var = ['neighbourhood_group','neighbourhood','room_type']
df[cat_var] = df[cat_var].astype('category')

df.info()


# Notice that converting the categorical variables into "category" data type significantly reduced the memory from 6.0+ MB to 2.5+ MB

# # Exploring the data - Lets know it better

# **** Which are the most popular neighborhood groups among the listings?****

# In[ ]:


#popular neighborhood groups
ax = sns.countplot(x="neighbourhood_group", data=df)
#df['neighbourhood_group'].value_counts().plot(kind="bar")
plt.title('Popular neighborhood groups')
plt.xlabel('Neighborhood Group')
plt.ylabel('Count')
plt.show()


# **Manhattan** and **Brooklyn** clearly win the race for popular neighborhoods. Manhattan is the **smallest** of all boroughs in size but the most densly populated and Brooklyn is the city's most populated borough which collectively makes them the most popular neighborhood groups. 

# ** Which are the most popular neighborhoods among the listings?**

# In[ ]:


ax = sns.countplot(y="neighbourhood", hue="neighbourhood_group", data=df,
              order=df['neighbourhood'].value_counts().iloc[:5].index)
plt.title('Popular Neighborhoods')
plt.ylabel('Neighborhood')
plt.xlabel('Count')
plt.show()


# ** Which are the most occupied room types among the listings?**

# In[ ]:


ax = sns.countplot(x="room_type", data=df)
plt.title('Room Type distribution')
plt.xlabel('Room Type')
plt.ylabel('Frequency')
plt.show()


# **Room type distribution in the neighborhood groups**

# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.countplot(x="room_type", data=df,hue="neighbourhood_group")


# **Exploring the "price" **

# In[ ]:


df['price'].describe()


# In[ ]:


sns.boxplot(x='neighbourhood_group',y='price',data = df)
plt.title("Price distribution among the neighborhood groups")
plt.show()


# From the descriptive stats above, we can see that 75% of the prices are below "189".
# We cannot declare that the values above $189 are outliers because pricing depends on the amenities offered and the locations they are present.
# 
# As majority of the data is below the price range of $189, a new data set is created and its price distribution is observed below.

# In[ ]:


df3 = df[df['price'] <= 200]
df3.price.plot(kind='hist')
plt.xlabel("Price")
plt.title("Price distribution for chosen listings(Price <= 200)")
plt.show()


# Most of the listings are between the price range of 25 to 110. 

# **Price range distribution among the neighborhoods**

# In[ ]:


sns.boxplot(x='neighbourhood_group',y='price',data = df3)
plt.show()


# Clearly, Manhattan and Brooklyn are the costliest. Manhattan listings have an average price range of 120 dollars, while Brooklyn has 90 dollars. All the other three neighborhoods have almost similar average price range of 70-75 dollars.

# **Price range distribution among the room types**

# In[ ]:


sns.boxplot(x='room_type',y='price',data = df3)
plt.show()


# **Distribution of Reviews**

# In[ ]:


df['number_of_reviews'].plot(kind='hist')
plt.xlabel("Price")
plt.show()


# **Distribution of reviews among available room types and neighborhood groups**

# In[ ]:


plt.figure(figsize=(9, 6))
plt.subplot(1,2,1)
df3.groupby(['room_type']).count()['number_of_reviews'].plot(kind='bar',alpha = 0.6,color = 'orange')
plt.title('Room Type Vs Number of Reviews',fontsize=15)

plt.subplot(1,2,2)
df3.groupby(['neighbourhood_group']).count()['number_of_reviews'].plot(kind='bar',color='green',alpha=0.5)
plt.title('Neighborhood Vs Number of Reviews',fontsize=15)
plt.tight_layout()
plt.show()


# **Distribution of minimum nights among available room types and neighborhood groups**

# In[ ]:


plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
df3.groupby(['room_type']).count()['minimum_nights'].plot(kind='bar',alpha = 0.6,color = 'orange')
plt.title('Room Type Vs Minimum Nights',fontsize=15)

plt.subplot(1,2,2)
df3.groupby(['neighbourhood_group']).count()['minimum_nights'].plot(kind='bar',alpha = 0.6,color = 'green')
plt.title('Neighborhood Group Vs Minimum Nights',fontsize=15)
plt.tight_layout()
plt.show()


# **Commonly used words in the description**

# In[ ]:


text = " ".join(str(each) for each in df.name)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=100, background_color="white").generate(text)
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# **Correlation among numeric variables**

# In[ ]:


columns =['price','minimum_nights','number_of_reviews','reviews_per_month','availability_365']
#sns.heatmap(df[columns])
corr = df[columns].corr()
corr.style.background_gradient(cmap='coolwarm')


# # **Insights developed :**
# * **Manhattan** is the costlies and popular neighborhood groups among the listings followed by **Brooklyn**.
# * The top 5 popular neighborhood groups are from Manhattan and Brooklyn.
# * Entire Homes are the highly occupied room types followed by Private and Shared rooms.
# * Manhattan has high number of Entire Home and Shared room listings. Brooklyn has high private room bookings.
# * 75% of the listing's prices are below **189 dollars**.
# * **Manhattan** listings have an average price range of **120 dollars**, while **Brooklyn** has **90 dollars**. All the other three neighborhoods have almost similar average price range of **70-75 dollars**.
# * Most number of reviews are collected from **Private** room type and the neighborhood of **Brooklyn**.
# * There is no strong correlation among the numeric variables.
# 
