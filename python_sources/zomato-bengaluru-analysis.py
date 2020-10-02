#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# A quick check on  what Zomato is for all those who are cluless here. 
# Zomato is an Indian restaurant search and discovery service founded in 2008 by Deepinder Goyal and Pankaj Chaddah. It currently operates in 24 countries. It provides information and reviews of restaurants, including images of menus where the restaurant does not have its own website and also online delivery. 
# 
# 

# ## Data Explaination:
# This data would be fascianting for any Data Science practitioner. Well even more excitment for all the foodie Data Scientist. This dataset is rich with information on restaurants who are linked with Zomato in bengaluru, and since Zomato is a big name in India it would have all the corners and streets covered with ratings from dozens of people, reviews and their favourite dishes. This gives insights on customer preferences and change in taste with change in demographics.

# In[ ]:


import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.colors as mcolors
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_main = pd.read_csv("../input/zomato.csv")
df_main.describe()


# **Columns description**
# 
# - **url**
# contains the url of the restaurant in the zomato website
# 
# - **address**
# contains the address of the restaurant in Bengaluru
# 
# - **name**
# contains the name of the restaurant
# 
# - **online_order**
# whether online ordering is available in the restaurant or not
# 
# - **book_table**
# table book option available or not
# 
# - **rate**
# contains the overall rating of the restaurant out of 5
# 
# - **votes**
# contains total number of rating for the restaurant as of the above mentioned date
# 
# - **phone**
# contains the phone number of the restaurant
# 
# - **location**
# contains the neighborhood in which the restaurant is located
# 
# - **rest_type**
# restaurant type
# 
# - **dish_liked**
# dishes people liked in the restaurant
# 
# - **cuisines**
# food styles, separated by comma
# 
# - **approx_cost(for two people)**
# contains the approximate cost for meal for two people
# 
# - **reviews_list**
# list of tuples containing reviews for the restaurant, each tuple 
# 
# - **menu_item**
# contains list of menus available in the restaurant
# 
# - **listed_in(type)**
# type of meal
# 
# - **listed_in(city)**
# contains the neighborhood in which the restaurant is listed
# 

# In[ ]:


df_main.info()


# This dataset does not have any empty values which is a green flag for us to begin digging in.

# In[ ]:


df_main.head(1)


# ## Starting with Location that has most restaurants:

# In[ ]:


df_loc = df_main['location'].value_counts()[:20]
plt.figure(figsize=(20,10))
sns.barplot(x=df_loc,y=df_loc.index)
plt.title('Top 20 locations with highest number of Restaurants.')
plt.xlabel('Count')
plt.ylabel('Restaurant Name')


# Looks like BTM is winning the race with highest number of restaurnats.
# 
# **The major reason for BTM being a great attraction for restaurants could be:**
# - Its Layout's proximity to Outer Ring Road, Bangalore, Koramangala, HSR Layout, Bannerghatta Road, J P Nagar and Jayanagar and that makes it one of the most popular residential and commercial places in Bangalore- says Wikipedia too.
# - It is one of the high growth neighbourhoods in terms of property prices, showing an annual growth rate of close to 60% in early 2010.
# - It is also a great touroist attraction place with lakes and national parks.
# 
# *Fun Fact: BTM Layout got its abbreviated name as it is situated between Bommanhalli, Tavarekere and Madiwala.*
# 

# **Digging deeper into BTM and its Restaurants:**

# In[ ]:


df_BTM =df_main.loc[df_main['location']=='BTM']
df_BTM_REST= df_BTM['rest_type'].value_counts()

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)

sns.barplot(x=df_BTM_REST, y= df_BTM_REST.index,ax=ax1)
plt.title('Count of restaurant types in BTM')
plt.xlabel('Count')
plt.ylabel('Restaurant Name')


# In[ ]:



plt.figure(figsize=(20,15))
df_BTM_REST1 = df_BTM_REST[:10]
labels = df_BTM_REST1.index
explode = (0.1, 0,0,0,0,0,0,0,0,0)  
plt.pie(df_BTM_REST1.values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('top 10 restaurant types in BTM')

print("Quick bites are {} % of all the Restaurant types".format((df_BTM_REST.values[0]/df_BTM_REST.sum())*100))


# Well! I am not surprised with BTM having such high number of Quick bite places with it being a perfect weekend picnic spot and a great tourist attraction. Quick Bites covers 53.7% of all the Restaurant types in BTM which is quite huge.

# ## Lets dive deeper into rating in BTM Location Restaurants:

# In[ ]:


df_RATE_BTM=df_BTM[['rate','rest_type','online_order','votes','book_table','approx_cost(for two people)','listed_in(type)','listed_in(city)','cuisines','reviews_list','listed_in(type)']].dropna()
df_RATE_BTM['rate']=df_RATE_BTM['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
df_RATE_BTM['approx_cost(for two people)']=df_RATE_BTM['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))

df_rating = df_BTM['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()


# In[ ]:



f, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
sns.despine(left=True)


sns.distplot(df_rating,bins= 20,ax = axes[0]).set_title('Rating distribution in BTM Region')

plt.xlabel('Rating')

df_grp= df_RATE_BTM.groupby(by= 'rest_type').agg('mean').sort_values(by='votes', ascending=False)


sns.distplot(df_grp['rate'],bins= 20,ax = axes[1]).set_title('Average Rating distribution in BTM Region')


# **CAUTION!!**
# This is a great example to not fall victim of distribution of certain value over population vs distribution of the same value when taken average.
# There is a drastical change and missing out on this could kill your visual story line completely. 
# 
# **CAUTION!!**
# It might appear from the average rating distribution by Restaurant Type that some of these restaurants have gotten rating as low as 0, but not to forget we set the values with no rating as 0 and that could be the reason we have a peak at 0. While the graph without mean was not altered with null values and it shows a uniform distribution.
# 
# ** Lesson Learned: **
# Null != 0 (necessarily)

# In[ ]:



df_grp.reset_index(inplace=True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x=df_grp['votes'], y= df_grp['rest_type']).set_title('Average Votes distribution in BTM Region')


df_grp1= df_RATE_BTM.groupby(by='rest_type').agg('mean').sort_values(by='rate', ascending=False)
plt.figure(figsize=(20,10))
sns.barplot(y= df_grp1.index, x=df_grp1['rate']).set_title('Average Rating distributed in BTM Region')
plt.xlim(2.5,5)



# *Even with massive number of Quick Bite places in BTM region the votes and rating is incredibly high with Casual Dinning and Bars.
# I believe one of the major reason here would be the experience and service dinning and bar provides that quick bites do not. It is not only the food that attracts customers, it sure is important but there is much more that goes into the restaurant business, the uniquiness, the experience and how it stands out. The more you stand out the more you have power over your customers spending in your restaurant. Like having a place with good ambiance and feel to a restaurant can give them a boost next to those small food joints and have them more customers aswell.* 
# 

# In[ ]:


df_grp2 =  df_RATE_BTM.groupby(by= 'rest_type').agg('mean').sort_values(by='approx_cost(for two people)', ascending=False)

plt.figure(figsize=(20,10))
sns.barplot(y= df_grp2.index, x=df_grp2['approx_cost(for two people)']).set_title('Average Cost for 2 distributed in BTM Region')


# *Like I mentioned in the above finding the more you stand out and the better experience you provide your customer the more you have control over their spending. High end Dinning places and Bars can over charge their customers for the simplesest of dishes for the experience and services the provide, it is one of the mant Restaurant Business techniques.* 
# **This also says a lot about Bengaluru population and demograghics:** 
# - The average salary in Bangalore, Karnataka is Rs 636,959. The most popular occupations in Bangalore are Software Developer, Software Engineer, and Senior Software Engineer which pay between Rs 508,331 and Rs 859,301 per year (Which is quite great for a country like India). This proves that Bengaluru has a huge portion of its population who can afford these expensive Dinning places and Bars.
# - It also tells us about the literate young population which prefers new experience and are open to explore. Male literacy rate in Bangalore district of Karnataka is 91.01 percent. Female literacy rate in Bangalore district of Karnataka is 84.01 percent which is amazing.
# - This massive difference can also be due to the social media effect where people want showoff the good places they go and share their experience.

# 
# 

# ## Lets dive deeper into Casual Dinning:

# In[ ]:


df_Count_CasualDinning =df_main.loc[df_main['rest_type'] =='Casual Dining, Bar'].groupby(by='location').agg('count').sort_values(by='rest_type')


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x=df_Count_CasualDinning['rest_type'], y= df_Count_CasualDinning.index).set_title("Count of Casual Dining, Bar in Bengaluru")
print('There are about {} number of Casual Dining, Bar in Bengaluru.'.format(df_Count_CasualDinning['rest_type'].sum()))


# In[ ]:


df_count_casual= df_main.loc[df_main['rest_type'] =='Casual Dining, Microbrewery'].groupby(by='location').agg('count').sort_values(by='rest_type')


# In[ ]:


sns.barplot(x=df_count_casual['name'],y=df_count_casual.index).set_title('Number of Casual Dining, Microbrewery in Bengaluru ')
plt.xlabel('Count')


# Lavelle Road is the leading location when it comes to high end dinning restaurants and Bars. 
# 
# *This can mean that Lavelle Road has an affluent population or it is surrounded by such location.*

# ## What about the trending Online Orders and Book table before you reach the Restaurant?

# In[ ]:


df_count_online1=df_main.groupby(by='online_order').agg('count')
df_count_online1.reset_index(inplace=True)

ax1 = plt.subplot2grid((2,2),(0,0))
plt.pie(df_count_online1['url'], labels=df_count_online1['online_order'], autopct='%1.1f%%', shadow=True)

plt.title('online orders?')

df_count_online=df_main.groupby(by='book_table').agg('count')
df_count_online.reset_index(inplace=True)

ax1 = plt.subplot2grid((2,2), (0, 1))
plt.pie(df_count_online['url'], labels=df_count_online['book_table'], autopct='%1.1f%%', shadow=True)
plt.title('Book Table ?')


# - *Looks like major Restaurannts are switching to online order as convenience for the customer means everything to the business. While on the other hand not many of these restaurants have Table booking prior arrival which is quite strange.*
# 

# ** Lets look if convenience for customers plays any role in voting and rating :**

# In[ ]:


df_count_online1=df_RATE_BTM.groupby(by='online_order').agg('mean')
df_count_online1.reset_index(inplace=True)
fig, axarr = plt.subplots(1, 2, figsize=(12, 8))

sns.barplot(x=df_count_online1['online_order'], y=df_count_online1['rate'], ax= axarr[0]).set_title('Average rating for book table Restaurants')

sns.barplot(x=df_count_online1['online_order'], y=df_count_online1['votes'],ax= axarr[1]).set_title('Average votes for book table Restaurants')


# *Clearly!! the Restaurants that offer onlinr orders have much higher average rating and votes. Convenience and services does impact how a customer feels about the restaurant, and it impacts their review as well.* 

# In[ ]:


df_count_online1=df_RATE_BTM.groupby(by='book_table').agg('mean')
df_count_online1.reset_index(inplace=True)

fig, axarr = plt.subplots(1, 2, figsize=(12, 8))

sns.barplot(x=df_count_online1['book_table'], y=df_count_online1['rate'], ax= axarr[0]).set_title('Average rating for book table Restaurants')

sns.barplot(x=df_count_online1['book_table'], y=df_count_online1['votes'],ax= axarr[1]).set_title('Average votes for book table Restaurants')


# **INTRESTING :**
# Clearly Restaurants with booking table services are getting much higher Votes and Ratings but the count for Table Book Restaurants are low. 
# This is some Data Analysis Insight that Restaurants can use to boost up their ratings and votes.

# ## Last but the most important, Time to talk about the cuisines:

# In[ ]:


df_cus= df_RATE_BTM.groupby(by='cuisines').agg('mean').sort_values('rate', ascending = False)[:20]
df_cus.reset_index(inplace=True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x=df_cus['rate'],y=df_cus['cuisines']).set_title('Average Rating on Cuisines')
plt.xlim(3.5,5)


# In[ ]:


df_cus= df_RATE_BTM.groupby(by='cuisines').agg('mean').sort_values('votes', ascending = False)[:20]
df_cus.reset_index(inplace=True)

plt.figure(figsize=(20,10))
sns.barplot(x=df_cus['votes'],y=df_cus['cuisines']).set_title('Average Votes on Cuisines')


# *European, Mediterranean, North Indian and BBQ are the most rated and voted of all the cuisines which is strange to since Bengaluru is in South India.
# The only reason I can come up with this starnge behaviour is the diversity in Bengaluru. It is called the Sillicon valley of India for all the young working IT professional and being a hib for IT startups. This explains the strange behaviour.*

# ## More digging...

# ** Lets see the cost for 2 wtr to rating and votes**

# In[ ]:



sns.lmplot(x='rate',y='approx_cost(for two people)',data= df_cus)
plt.ylabel('Cost for 2')

sns.lmplot(x='votes',y='approx_cost(for two people)',data= df_cus)
plt.ylabel('Cost for 2')


# ## What!! wow!! 
# As the price for two keeps increasing the rating and votes for that Restaurant keeps increasing. Is it some strange human psychology chapter on NETFLIX?
# 

# In[ ]:


sns.lmplot(x='rate',y='approx_cost(for two people)',data= df_RATE_BTM)
plt.ylabel('Cost for 2')


# The Regression plot for the entire restaurant data shows that there sure is a increase in rating with increase in price per two people, but this also could be because of those outliers.

# In[ ]:


sns.lmplot(x='votes',y='approx_cost(for two people)',data= df_RATE_BTM)
plt.ylabel('Cost for 2')


# The Regression plot for the entire restaurant data shows that there sure is a increase in votes with increase in price per two people, but this also could be because of those outliers.

# In[ ]:



sns.jointplot(y='votes', x='rate', data=df_RATE_BTM, kind='hex',gridsize=20)

sns.jointplot(y='votes', x='rate', data=df_RATE_BTM)


# In[ ]:


sns.jointplot(y='votes', x='approx_cost(for two people)', data=df_RATE_BTM, kind='hex',gridsize=20)

sns.jointplot(y='votes', x='approx_cost(for two people)', data=df_RATE_BTM)


# In[ ]:


sns.jointplot(y='rate', x='approx_cost(for two people)', data=df_RATE_BTM, kind='hex',gridsize=20)

sns.jointplot(y='rate', x='approx_cost(for two people)', data=df_RATE_BTM)


# In[ ]:


import plotly_express as px
px.scatter(df_RATE_BTM, x="rate", y="votes",color='approx_cost(for two people)', marginal_y="violin",
           marginal_x="box", trendline="ols")


# Another great visualisation to show how ratings generally fall between 3 and 4 for restaurants with less cost for 2, and hikes to 5 for a couple high end restaurants. As we have concluded it being the experiece that money gets you or the service and food quality. 

# ##  Hope you guys liked the efforts and insights. Just a beginner trying to learn through playful datasets.
#   ##                                 Share some love.

# In[ ]:


tom =df_main['reviews_list'].apply(lambda x: x.replace('RATED\\n',''))


# In[ ]:


tom = tom.apply(lambda x: x.replace('\'',""))


# In[ ]:


tom = tom.apply(lambda x: x.replace('\\n',""))


# In[ ]:


tom[0].split(')')


# In[ ]:




