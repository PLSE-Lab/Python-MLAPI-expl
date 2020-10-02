#!/usr/bin/env python
# coding: utf-8

# ## We will start the EDA for Zomato dataset. Starting from the basic analysis we will try to find some hidden insights from the data.

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')

df1 = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1") #import the zomato restaurants data file
df2 = pd.read_excel('../input/Country-Code.xlsx') #import zomato country code file


# #### Since zomato country code file only has country names and code, it makes sense to merge both the data sets to form a single data set on which it will be easy to work for us.

# In[7]:


zomato = pd.merge(df1,df2, on = 'Country Code')
zomato.head(3)
#Now we have both country code and name in same data set.


# ####  Now lets have a quick look at the data set columns and data types and other basic info about Null values.

# In[8]:


zomato.info()


# * #### From the above information we can see that only one column 'Cuisines' has some NULL values which are very less as compared to the total rows. Hence, for the sake of convenience, we can ignore the Null values for the time being.
# 
# #### -- Now Lets start our analysis and see which top 15 restaurants in the data set have maximum number of outlets

# In[9]:


#Top 15 Restro with maximum number of outlets
ax=zomato['Restaurant Name'].value_counts().head(15).plot.bar(figsize =(12,6))
ax.set_title("Top 15 Restarurents with maximum outlets")
for i in ax.patches:
    ax.annotate(i.get_height(), (i.get_x() * 1.005, i.get_height() * 1.005))
#-------------------------------------------------------------


# * ####  We can see CCD, Dominos and Subway are the clear winners here. But what about the other restaurants apart from top 15!!
# * ####  A better way to visualize the no. of outlets for more restaurants can be with wordcloud.

# In[10]:


stopwords = set(STOPWORDS)

wordcloud = (WordCloud(width=500, height=300, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(zomato['Restaurant Name'].value_counts().head(35)))
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#---------------------------------------------------------


# * ####  As we can see in the picture, we can quickly have a look at the top 35 restaurants (by number of outlets). Bigger the name, more the number of outlets it has.
# #### Now lets see how many countries are represented in the data set and how many restaurants each country has in the data.

# In[11]:


#No. of unique countries & number of restro listed in data set
zomato.Country.value_counts()


# * ####  We can clearly see that this data has maximum number of listings from India. So it makes sense to have some analysis for the data of Indian restaurants.
# ####  Lets see top 10 cities in India represented in the data set.

# In[12]:


#Number of restro on zomato in different cities in India
zomato.loc[zomato['Country']=='India'].City.value_counts().head(10)
#--------------------------------------------------------


# * ####  Clearly this data has maximum number of restaurants listed in Delhi NCR region (Comprising of New Delhi, Gurgaon, Noida, Faridabad). Other cities have almost equal distribution but significantly less as compared to Delhi NCR region.
# 
# #### Lets find out top 10 restaurants in the data with highest number of votes.

# In[13]:


#Top 10 Restro with highest no. of votes
max_votes =zomato.Votes.sort_values(ascending=False).head(10)
zomato.loc[zomato['Votes'].isin(max_votes)][['Restaurant Name','Votes']]
zomato.loc[zomato['Votes'].isin(max_votes)][['Restaurant Name','Votes']].plot.bar(x='Restaurant Name', y='Votes',
                                                                                  figsize = (10,6), color='purple')
#-------------------------------------------------------


# * ####  We can see in the above graph that the restaurants with maximum number of outlets are not the one whic have highest number of votes. The above list is totally different that our list of top 10 restaurants with maximum number of outlets.
# 
# #### Lets separate the data for restaurants from India for ease of analysis. If we consider the whole data set, we will have to take into account the currency values for each country. For example while considering 'Average cost for two' on graph the data should be uniform.

# In[14]:


zomato_india = zomato.loc[zomato['Country']=='India']
zomato_india.head(3)


# ####  Lets try to find out If there is any relation between average cost for two and aggregate rating of restaurants.

# In[15]:


#Is there any relation between average cost for two and aggregate rating of restaurants
zomato_india.plot.scatter(x='Average Cost for two',y='Aggregate rating',figsize=(10,6), color='orange', title="Cost vs Agg Rating")
#-------------------------------------------------------------


# * ####  From the above graph, we can see that most of the data is clustered around cost upto 2000 and rating values from 2 to 4.5 approximately. There are few restaurants with cost range between 2500 to 6000.
# *  ####  There are some outliers in the data where cost is listed as 0 because it might not be captured for those restaurants.
# ####  A more concise and detailed view of above data can be via hex plot as shown below.

# In[16]:


#Better view of relation between average cost for two and aggregate rating of restaurants
sns.jointplot(x='Average Cost for two',y='Aggregate rating',kind ='hex',gridsize=18,data =zomato_india,color='blue')
#--------------------------------------------------


# * ####  In above graph, we can see more clearly that the maximum number of rating values are around 3 to 3.5 and the 'Avg cost for two, for maximum data is also up to 1000.
# 
# #### Lets see what are the top 10 cuisines- served by maximum number of restaurants

# In[17]:


#Top 10 Cuisines served by restaurants
zomato_india['Cuisines'].value_counts().sort_values(ascending=False).head(10)
zomato_india['Cuisines'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6), 
title="Most Popular Cuisines", autopct='%1.2f%%')
plt.axis('equal')
#-------------------------------------------------------


# * #### From the above graph, we can clearly see that 'North Indian' cuisine is the most popular cuisine and it makes sense as well since the maximum data has restaurants listed from North India.
# 
# #### Next lets try to find out does there exist any correlation among 'avg cost for two', 'price range' and 'agg rating'

# In[18]:


#Correlation among avg cost, price range, agg rating
zomato_corr = zomato[['Average Cost for two', 'Price range', 'Aggregate rating']]
sns.heatmap(zomato_corr.corr(),linewidth=1.0)
#cmap='PuOr' cmap='YlGnBu'
#------------------------------------------------------------


# * #### We can see 'price range'-'agg rating' appear to be correlated upto an extent but still, witht this data, the above graph is not good enough to provide a confident answer.
# 
# #### Lets try to find out more insights for correlation among 'avg cost for two', 'price range' and 'agg rating' by using pair plot but keeping our data limited to top 5 cities with max restaurants.

# In[19]:


#More insight for correlation by using pair plot keeping top 10 cities with max restro
top5_indian_cities = ['New Delhi', 'Gurgaon', 'Noida','Faridabad', 'Ghaziabad']
zomato_p = zomato.loc[zomato['City'].isin(top5_indian_cities)]
zomato_pair = zomato_p[['Average Cost for two', 'Price range', 'Aggregate rating', 'City']]
sns.pairplot(zomato_pair, size=3, hue='City', palette="husl")
#-----------------------------------------------


# *  #### From the graph above we can see, the same price range 4, has different 'Average cost for two' in different cities. In New Delhi, price range 4 has average cost from around 3500 to 8000 but same price range 4 has 'Average cost' around 2000 to 4000 in Noida.
# * #### From 'Agg Rating' and 'Price Range' we can see that all the price ranges have mix of 'Agg Ratings'
# 
# #### Next lets try to find out similar relation by keeping our data limited to top 5 cuisines instead of cities.

# In[20]:


#Correlation of cost, price range with top 10 cuisines
top5cuisines_list=['North Indian', 'North Indian, Chinese', 'Fast Food', 'North Indian, Mughlai', 'Cafe' ]

zomato_cuisines = zomato.loc[zomato['Cuisines'].isin(top5cuisines_list)]
zomato_cuisines_data = zomato_cuisines[['Average Cost for two', 'Price range', 'Aggregate rating', 'Cuisines']]
sns.pairplot(zomato_cuisines_data, size=3, hue='Cuisines')

#--------------------------------------------------------


# * #### From the graphs above we can see that 'Agg Rating' is getting higher when 'Avg. cost for two' is increasing.
# * #### North Indian cuisine is scoring the highest 'Avg cost for two' among all top 5 cuisines.
# 
# #### Now lets analyze top 10 cuisines with 'price range' and 'agg rating' and look at our findings.

# In[21]:


#ANalysis of top 10 cuisines with price range and agg rating
top10cuisines_list=['North Indian', 'North Indian, Chinese', 'Fast Food', 'North Indian, Mughlai', 'Cafe', 'Bakery',
                   'North Indian, Mughlai, Chinese', 'Bakery, Desserts', 'Street Food' ]
zomato_cuisines = zomato.loc[zomato['Cuisines'].isin(top10cuisines_list)]
zomato_cuisines_data = zomato_cuisines[['Average Cost for two', 'Price range', 'Aggregate rating', 'Cuisines']]

fig, axx =plt.subplots(figsize=(16,8))
sns.barplot(x='Cuisines', y='Aggregate rating', hue='Price range', data=zomato_cuisines_data, palette="Set1")
axx.set_title("Analysis of Top10 Cuisines with price range and Agg. rating ")
#------------------------------------------------------


# * #### From the above graph, we can clearly see that among the top 10 cuisines, price range 4 has always got the highest rating!! 
# * #### May be we can assume that the high end restaurants (with higher price range) serves the best and hence the higher 'Agg rating' in all cuisines.
# * #### Similar is the case with price range 1 which has lowest 'Agg rating' amoung all cuisines
# * #### In all the cuisines, we can see the trend, higher the price range, better is the 'Agg rating'
# #### Now lets see what is the most common 'Agg Rating' for each type of cuisine among top 10 cuisines.

# In[22]:


#Most common agg. rating for each type of cuisine
top10cuisines_list=['North Indian', 'North Indian, Chinese', 'Fast Food', 'North Indian, Mughlai', 'Cafe', 'Bakery',
                   'North Indian, Mughlai, Chinese', 'Bakery, Desserts', 'Street Food' ]
zomato_cuisines = zomato.loc[zomato['Cuisines'].isin(top10cuisines_list)]
zomato_cuisines_data = zomato_cuisines[['Average Cost for two', 'Price range', 'Aggregate rating', 'Cuisines']]
fig, axx =plt.subplots(figsize=(16,6))
sns.boxplot(x='Cuisines', y='Aggregate rating', data=zomato_cuisines_data)
#----------------------------------------------------------------------


# * #### From the above box plot graph, we can see that number one cuisine in our top 10 list which is 'North Indian', mid 50% of its 'Agg ratings' are upto 3 with median rating of around 2.8 which is in fact the lowest median rating among all the cuisine types.
# * #### 'Cafe' catrgory of cuisines manages to maintain highest median rating of around 3.4
# * #### The popularity of cuisine does not necessary means it is highly rated. In our case 'North Indian' cuisine is the most popular in our data but maintains least median rating. It is widely served may be because most of the restaurants are in North India
# * #### Bakery, Fast Food, Street Food, Desserts are almost equivalent in terms of rating comparison.
# 
# 
# #### Now lets find out percentage rating of restaurants in our top 5 cities and then plot it on a graph and make some findings.
# #### Percentage rating means what percentage of restaurants in a city are Excellent, average etc.

# In[23]:


#Restaurant Percentage wise rating in top 5 cities
top5_indian_cities = ['New Delhi', 'Gurgaon', 'Noida','Faridabad', 'Ghaziabad']
zomato_rate = zomato.loc[zomato['City'].isin(top5_indian_cities)]

#Find total number of restaurants
total_restro = zomato_rate.groupby(['City'], as_index=False).count()[['City','Restaurant ID']]
total_restro.columns=['City','Total Restaurants']

#Find total rating count of each type
top5rest = zomato_rate.groupby(['City','Rating text'], as_index=False)[['Restaurant Name']].count()
top5rest.columns=['City','Rating text', 'Total Ratings']

#Merge both the dataframes and calculate percentage
top5restro_rating_percent = pd.merge(total_restro, top5rest, on='City')
top5restro_rating_percent['Percentage']= (top5restro_rating_percent['Total Ratings']/
                                       top5restro_rating_percent['Total Restaurants'])*100

top5restro_rating_percent


# #### Now lets plot the above data in the form of a graph and list our findings.

# In[24]:


#Plot Rating percentage of restaurants in top 5 cities
fig, axx =plt.subplots(figsize=(12,6))
axx.set_title("Percentage Rating of Restaurants in Top 5 Cities")
sns.barplot(x='City', y='Percentage',hue='Rating text', data=top5restro_rating_percent, palette='Set3')


# * #### From the above graph, we can see that maximum number of restaurants in top 5 cities have the rating 'Average'
# * #### Excellent rating is almost negligible.
# * #### There are significant number of restaurants which are not rated as well. So the data my needs to take that into account while making any accurate predictions.
