#!/usr/bin/env python
# coding: utf-8

# **Overview**

# My primary objective in this notebook data analysis is to present a comparable study of the cost of living,availablities of the rooms in various neighbourhoods across NYC.

# In[ ]:


#importing necessery libraries for future analysis of the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#using pandas library and 'read_csv' function to read Airbnb csv file as file already formated for us from Kaggle
airbnb=pd.read_csv('../input/AB_NYC_2019.csv')
#Display the file with the first few lines.
airbnb.head(5)


# In[ ]:


#checking type of every column in the dataset
airbnb.dtypes


# **Cleaning up the data.**
# There are data that are redundant for the data usage.We would like to drop those tables using the below given commands.

# In[ ]:


#after looking at the head of the dataset we already were able to notice some NaN values, therefore need to examine missing values further before continuing with analysis
#looking to find out first what columns have null values
#using 'sum' function will show us how many nulls are found in each column in dataset
airbnb.isnull().sum()


# In[ ]:


#checking amount of rows in given dataset to understand the size we are working with
len(airbnb)


# To fill in the places with the reviews per month with zero  where it is null values.

# In[ ]:


airbnb.fillna({'reviews_per_month':0}, inplace =True)
airbnb.reviews_per_month.isnull().sum()


# In[ ]:


# Displaying the unique neighbourhood group values
airbnb.neighbourhood_group.unique()


# **Presenting the room_type availbilities in Different neighbourhood_groups**

# In[ ]:


airbnb.room_type.value_counts()


# **Breakdown of the roomtypes among the neighbourhood groups is shown below.**

# In[ ]:


rooms=airbnb.loc[airbnb['neighbourhood_group'].isin(['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx'])]
#using catplot to represent multiple interesting attributes together and a count
viz_3=sns.catplot(x='room_type',col='neighbourhood_group', data=rooms, kind='count',height=5, aspect=0.7)
viz_3.set_xticklabels(rotation=90)


# **Average prices in each neighbourhood group**

# In[ ]:


ave_ng=airbnb.groupby('neighbourhood_group', as_index=False)['price'].mean()
ave_ng


# **AVERAGE PRICE IN EACH NEIGHBOURHOOD_GROUP FOR EACH ROOM_TYPES **

# In[ ]:


airbnb_ng_average = airbnb.groupby(['neighbourhood_group','room_type']).agg({'price':'mean'}).reset_index()
airbnb_ng_average


# **Now,let's display them in a graph for a better understanding.**

# In[ ]:


plt.figure(figsize=(12,8))
sns.set_palette("Set1")
sns.lineplot(x='neighbourhood_group', y='price', 
             data=airbnb_ng_average[airbnb_ng_average['room_type']=='Entire home/apt'],
             label='Entire home/apt')
sns.lineplot(x='neighbourhood_group', y='price', 
             data=airbnb_ng_average[airbnb_ng_average['room_type']=='Private room'],
             label='Private room')
sns.lineplot(x='neighbourhood_group', y='price', 
             data=airbnb_ng_average[airbnb_ng_average['room_type']=='Shared room'],
             label='Shared room')
plt.xlabel("Neighbourhood_group", size=13)
plt.ylabel("Average price", size=13)
plt.title("Neighbourhood_group vs Average Price vs Room_type",size=15, weight='bold')



# From the graph, it is obvious and clear that the Entire home/apt is way too high in pricing  and among the neighbourhood groups,Manhattan is the expensive place to stay.

# **Finding the Average price of the neighbourhoodgroups among the room_types and plotting using a pivot table**

# In[ ]:


#Displaying the neighbour hood group room average pricing using #renaming a column
airbnb_ng_average = airbnb_ng_average.rename(columns = {'price': 'ave_ng_price'})
#creating a pivot table
airbnb_ng_average_pivot = pd.pivot_table(airbnb_ng_average, values='ave_ng_price', 
                           index=['neighbourhood_group'], columns=['room_type'])
airbnb_ng_average_pivot


# In[ ]:



ax=airbnb_ng_average_pivot.plot(kind='bar', width = 0.5)
ax.set_xlabel('neighbourhood_group', fontsize = 20)
ax.set_ylabel('average_price', fontsize = 20)
labels=list(airbnb_ng_average_pivot.index[:5])
ax.set_xticklabels(rotation=30,labels=labels,fontsize=10)
plt.show()



# Now,let's try to display the average price for all the neighbourhoods vs neighbourhood groups

# In[ ]:


airbnb_ngg_average = airbnb.groupby(['neighbourhood_group','neighbourhood']).agg({'price':'min'}).reset_index() 
plt.figure(figsize=(12,8))
sns.set_palette("Set1")
sns.lineplot(x='neighbourhood', y='price', 
             data=airbnb_ngg_average[airbnb_ngg_average['neighbourhood_group']=='Brooklyn'],
             label='Brooklyn')
sns.lineplot(x='neighbourhood', y='price', 
             data=airbnb_ngg_average[airbnb_ngg_average['neighbourhood_group']=='Manhattan'],
             label='Manhattan')
sns.lineplot(x='neighbourhood', y='price', 
             data=airbnb_ngg_average[airbnb_ngg_average['neighbourhood_group']=='Queens'],
             label='Queens')
sns.lineplot(x='neighbourhood', y='price', 
             data=airbnb_ngg_average[airbnb_ngg_average['neighbourhood_group']=='Bronx'],
             label='Bronx')
sns.lineplot(x='neighbourhood', y='price', 
             data=airbnb_ngg_average[airbnb_ngg_average['neighbourhood_group']=='Staten Island'],
             label='Staten Island')
plt.xlabel("Neighbourhood", size=13)
plt.ylabel("Average price", size=13)
plt.title(" Average Price vs Neighbourhood",size=15, weight='bold')


# Finding 5 cheap neighbourhoods in each neighbourhood groups

# In[ ]:


airbnb_ngg_average = airbnb.groupby(['neighbourhood_group','neighbourhood']).agg({'price':'min'}).reset_index() 
airbnb_ngg_average.sort_values(by='price',ascending=True)
y=airbnb.groupby(['neighbourhood','neighbourhood_group']).agg({'price':'mean'}).reset_index()
g =y.groupby(["neighbourhood_group"]).apply(lambda x: x.sort_values(["price"], ascending = True)).reset_index(drop=True)
# select top N rows within each continent
ngrooms=g.groupby('neighbourhood_group').head(5)
ngrooms=ngrooms.rename(columns = {'price': 'avp'})
ngrooms


# In[ ]:


ax1=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=ngrooms[ngrooms['neighbourhood_group']=='Bronx'],height=5, aspect=0.7)
ax1.set_xticklabels(rotation=90),
ax2=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=ngrooms[ngrooms['neighbourhood_group']=='Brooklyn'],height=5, aspect=0.7)
ax2.set_xticklabels(rotation=90)
ax3=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=ngrooms[ngrooms['neighbourhood_group']=='Manhattan'],height=5, aspect=0.7)
ax3.set_xticklabels(rotation=90)
ax4=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=ngrooms[ngrooms['neighbourhood_group']=='Queens'],height=5, aspect=0.7)
ax4.set_xticklabels(rotation=90)
ax5=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=ngrooms[ngrooms['neighbourhood_group']=='Staten Island'],height=5, aspect=0.7)
ax5.set_xticklabels(rotation=90)


# **Similarly,we will try to find the highest priced neighbourhoods in each neighbourhood group**

# In[ ]:


airbnb_ngg_high = airbnb.groupby(['neighbourhood_group','neighbourhood']).agg({'price':'max'}).reset_index() 
airbnb_ngg_high.sort_values(by='price',ascending=False)
a1=airbnb.groupby(['neighbourhood','neighbourhood_group']).agg({'price':'mean'}).reset_index()
a2 =a1.groupby(["neighbourhood_group"]).apply(lambda x: x.sort_values(["price"], ascending = False)).reset_index(drop=True)
high_rooms=a2.groupby('neighbourhood_group').head(5)
high_rooms=high_rooms.rename(columns = {'price': 'avp'})
high_rooms


# In[ ]:


ax1=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=high_rooms[high_rooms['neighbourhood_group']=='Bronx'],height=5, aspect=0.7)
ax1.set_xticklabels(rotation=90),
ax2=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=high_rooms[high_rooms['neighbourhood_group']=='Brooklyn'],height=5, aspect=0.7)
ax2.set_xticklabels(rotation=90)
ax3=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=high_rooms[high_rooms['neighbourhood_group']=='Manhattan'],height=5, aspect=0.7)
ax3.set_xticklabels(rotation=90)
ax4=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=high_rooms[high_rooms['neighbourhood_group']=='Queens'],height=5, aspect=0.7)
ax4.set_xticklabels(rotation=90)
ax5=sns.relplot(x='neighbourhood',y='avp',col='neighbourhood_group',data=high_rooms[high_rooms['neighbourhood_group']=='Staten Island'],height=5, aspect=0.7)
ax5.set_xticklabels(rotation=90)


# **finding the density distribution of the price range among the neighbourhood groups using viloin plot**

# In[ ]:


density_chk=airbnb[airbnb.price < 500]
#using violinplot to showcase density and distribtuion of prices 
viz_2=sns.violinplot(data=density_chk, x='neighbourhood_group', y='price')
viz_2.set_title('Density and distribution of prices for each neighbourhood_group')


# From the viloin plot, it gives a better understanding of the price range fluctuation

# # Displaying in the wordcloud for the cheapest neighbourhoods from all the neighbourhoodgroups

# In[ ]:


from wordcloud import WordCloud


# In[ ]:


plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='green',
                          width=1800,
                          height=1000
                         ).generate(" ".join(ngrooms.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
plt.show()


# We can see that the cheapest neighbourhood from all the neighbourhood groups are in the decreasing order of the fonts.The bold and big font sized are the cheapest and small sized fonts neighbourhood groups are the expensive ones.This gives a quick view and easier way to remember of those highlighted names.

# In my short first attempt notebook in Kaggle,I have tried to clean up and remove the unwanted data.
# I have tried to find the room_type availablities among the neighbourhood groups and then tried to find out the average price range among the neighbour hood groups.Afterwards, tried to find the cheapest and expensive neighbourhoods among the neighbourhood groups.This way I suppose it gives a brief overview for the people to find out the  the places and room types to choose according to their budget.
# Will try to add in more analysis in the next versions.
# 
