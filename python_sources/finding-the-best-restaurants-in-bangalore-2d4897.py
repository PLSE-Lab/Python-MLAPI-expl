#!/usr/bin/env python
# coding: utf-8

# ## Breakdown of this notebook:
# 1. **Loading the dataset**: Load the data and import the libraries.
# 1. **Data Cleaning**: 
#      * Deleting redundant columns.
#      * Renaming the columns.
#      * Dropping duplicates.
#      * Cleaning individual columns.
# 1. **Data Visualization:** Using plots to find relations between the features.
# 1. **Finding the best cheap restaurants:** 
#       * **Cheapest, Highest rated and largely voted**.
#       * Is there a **relation** between **cuisine,location and the cost**?
# 1. **Exploring the best expensive restaurants.**
#       * Restaurants that are **expensive, Highest rated and largely voted**.
#       * Is there a **relation** between **restaurant type,location and the cost**?
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/zomato.csv')
data.head()
# data.shape


# In[ ]:


print("Percentage null or na values in df")
((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)


# In[ ]:


data.rate = data.rate.replace("NEW", np.nan)
data.dropna(how ='any', inplace = True)


# In[ ]:


# data.url.unique()
# data.address.unique()
# data.phone.unique()
# data[['location','listed_in(city)']]


# * If you look closely at each column of the dataframe closely you will notice that there are some columns that won't contribute to the ratings and reviews. The **url** or the full **address** of the restaurant or their **phone number** can't justify their ratings or reviews.
# * Note that only the address column is omitted from the dataframe and not the listed_in(city) column,because location details in listed_in(city) column can be very useful in extracting the information about the restaurants.
# * Also,location and listed_in(city) are the same columns.So, we **drop the location column**.
# * The names of columns are a bit non descriptive and confusing so its better to **rename** some of these columns.

# In[ ]:


del data['url']
del data['address']
del data['phone']
del data['location']
data.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality','listed_in(type)': 'restaurant_type'}, inplace=True)
data.head()


# As you can see the rate column is string type with an extra /5 with all the ratings. This should be cleaned.It is important to convert the string back to float !!

# In[ ]:


X = data
X.rate = X.rate.astype(str)
X.rate = X.rate.apply(lambda x: x.replace('/5',''))
X.rate = X.rate.apply(lambda x: float(x))
X.head()


# ### Data Visualization

# #### Are the locations of restaurants localised to specific parts of city?

# In[ ]:


rcParams['figure.figsize'] = 15,7
g = sns.countplot(x="locality",data=data, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('locality',size = 20)


# ### Restaurant type distribution plot

# In[ ]:


rcParams['figure.figsize'] = 15,7
g = sns.countplot(x="rest_type",data=data, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('rest_type',size = 20)


# ### Is there a relation between online order option and rating of the restaurant?

# In[ ]:


plt.rcParams['figure.figsize'] = (3, 4)
plt.style.use('_classic_test')

X['online_order'].value_counts().plot.bar(color = 'cyan')
plt.title('Online orders', fontsize = 20)
plt.ylabel('Number of orders', fontsize = 15)
plt.show()


# In[ ]:


# X[['online_order','rate']].groupby(['rate']).sum(axis=0)
plt.rcParams['figure.figsize'] = (15, 9)
x = pd.crosstab(X['rate'], X['online_order'])
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])
plt.title('online order vs rate', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()


# *You are more likely to receive a higher rating if your restaurant offers* **online order** *option.*

# ### Is there a relation between table booking option and rating of the restaurant?

# In[ ]:


plt.rcParams['figure.figsize'] = (7, 9)
plt.style.use('_classic_test')

X['book_table'].value_counts().plot.bar(color = 'cyan')
plt.title('Table booking', fontsize = 20)
plt.ylabel('Number of bookings', fontsize = 15)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 9)
x = pd.crosstab(X['rate'], X['book_table'])
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])
plt.title('table booking vs rate', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()


# 
# 
# *You can say that you have the table booking option for Highly rated restaurants.*
# 
# 
# 

# ### Cost distribution of all the restaurants in City

# In[ ]:


X.head()
X.average_cost = X.average_cost.apply(lambda x: x.replace(',',''))
X.average_cost = X.average_cost.astype(int)
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(X['average_cost'],ax=ax)
ax.set_title('Cost Distrubution for all restaurants')


# In[ ]:


restaurantTypeCount=data['restaurant_type'].value_counts().sort_values(ascending=True)
slices=[restaurantTypeCount[0],
        restaurantTypeCount[1],
        restaurantTypeCount[2],
        restaurantTypeCount[3],
        restaurantTypeCount[4],
        restaurantTypeCount[5],
        restaurantTypeCount[6]]
labels=['Pubs and bars','Buffet','Drinks & nightlife','Cafes','Desserts','Dine-out','Delivery ']
colors = ['#3333cc','#ffff1a','#ff3333','#c2c2d6','#6699ff','#c4ff4d','#339933']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their Type", bbox={'facecolor':'2', 'pad':5})

fig.set_size_inches(12,12)
plt.show()


# ## Finding the best restaurants:-
# ### The criteria for best restaurants would be  
# * **cheapest**,  
# * **highly rated**, 
# * **reliable**(large number of votes) options.

# #### First step will be to find the restaurants with average cost 1/4th the average cost of most expensive restaurant in our dataframe.
# 
# 
# *Let me explain:-*
# The most **expensive** restaurant has an average meal cost= **6000**. We'll try to stay economical and only pick the restaurants that are** 1/4th of 6000.**
# *Uncomment the comments in code below to get a clearer idea !*

# In[ ]:


# X.average_cost.describe()
# maxi=X.average_cost.max()
# mean=X.rate.mean()
# print(mean)
X= X.drop_duplicates(subset='name',keep='first')
# dups_name = X1.pivot_table(index=['name'],aggfunc='size')
newdf=X[['name','average_cost','locality','rest_type','cuisines']].groupby(['average_cost'], sort = True)
newdf=newdf.filter(lambda x: x.mean() <= 1500)
newdf=newdf.sort_values(by=['average_cost'])

newdf_expensive=X[['name','average_cost','locality','rest_type','cuisines']].groupby(['average_cost'], sort = True)
newdf_expensive=newdf_expensive.filter(lambda x: x.mean() >= 3000)
newdf_expensive=newdf_expensive.sort_values(by=['average_cost'])
# newdf


# **Now lets find the highest rated restaurants i.e rating above 4.0**
# *Uncomment the last line in code below to get a clearer idea*

# In[ ]:


newdf_rate=X[['name','rate']].groupby(['rate'], sort = True)
newdf_rate=newdf_rate.filter(lambda x: x.mean() >= 4.0)
newdf_rate=newdf_rate.sort_values(by=['rate'])
X.rate.value_counts()
X.rate.unique()
X.nunique()
# newdf_rate


# 
# 
# 
# 

# **Now, we'll merge both the dataframes obtained above to get the intersection of both  i.e the highest rated and cheapest restaurants !!**

# In[ ]:


s1 = pd.merge(newdf, newdf_rate, how='inner', on=['name'])

s2= pd.merge(newdf_expensive, newdf_rate, how='inner', on=['name'])

print("Cheap restaurants with low cost,high rating \n")
s1


# 
# 
# 
# 

# In[ ]:


print("Expensive restaurants with high cost,high rating \n")
s2


# 
# 
# 
# 
# 

# ### Find the most reliable restaurants: 
# **Voted more the mean number of votes:- 175**  
# *Uncomment the last line in code below to get a clearer idea*

# In[ ]:


# X1.votes.describe()
newdf_votes=X[['name','votes']].groupby(['votes'], sort = True)
newdf_votes=newdf_votes.filter(lambda x: x.mean() >= 175)
newdf_votes=newdf_votes.sort_values(by=['votes'])
# newdf_votes


# 
# 
# 
# 
# 
# 
# 

# ## These are the most reliable, highest rated and economical restaurants:- 
# 
# We obtain this dataframe by simply taking the intersection of all the dataframes obtained above.
# 
# 
# This dataframe obtained below shows the restaurants whose:
# * Cost is below **1500**
# * Rating is above **4.0**
# * Votes are above **175**

# In[ ]:


s = pd.merge(s1, newdf_votes, how='inner', on=['name'])
s=s.sort_values(by=['votes','rate'],ascending=False)

s2 = s[s.cuisines == 'South Indian']

print("Top voted cheap South Indian restaurants in Bangalore,high votes,high rating")
s2


# #### Best restaurant options under 500 Rupees (average cost):-
# * **Brahmin's Coffee Bar** with average cost=100 and rating=4.8 and votes=2679
# * **CTR**  with average cost=150 and rating=4.7 and votes=4408
# * **Veena Stores** with average cost=150 and rating=4.5 and votes=2407
# * **O.G. Variar & Sons** with average cost=200 and rating=4.8 and votes=1156
# * **Mavalli Tiffin Room (MTR)** with average cost=250 and rating=4.5 and votes=2896
# * **Belgian Waffle Factory** with average cost=400 and rating=4.9 and votes=1746
# 

# #### Other findings:-

# * Also, observe that these cheaper options (cost<500) are all either **Quick Bites, Cafe or Dessert Parlour**.
# * **Casual Dining restaurants** start above 600
# * 6 out of 10 of the cheapest restaurants serve **South Indian Cuisine**
# * As for the **location**, these cheap restaurant option are **scattered and not localised** to any specific location of the city.

# ## We can also explore the expensive options :-
# 
# Here, we are only picking up the restaurants that **cost more than** **3000**(half of most expensive restaurant) and are highest rated , have large votes.

# In[ ]:


s = pd.merge(s2, newdf_votes, how='inner', on=['name'])
s=s.sort_values(by=['average_cost'])
s


# 
# 
# *No surprises there!!*
# 
# **The Oberoi Hotel, Karavalli and JW Marriott** make this high profile list
# 
# Interestingly, all these restaurants have the **same location**- **Brigade Road**  and **same restaurant type**- **Fine dining**

# 
# 
# 
# 

# ## Please upvote and feel free to give any feedback/comment below!!
