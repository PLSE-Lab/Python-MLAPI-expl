#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import sys


# **Lets get an overview of the data Model scrapped**

# In[ ]:


df=pd.read_csv("../input/zomato.csv")
df.head()


# *The total number of columns are : 17  and rows are : 51717*

# In[ ]:


df.shape


# **Columns description**
#  
# * url contains the url of the restaurant in the zomato website
#  
# * address contains the address of the restaurant in Bengaluru
# 
# * name contains the name of the restaurant
#  
# * online_order whether online ordering is available in the restaurant or not
#  
# * book_table table book option available or not
# 
# * rate contains the overall rating of the restaurant out of 5
# 
# * votes contains total number of rating for the restaurant as of the above mentioned date
#  
# * phone contains the phone number of the restaurant
#  
# * location contains the neighborhood in which the restaurant is located
# 
# * rest_type restaurant type
#  
# * dish_liked dishes people liked in the restaurant
#  
# * cuisines food styles, separated by comma
#  
# * approx_cost(for two people) contains the approximate cost for meal for two people
#  
# * reviews_list list of tuples containing reviews for the restaurant, each tuple
#  
# * menu_item contains list of menus available in the restaurant
#  
# * listed_in(type) type of meal
# 
# * listed_in(city) contains the neighborhood in which the restaurant is listed
# 
# 
# 

# In[ ]:


plt.figure(figsize=(10,7))
chains=df['name'].value_counts()[:30]
sns.barplot(x=chains,y=chains.index,palette='rocket')
plt.title("Most famous restaurants chains in Bangaluru")
plt.xlabel("Number of outlets")


# **These are the top food chains in Bangalore Apparently**

# In[ ]:


plt.figure(figsize=(10,7))
locs=df['location'].value_counts()[:10]
sns.barplot(x=locs,y=locs.index,palette='deep')
plt.title("Top 20 spots in Bangalore with most number of Restaurants")
plt.xlabel("Number of outlets")


# We can see that BTM , HSR , Koramangala 5th Block, JP Nagar and Whitefield has the highest number of food Joints in Bangalore

# So My goal is to have a food joint which will take online orders , I need to know which place takes highest number of online orders.

# In[ ]:




plt.rcParams['figure.figsize'] = (20, 9)
x = pd.crosstab(df['location'], df['online_order']=='Yes')
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['blue','pink'])
plt.title('location vs online order', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()


# In[ ]:


df=df[df['dish_liked'].notnull()]
df.index=range(df.shape[0])
import re
likes=[]
for i in range(df.shape[0]):
    splited_array=re.split(',',df['dish_liked'][i])
    for item in splited_array:
        likes.append(item)

sns.barplot(pd.DataFrame(likes)[0].value_counts().head(10),pd.DataFrame(likes)[0].value_counts().head(10).index,orient='h')


# In[ ]:


df_yes=df[df['online_order']=='Yes']
df_no=df[df['online_order']=='No']
plt.figure(figsize=(10,7))
locs=df_yes['location'].value_counts()[:10]
sns.barplot(x=locs,y=locs.index,palette='rocket')
plt.title("Top 10 spots in Bangalore which takes online order")
plt.xlabel("NLocation")


# So we can see that Koramangala 5th Block and BTM takes the highest number of online order

# In[ ]:



pd.DataFrame(likes)[0].value_counts().tail(20)


# **These are the top 20 disliked food in Bangalore**
# Should definitely avoid adding in the list

# In[ ]:


rating_data=df[np.logical_and(df['rate'].notnull(), df['rate']!='NEW')]
rating_data.index=range(rating_data.shape[0])
import re
rating=[]
for i in range(rating_data.shape[0]):
    rating.append(rating_data['rate'][i][0:3])

rating_data['rate']=rating
rating_data.sort_values('rate',ascending=False)[['name','location','rate','rest_type']].head(60).drop_duplicates()


# These are the highest Rated Restaurants in the city

# I had decided to establish a Casual Dining,Bar Type food place which takes online order.

# In[ ]:


df_cdb=df[df['rest_type']=='Casual Dining, Bar']
df_cdb.head(5)

plt.figure(figsize=(15,10))
locs=df_cdb['location'].value_counts()[:30]
sns.barplot(x=locs,y=locs.index,palette='deep')
plt.title("Top 30 spots in Bangalore with most number of Restaurants of type 'Casual Dining, Bar'")
plt.xlabel("Number of outlets")


# In[ ]:



rating_data=df_cdb[np.logical_and(df_cdb['rate'].notnull(), df_cdb['rate']!='NEW')]
rating_data.index=range(rating_data.shape[0])
import re
rating=[]
for i in range(rating_data.shape[0]):
    rating.append(rating_data['rate'][i][0:3])

rating_data['rate']=rating
rating_data.sort_values('rate',ascending=False)[['name','location','rate','rest_type']].head(10).drop_duplicates()


# **These are the Top Rated Casual Dining,Bar places**

# In[ ]:


plt.figure(figsize=(24, 18))

plt.subplot(2,1,1)
sns.countplot(x= 'rate', hue= 'online_order', data= df[df.rate != 0])
plt.title('Ratings vs online order', fontsize='xx-large')
plt.xlabel('Ratings', fontsize='large')
plt.ylabel('Count', fontsize='large')
plt.xticks(fontsize='large')
plt.xticks(fontsize='large')
plt.legend(fontsize='large')



# This above chart is about rating vs online orders

# In[ ]:


df_cdb=df[df['rest_type']=='Quick Bites']
df_cdb.head(5)

plt.figure(figsize=(15,10))
locs=df_cdb['location'].value_counts()[:10]
sns.barplot(x=locs,y=locs.index,palette='rocket')
plt.title("Top 10 spots in Bangalore with most number of Restaurants of type 'Quick Bites'")
plt.xlabel("Number of outlets")


# In[ ]:




