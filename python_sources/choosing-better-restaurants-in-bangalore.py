#!/usr/bin/env python
# coding: utf-8

# # Food + Data Can we choose restaurants better ? 
# 
# Well we got the cuisine rating and location....so lets just jump into it 
# 
# 1)  Where can I score some good food around here ?
# 
# 2)  What could be the most reviewed food chain in bangalore ?
# 
# 3) What does it cost to eat a good meal in Bangalore ?
# 
# 4) What kind of cuisines are popular in different areas ?
# 
# 5) May I suggest, The concierge function
# 
# 

# ![img ](https://i.kym-cdn.com/editorials/icons/mobile/000/000/390/Screen_Shot_2019-04-07_at_1.37.50_PM.jpg)

# [](https://i.kym-cdn.com/editorials/icons/mobile/000/000/390/Screen_Shot_2019-04-07_at_1.37.50_PM.jpg)

# In[ ]:


# All Packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings('ignore')


# In[ ]:


# Get a sense of the data
data=pd.read_csv('../input/zomato.csv')
data.sample(3)


# In[ ]:


# Null and stuff
print(data.isnull().sum())


# ** Now based on what we have we decide what we want  **
# 
# - Urls and phone numbers don't seem unnecessary for now 
# 
# - Review list is an NLP problem in itself (Maybe later)
# 
# - Dish_liked is mostly null :( 
# 
# - Cuisine covers menu items 
# 
# - Location covers listed_in(city)

# In[ ]:


data = data.drop(['url','phone','reviews_list','dish_liked','menu_item','listed_in(city)'],axis=1)


# In[ ]:


df = data.dropna().reset_index(drop=True)


# In[ ]:


data.head()


# ## Let's get some of the basic stats 
# 
# #  Where can I score some good food around here ?
# 
# The more foodie friendly areas of bangalore where every street is filled with tons of options leaving you confused and hungry
# 
# BTM beats the lot by a lot followed by the infamous HSR and Koramangala
# 
# 
# 

# In[ ]:


location_count  = data['location'].value_counts()
location_count = location_count[:25,]
plt.figure(figsize=(20,9))
sns.barplot(location_count.index, location_count.values, alpha=0.9)
plt.xticks(rotation=80,fontsize=20)
plt.title('Number of restaurants per Locality',fontsize=20)
plt.ylabel('Number of restaurants', fontsize=20)
plt.xlabel('', )
plt.show()


# # What could be the most reviewed food chain in bangalore ?
# 
# Well it is cafe coffee day...well every corner has one so no wonder...But closely followed by Onesta (Pizza time) 

# In[ ]:


print("The Most reviewed food chains \n")
print(data['name'].value_counts()[:10])


# # What does it cost to eat a good meal in Bangalore ?
# 
# The least amount of money you must carry on your dinner date (227 bucks)
# 

# In[ ]:


data_cost = data[data['approx_cost(for two people)'].notnull()]
cost_for_two = [re.sub("[^0-9]", "", str(i)) for i in data_cost['approx_cost(for two people)']]
cost_for_two = filter(None, cost_for_two)
cost_for_two = [int(i) for i in cost_for_two]
data_cost['cost_for_two'] = cost_for_two
data_cost = data_cost[['location','cost_for_two']]
print("Average cost of meal for 2 is:",sum(cost_for_two)/len(cost_for_two)/2)


# In[ ]:


dc = data_cost[(data_cost['location'].isin(list(data_cost['location'].value_counts()[:25].index)))]


# MG road and Ulsoor seem to  have more of a mix of expensive and affordable places 

# In[ ]:


plt.figure(figsize=(20,7))
sns.boxplot(data=dc,
                x = 'location',
                y = 'cost_for_two')
plt.xticks(rotation=80,fontsize=20)
plt.title('Box plots of area name and cost for two ',fontsize=20)
plt.ylabel('Cost for two', fontsize=20)
plt.xlabel('')
plt.show()


# # What kind of cuisines are popular in different areas ?

# In[ ]:


data_cuisines = data.dropna(subset=['cuisines'])
cuisines= [i.split(',') for i in data_cuisines['cuisines']]
cus = []
for i in cuisines:
    cus.append([j.replace(' ','') for j in i])
data_cuisines['cuisine'] = cus
data_cuisines = data_cuisines[['location','cuisine']]
area_wise_cuisines = data_cuisines.groupby('location')['cuisine'].apply(list)
area_wise_cuisines= area_wise_cuisines.to_frame()
area_wise_cuisines =area_wise_cuisines.reset_index()


# In[ ]:


area_wise_cuisines['cuisine'] = [list(itertools.chain.from_iterable(i)) for i in area_wise_cuisines['cuisine']]
area_wise_cuisines['location'] = [i.lower() for i in area_wise_cuisines['location']]
area_wise_cuisines.head()


# # Fn what_should_i_eat_in  (area_name)
# 
# Well what kinda food is popular in koramangala ?

# In[ ]:


def what_should_i_eat_in(area_name):
    try:
        area_name = area_name.lower()
        index_ = (area_wise_cuisines[(area_wise_cuisines['location'] == area_name)]['cuisine']).index[0]
        s = Counter(area_wise_cuisines['cuisine'][index_]).most_common(7)
        print("The Most Popular cuisines in ",area_name," in the order of popularity\n")
        for i in s:
            print(i[0])
    except:
        print(area_name,"NOT FOUND")


# In[ ]:


### NORTH INDIAN FOOD
what_should_i_eat_in('koramangala')


# In[ ]:


### SOUTH INDIAN FOOD
what_should_i_eat_in('majestic')


# In[ ]:


## WELL MEH
what_should_i_eat_in('random place')


# In[ ]:


df = df.drop_duplicates(subset='name', keep="last")
df['rate'] = df['rate'].replace({'NEW': '0/5','-':'0/5'})
df['rating'] = [float(i.split('/')[0]) for i in df['rate']]
df['rating*votes'] = df['rating'] * df['votes']


# ## Restaurants rated consistently high by customers 
# 
# The most reviewed yet highly rated places to eat in Bangalore 
# 
# - Byg Brewski Brewery in Sarjapur road
# - Toit in Indiranagr
# - Big Pitcher in	Old Airport Road 

# In[ ]:


# THE TOP 10 LIST
df.sort_values('rating*votes',ascending=False)[['name','location','rating','rating*votes']].head(10).reset_index(drop=True)


# In[ ]:


df.sort_values('rating',ascending=False)[['name','location','rating']].head(10)


# In[ ]:


df = data.dropna().reset_index(drop=True)
df['rate'] = df['rate'].replace({'NEW': '0/5','-':'0/5'})
df['rating'] = [float(i.split('/')[0]) for i in df['rate']]


# In[ ]:


data_rate = df.groupby('name')['rating'].mean().to_frame()
data_rate.sort_values('rating',ascending=False).head(10)


# # May I suggest
# 
# Become a data backed concierge and choose the best place to order from based on 
# 
# - Location
# - Rating 
# - Cuisines 
# - Cost for 2 

# In[ ]:


df = data.dropna().reset_index(drop=True)
df['rate'] = df['rate'].replace({'NEW': '0/5','-':'0/5'})
df['rating'] = [float(i.split('/')[0]) for i in df['rate']]
cost_for_two = [re.sub("[^0-9]", "", str(i)) for i in df['approx_cost(for two people)']]
cost_for_two = filter(None, cost_for_two)
cost_for_two = [int(i) for i in cost_for_two]
df['cost_for_two'] = cost_for_two
data_needed = df[['name','online_order','location','cuisines','cost_for_two','rating']]


# In[ ]:


def may_i_suggest(location,cuisine,cost_for_two):
    try:
        order_from = df[(df['cuisines'].str.match(cuisine)) & (data_needed['location']==location) &
           (data_needed['cost_for_two'] <= cost_for_two)].sort_values('rating',ascending=False)[['name','address']].drop_duplicates().reset_index(drop=True)
        print("BEST PLACES TO ORDER FROM: \n")

        for i in range(0,5):
            print(i+1,'.',order_from['name'][i],'- Addr:',order_from['address'][i],"\n")
    except:
        print("NOT ABLE TO FIND ONE, I SUGGEST GOING OUTSIDE IT IS A WONDERFUL DAY AFTER ALL !")


# In[ ]:


### SUGGESTIONS FOR NORTH INDIAN PLACES UNDER 800 FOR TWO IN BANASHANKARI

may_i_suggest('Banashankari','North Indian',800)


# In[ ]:


### SUGGESTIONS FOR SOUTH INDIAN PLACES UNDER 10 BUCKS FOR TWO IN BTM

may_i_suggest('BTM','South Indian',10)


# In[ ]:


may_i_suggest('HSR','Italian',700)


# In[ ]:


may_i_suggest('Whitefield','Pizza',700)


# In[ ]:


may_i_suggest('Whitefield','Beverages',700)


# 
# ![](https://www.stickers-muraux.fr/ar-stickers-bon-appetit-23825.jpg)
