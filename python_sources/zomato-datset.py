#!/usr/bin/env python
# coding: utf-8

# Hi all, This is a beginners' work. Feel Free to upvote it if you like.Thank you!!!
# 
# 

# WORK DONE IN THIS KERNAL
# * -Importing file and cleaning the data
# * -Dashboard to search the restaurant according to there names,location,rating,cost of two.

# Analysis of the restaurent on the basis of :
# * Their Rating.
# * online Delivery.
# * There type whether bar,dinner,buffet ect.
# * Most liked dishes.
# * Average price of two people.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


zomato= pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
zomato.head()


# In[ ]:


zomato.info()


# DATA CLEANING

# Removing the unnecessary data such as url,address,ohone columns from DataFrame

# In[ ]:


del zomato['url']
del zomato['address']
del zomato['phone']


# In[ ]:


zomato.head()


# In[ ]:


# Replacing restaurents with there ratings given as new to NAN and dopping them finally
zomato['rate']=zomato['rate'].replace('NEW',np.NAN)
zomato['rate']=zomato['rate'].replace('-',np.NAN)
zomato.dropna(how='any',inplace=True)


# In[ ]:


zomato['rate']=zomato.loc[:,'rate'].replace('[ ]','',regex=True)
zomato['rate']=zomato['rate'].astype(str)
zomato['rate']=zomato['rate'].apply(lambda r:r.replace('/5',''))
zomato['rate']=zomato['rate'].apply(lambda r:float(r))


# In[ ]:


# now converting cost from string to integer 
zomato['approx_cost(for two people)']=zomato['approx_cost(for two people)'].str.replace(',','')
zomato['approx_cost(for two people)']=zomato['approx_cost(for two people)'].astype(int)


# In[ ]:


zomato.head()


# In[ ]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[ ]:


location=['Banashankari', 'Basavanagudi', 'Jayanagar', 'Kumaraswamy Layout',
       'Rajarajeshwari Nagar', 'Mysore Road', 'Uttarahalli',
       'South Bangalore', 'Vijay Nagar', 'Bannerghatta Road', 'JP Nagar',
       'BTM', 'Wilson Garden', 'Koramangala 5th Block', 'Shanti Nagar',
       'Richmond Road', 'City Market', 'Bellandur', 'Sarjapur Road',
       'Marathahalli', 'HSR', 'Old Airport Road', 'Indiranagar',
       'Koramangala 1st Block', 'East Bangalore', 'MG Road',
       'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor',
       'Residency Road', 'Shivajinagar', 'Infantry Road',
       'St. Marks Road', 'Cunningham Road', 'Race Course Road', 'Domlur',
       'Koramangala 8th Block', 'Frazer Town', 'Ejipura', 'Vasanth Nagar',
       'Jeevan Bhima Nagar', 'Old Madras Road', 'Commercial Street',
       'Koramangala 6th Block', 'Majestic', 'Langford Town',
       'Koramangala 7th Block', 'Brookefield', 'Whitefield',
       'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield',
       'Koramangala 2nd Block', 'Koramangala 3rd Block',
       'Koramangala 4th Block', 'Koramangala', 'Bommanahalli',
       'Hosur Road', 'Seshadripuram', 'Electronic City', 'Banaswadi',
       'North Bangalore', 'RT Nagar', 'Kammanahalli', 'Hennur',
       'HBR Layout', 'Kalyan Nagar', 'Thippasandra', 'CV Raman Nagar',
       'Kaggadasapura', 'Kanakapura Road', 'Nagawara', 'Rammurthy Nagar',
       'Sankey Road', 'Central Bangalore', 'Malleshwaram',
       'Sadashiv Nagar', 'Basaveshwara Nagar', 'Rajajinagar',
       'New BEL Road', 'West Bangalore', 'Yeshwantpur', 'Sanjay Nagar',
       'Sahakara Nagar', 'Jalahalli', 'Yelahanka', 'Magadi Road',
       'KR Puram']
location.sort()
print("Search Restaurants according to their name")
@interact
def show_articles_more_than(Restaurant_Name=''):
    return zomato[zomato['name'].str.contains(Restaurant_Name)]


# SEARCH RESTAURENTS ACCORDING TO YOUR REQUIRMENTS:
# * Loaction- Choose any location where you want to go.
# * Type- Choose the type if restaurent you want to go weheter cafe,pub ect.
# * Max_cost_for_two= Select the max cost for two people and it will show restuarents having that much cost or below it.

# In[ ]:


@interact 
def show_Restaurants_according_to_search(Location=location,
                                         Restaurant_Type=['Buffet', 
                                             'Cafes',
                                             'Delivery',
                                             'Desserts',
                                             'Dine-out',
                                             'Drinks & nightlife',
                                             'Pubs and bars'],
                            Min_Rating=(0,5,0.1),
                            Max_Cost_For_Two_People=(100,5000,50)):
    print("")
    return zomato[ (zomato['rate'] > Min_Rating) 
                &(zomato['listed_in(type)'] == Restaurant_Type) 
                &(zomato['location'] == Location) 
                & (zomato['approx_cost(for two people)'] < Max_Cost_For_Two_People)]


# ..

# ANALYSIS OF RESTUARENTS BASED ON THERE ONLINE DELIVERY

# In[ ]:


print('number of restaurents with online delivery')
(zomato.online_order == 'Yes').sum()


# In[ ]:


print('Number of restaurents which does not deliver online')
(zomato.online_order == 'No').sum()


# In[ ]:


zomato.name.count()


# In[ ]:


sns.countplot(x=zomato['online_order'])
plt.title('Restuarents delivering online or not')


# In[ ]:


sns.countplot(x=zomato['online_order'],hue=zomato['listed_in(type)'],)
fig = plt.gcf()  # here gcf means 'GET THE CURRENT FIGURE'
fig.set_size_inches(10,10)
plt.title('Type of restaurents delivering online or not')


# ANALYSIS OF RESTAURENTS BASED ON THEIR TABLE BOOKING FACILITY

# In[ ]:


print("Number of restaurents with table booking facility")
(zomato.book_table == 'Yes').sum()


# In[ ]:


print('Number of restaurents without table facility')
(zomato.book_table == 'No').sum()


# In[ ]:


sns.countplot(x=zomato['book_table'])
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.title('Restaurents providing table booking facility')


# In[ ]:


sns.countplot(x=zomato['book_table'],hue=zomato['listed_in(type)'])
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title("Type of restaurents providing table booking facility")
plt.show()


# ANALYSIS OF RESTAURENTS BASED ON THERE RATINGS

# In[ ]:


print('Restaurents on there unique ratings')
zomato.rate.unique()


# In[ ]:


print("Number of restaurents rating between 1.5 and 2")
((zomato.rate>=1.5) & (zomato.rate<2)).sum()


# In[ ]:


print('number of restaurents rating between 2 and 2.5')
((zomato.rate>=2)&(zomato.rate<2.5)).sum()


# In[ ]:


print('number of restaurents rating between 2.5 and 3')
((zomato.rate>=2.5) & (zomato.rate<3)).sum()


# In[ ]:


print('number of restaurents rating between 3 and 3.5')
((zomato.rate>=3)&(zomato.rate<3.5)).sum()


# In[ ]:


print('number of restaurents rating between 3.5 and 4')
((zomato.rate>=3.5)&(zomato.rate<4)).sum()


# In[ ]:


print('number of restaurents rating between 4 and 4.5')
((zomato.rate>=4)&(zomato.rate<4.5)).sum()


# In[ ]:


print('number of restaurents rating between 4.5 and 5')
((zomato.rate>=4.5)&(zomato.rate<=5)).sum()


# In[ ]:


slices=[((zomato.rate>=1.5) & (zomato.rate<2)).sum(),
        ((zomato.rate>=2) & (zomato.rate<2.5)).sum(),
        ((zomato.rate>=2.5) & (zomato.rate<3)).sum(),
        ((zomato.rate>=3.0) & (zomato.rate<3.5)).sum(),
        ((zomato.rate>=3.5) & (zomato.rate<4)).sum(),
        ((zomato.rate>=4) & (zomato.rate<4.5)).sum(),
        ((zomato.rate>=4.5) & (zomato.rate<5)).sum()
       ]
labels=['1.5-2','2-2.5','2.5-3','3-3.5','3.5-4','4-4.5','4.5-5']
colors = ['Red','blue','Green','black','orange','pink']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their ratings", bbox={'facecolor':'2', 'pad':5})

fig.set_size_inches(10,10)
plt.show()


# Analysis of Restaurants based on their online order and how rating is related to it
# 

# In[ ]:


plt.figure(figsize=(20,10))
Aa=sns.countplot(x='rate',hue='book_table',data=zomato)
plt.title('Rating of restaurents VS book_table')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
Aa=sns.countplot(x='rate',hue='online_order',data=zomato)
plt.title('Rating of restaurents VS book_table')
plt.show()


# ANALYSIS OF RESTAURENTS BASED ON THEIR LOCATION

# In[ ]:


print("All unique locations of restaurents in bangalore")
zomato.location.unique()


# In[ ]:


print("count of restaurents at unique locations")
locationCount=zomato['location'].value_counts().sort_values(ascending=True)
locationCount


# In[ ]:


# Now lets check the location where there is maximum number of restaurents.
print('Maximum number of restaurents is at:')
count_max=max(locationCount)
for x,y in locationCount.items():
    if(y==count_max):
        print(x)


# So, In Koramangala 5th Block there are maximum number of restaurents.

# In[ ]:


# now, lets find the location where there is minimum number of restaurents.
print('Minimum number of restaurents at :')
count_min=min(locationCount)
for x,y in locationCount.items():
    if(y==count_min):
        print(x)


# Hence, KR Puram has the minimun number of restaurents.

# In[ ]:


fig=plt.figure(figsize=(20,40))
locationCount.plot(kind="barh",fontsize=20)
plt.ylabel("Location names",fontsize=50,color="red",fontweight='bold')
plt.title("LOCATION VS RESTAURANT COUNT GRAPH",fontsize=40,color="BLACK")
plt.show()


# ANALYSIS OF RESTAURENT BASED ON THEIR DINNING TYPE

# In[ ]:


print('all different dinning type of restaurents')
zomato['listed_in(type)'].unique()


# In[ ]:


print('Count of all different dinning type restaurents')
restaurantTypeCount=zomato['listed_in(type)'].value_counts().sort_values(ascending=True)
restaurantTypeCount


# In[ ]:


slices=[restaurantTypeCount[0],
        restaurantTypeCount[1],
        restaurantTypeCount[2],
        restaurantTypeCount[3],
        restaurantTypeCount[4],
        restaurantTypeCount[5],
        restaurantTypeCount[6]]
labels=['Pubs and bars','Buffet','Drinks & nightlife','Cafes','Desserts','Dine-out','Delivery ']
colors = ['Blue','green','pink','yellow','red','brown','orange']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their Type", bbox={'facecolor':'2', 'pad':5})

fig.set_size_inches(12,12)
plt.show()


# ANALYSIS ON THE BASIS OF FOOD TYPE

# Now, lets try to get all the North Indian food serving restaurents

# In[ ]:


NorthIndianFoodRestaurants = zomato[zomato['cuisines'].str.contains('North Indian', case=False, regex=True,na=False)]
NorthIndianFoodRestaurants.head()


# lets try to get all the South Indain food serving restaurents

# In[ ]:


SouthIndianFoodRestaurants = zomato[zomato['cuisines'].str.contains('South Indian', case=False, regex=True,na=False)]
SouthIndianFoodRestaurants.head()


# lets find all the chinese food serving restaurent

# In[ ]:


ChineseFoodRestaurants = zomato[zomato['cuisines'].str.contains('Chinese|Momos', case=False, regex=True,na=False)]
ChineseFoodRestaurants.head()


# lets find Italian food serving restaurents

# In[ ]:


ItalianFoodRestaurants = zomato[zomato['cuisines'].str.contains('Italian|Pizza', case=False, regex=True,na=False)]
ItalianFoodRestaurants.head()


# lets find Mexican food serving restaurents

# In[ ]:


MexicanFoodRestaurants = zomato[zomato['cuisines'].str.contains('Mexican', case=False, regex=True,na=False)]
MexicanFoodRestaurants.head()


# lets get all the american food serving restaurents

# In[ ]:


AmericanFoodRestaurants = zomato[zomato['cuisines'].str.contains('american|Burger', case=False, regex=True,na=False)]
AmericanFoodRestaurants.head()


# ANALYSIS OF BIGGEST FOOD CHAINS OF BANGALORE

# In[ ]:


branches = zomato.groupby(['name']).size().to_frame('count').reset_index().sort_values(['count'],ascending=False)
ax = sns.barplot(x='name', y='count', data=branches[:12])
plt.xlabel('')
plt.ylabel('Branches')
plt.title('Food chains and their counts')

fig = plt.gcf()
fig.set_size_inches(25,15)


# * Onesta has highest number of chains in Bangalore
