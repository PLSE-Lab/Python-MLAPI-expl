#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:




data = pd.read_csv('/kaggle/input/Datafiniti_Pizza_Restaurants_and_the_Pizza_They_Sell_May19.csv')


# In[ ]:


# lets look at the data
pd.set_option('display.max_columns',30)
pd.set_option('display.max_rows',100)
data.head()


# In[ ]:


# data dimensions
#  data columns

print(data.shape)
print(data.columns)


# In[ ]:


# Min and Max prices of menu and range too

prices = data[['menus.amountMax','menus.amountMin','priceRangeMax','priceRangeMin']]


# In[ ]:


prices.describe()


# In[ ]:


# Max :-  meuns.amountmax price is 1395 $ and max :-  menus.amountmin is 243 $
#  Lets investigate - menus max price 1395,  we can that 75% of menus.amountmax price is around  18 $  


# In[ ]:


data[data['menus.amountMax'] > 500]


# In[ ]:


# interesting Restaurant Rocco's in Pasco city might have a dish which is priced about 1395 $


# In[ ]:


# Lets look at the categories and primaryCategories column
""" 1. Whats the most and least popular in  primary category"""
    


# In[ ]:


pc = data['primaryCategories'].value_counts().reset_index().rename(columns ={'index' :'Primary Categories','primaryCategories':'Values'})
pc


# In[ ]:


import seaborn 
import matplotlib.pyplot as plt


# In[ ]:


y = seaborn.barplot(x = pc['Values'],y=pc['Primary Categories'],data=pc)
for index, value in enumerate(pc['Values']):
    y.text(value , index + .15, str(value), fontweight='bold')


# In[ ]:


# lets see the average prices of Management of Companies & Enterprises primary category
data[(data['primaryCategories'] == 'Management of Companies & Enterprises') |
     (data['primaryCategories'] == 'Accommodation & Food Services,Management of Companies & Enterprises') ]\
[['city','menus.amountMax','menus.amountMin','name','categories']]\
.groupby(['city','name','categories']).mean().reset_index()


# In[ ]:


""" From the above we can see how primary category (Management of Companies & Enterprises)
prices are changing from city to city further all the Restaurant places are coming under
Companies & Enterprises business establishment """


# In[ ]:


# Ignoring menus.amountprice 0 


# In[ ]:


data = data[(data['menus.amountMax'] != 0) & (data['menus.amountMin'] != 0)]


# In[ ]:


data.shape


# In[ ]:


# Most frequent province in this dataset


# In[ ]:


data['province'].value_counts().head(10)


# In[ ]:


g = seaborn.countplot(y = 'province',data = data , palette='Greens_d',
                  order=data.province.value_counts().iloc[:10].index)

for index, value in enumerate(list(data.province.value_counts().iloc[:10].values)):
    g.text(value , index + .15, str(value), fontweight='bold')


# In[ ]:


# From this dataset we can see that NY,CA,PA have lot more different pizza's


# In[ ]:


# New York Provinces has following cities.
cities_ny = list(np.unique(data[data['province'] == 'NY']['city']))
print(cities_ny[:5])
print("Total No of cities in NY - {} ".format(len(cities_ny)))


# In[ ]:


# Top 10 cities by count
g = seaborn.countplot(y = 'city',data = data , palette='Reds_d',
                  order=data.city.value_counts().iloc[:10].index)
for index, value in enumerate(list(data.city.value_counts().iloc[:10].values)):
    g.text(value , index + .15, str(value), fontweight='bold')


# In[ ]:


print('Total No unique restaurants in the dataset :- {} '.format(len(np.unique(data['id']))))


# In[ ]:


data['name'] = data['name'].replace("Papa Murphys","Papa Murphy's")
data['name'] = data['name'].replace("Papa Johns Pizza","Papa John's Pizza")
data['name'] = data['name'].replace("Zpizza","zpizza")


# In[ ]:


data['name'].value_counts().head(15).reset_index().rename(columns ={'index':'Name','name':'Values'}).plot('Name','Values',kind ='barh')


# In[ ]:


# Papa Murphy's followed by California Pizza Kitchen are present in different cities so they top the list


# In[ ]:


data = data[data['menus.amountMax'] != 1395]


# In[ ]:


# Starting Min least expensive pizza 


# In[ ]:


data[['city','name','menus.name','menus.amountMin']][data['menus.amountMin'] == data['menus.amountMin'].min()]


# In[ ]:


#  starting max expensive pizza 

data[['city','name','menus.name','menus.amountMin']][data['menus.amountMin'] == data['menus.amountMin'].max()]


# In[ ]:


#  ending max expensive pizza 

data[['name','menus.name','menus.amountMax']][data['menus.amountMax'] == data['menus.amountMax'].max()]


# In[ ]:


#  ending Min expensive pizza 

data[['name','menus.name','menus.amountMax']][data['menus.amountMax'] == data['menus.amountMax'].min()]


# In[ ]:


# price Difference 

data['PriceDiff'] = data['menus.amountMax'] - data['menus.amountMin']


# In[ ]:


#   Min diff expensive pizza 

data[['name','menus.name','PriceDiff']][data['PriceDiff'] == data['PriceDiff'].max()]


# In[ ]:


# avg price per pizza in cites 
avg_price_city = data[data['menus.amountMin'] == data['menus.amountMax']].groupby('city').mean()['menus.amountMin'].reset_index().rename(columns = {'menus.amountMin':'Avg Price'})
 


# In[ ]:


city_count = data[data['menus.amountMin'] == data['menus.amountMax']]['city'].value_counts().reset_index().rename(columns
                                                                                                                 ={'city':'values','index':'city'})


# In[ ]:


avg_city_price = pd.merge(avg_price_city,city_count,on='city').sort_values(by='values',ascending =False)


# In[ ]:


avg_city_price.head(10)


# In[ ]:



seaborn.barplot(x = 'Avg Price',y='city',data= avg_city_price.head(8),color='lightblue')
plt.title('City vs  Avg Pizza Price')


# In[ ]:


import re
large_pizza =[]
for i in data['menus.name']:
    if re.match('Large',i) != None:
        large_pizza.append('yes')
    else:
        large_pizza.append('No')


# In[ ]:


data['large_pizza'] = large_pizza


# In[ ]:


large_pizz = data[data['large_pizza'] == 'yes']


# In[ ]:


# Avg price of large pizza in cites 
large_pizza = large_pizz [large_pizz['menus.amountMin'] == large_pizz['menus.amountMax']].groupby('city').mean()['menus.amountMin'].reset_index().rename(columns = {'menus.amountMin':'Avg Price'})
city_count1 = large_pizz [large_pizz['menus.amountMin'] == large_pizz['menus.amountMax']]['city'].value_counts().reset_index().rename(columns  ={'city':'values','index':'city'})
avg_city_price_large_pizza = pd.merge(large_pizza,city_count1,on='city').sort_values(by='values',ascending =False)
                                                                                                                                      


# In[ ]:


avg_city_price_large_pizza.head(7)


# In[ ]:


# Avg price distribution per city

seaborn.distplot(avg_price_city['Avg Price'])


# In[ ]:


# Top 5 freq Pizza 
data['menus.name'].value_counts().head(10)


# In[ ]:


# Lets see where we get Veggie Pizza by map


# In[ ]:


veggie_pizza = data[data['menus.name'] == 'Veggie Pizza']


# In[ ]:


veggie_pizza.drop_duplicates('id',inplace = True)


# In[ ]:


veggie_pizza.shape


# In[ ]:


import plotly.graph_objects as go


# In[ ]:


fig = go.Figure(data=go.Scattergeo(
        lon = veggie_pizza['longitude'],
        lat = veggie_pizza['latitude'],
        text = veggie_pizza['name'] + ' :-' + veggie_pizza['province'],
        mode = 'markers',
    
        marker_color = 'green',
        ))

fig.update_layout(
        title = 'Viggie Pizza Places',
        geo_scope='usa',
    )
fig.show()


# In[ ]:


# Analysis on date
date = pd.DatetimeIndex(data['dateAdded'])


# In[ ]:


data['day_name'] = date.day_name()


# In[ ]:


data['date'] = date.date
data['year'] = date.year


# In[ ]:


# On which day the most pizza's were added in 2016
data_2016 = data[data['year'] == 2016]
data_2017 = data[data['year'] == 2017]


# In[ ]:


data_2016['date'].value_counts().head(3)


# In[ ]:





# In[ ]:




