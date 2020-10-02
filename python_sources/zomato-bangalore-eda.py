#!/usr/bin/env python
# coding: utf-8

# # Loading the libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
from geopy.extra.rate_limiter import RateLimiter
import folium
from collections import Counter


# # Custom functions

# In[ ]:


def draw_pie_graph_for_this_data(data):
    """
    It is a function which draws the pie graph for a variable.
    'data' will be a series(not a dataframe) having that particular variable data only.
    """
    labels = data.value_counts().index
    values = data.value_counts().values
    explode = [0.1, 0]
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, explode=explode,
            autopct='%1.1f%%', startangle=90)
    # autopct is used for labelling inside pie wedges, startangle rotates pie counterclockwise
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()
    

#Lets write a custom function to draw horizonatal bar graph for max values.
def draw_bar_graph_for_this_data(data):
    """
    It is a function which draws the pie graph for a variable.
    'data' will be a series(not a dataframe) having that particular variable data only.
    """
    plt.figure(figsize=(10, 7))
    top_thirty = data.value_counts()[:30]
    sns.barplot(x=top_thirty,
            y=top_thirty.index)
    plt.title("Number of restaurants in cities")
    plt.xlabel("Count")
    plt.ylabel("Name")
    
    
#Lets write a custom function to split comma seperated data to a list
def split_data(arr):
    """
    It is a function which accepts the array from of a series(arr) and then split them seperately.
    It then returns a list of tuples of favourite items and there count. 
    """
    list_of_item = []
    for item in tqdm(arr):
        curr_item = item.split(',')
        curr_item = [var.strip() for var in curr_item]
        list_of_item.extend(curr_item)
    count_dict = Counter(list_of_item)
    list_of_top_fav = count_dict.most_common(5)
    return list_of_top_fav


#Lets write a custom function to split favourite items and there counts
def seperate_name_and_count(list_of_top_fav):
    """
    It is a function which accepts the array from of a series(arr) and then split them seperately.
    """
    list_of_items = []
    respective_count_of_items = []
    for ele in list_of_top_fav:
        list_of_items.append(ele[0])
        respective_count_of_items.append(ele[1])
    return (list_of_items,respective_count_of_items)

#Lets write a custom function which will simply plot bargraph for two variables x and y
def direct_bar_graph(respective_count,list_of_item):
    """
    This function takes x and y two series as input, one of which has to be numeric. And plot graph for it.
    """
    plt.figure(figsize=(10, 7))
    sns.barplot(x=respective_count,y=list_of_item)
    plt.title("Number of favourite dishes")
    plt.xlabel("Count")
    plt.ylabel("Name")


# # Loading the data

# In[ ]:


# Load the data
zomato = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")


# In[ ]:


print("The shape of the data is ==>", zomato.shape)
print("\n")
print("The columns in data ==>", zomato.columns)


# In[ ]:


zomato.head()


# In[ ]:


zomato.describe()


# In[ ]:


zomato.info()


# In[ ]:


zomato.isnull().sum()


# In[ ]:


dummy_zomato = zomato
dummy_zomato = dummy_zomato.dropna()
(zomato.shape[0]-dummy_zomato.shape[0])/(zomato.shape[0])*100


# 
# <ul>
#     <li><b>There are total 51717 rows and 17 columns in data</b></li>
#     <li><b>Column votes is the only available column with integer values</b></li>
#     <li><b>rate,phone,location,rest_type,dish_liked,cuisines,approx_cost these columns in particular have missing values</b></li>
#     <li><b>More than 50% data in dataframe is missing. You have to use data imputation</b></li>
#     <li><b>First thing I am going to do is pick up various columns and draw EDA to get different inferences out of them</b></li>
# </ul>

# # Inspecting various columns

# ## 'online_order' column

# In[ ]:


online_order = zomato['online_order']


# In[ ]:


# Above data is binary, Plot pie chart for above data
draw_pie_graph_for_this_data(online_order)


# <ul><li><b>Almost 60% of restaurants accept online orders in the city</b></li></ul>

# ## 'name' column

# In[ ]:


names_of_restaurant = zomato[['name']]


# In[ ]:


#Lets draw bar graph for maximum number of restaurants top thirty
draw_bar_graph_for_this_data(names_of_restaurant["name"])


# In[ ]:


# Let's check for the rating of CCD
zomato_ccd = zomato[zomato['name'] == 'Cafe Coffee Day']


# In[ ]:


zomato_ccd = zomato_ccd[zomato_ccd['rate'] != 'New']
zomato_ccd = zomato_ccd[zomato_ccd['rate'] != '-']
zomato_ccd = zomato_ccd.dropna()


# In[ ]:


zomato_ccd['rate'] = zomato_ccd['rate'].apply(lambda x: float(x.split('/')[0]))


# In[ ]:


zomato_ccd['rate'].describe()['mean']


# <b>Although CCD has maximum number of outlets in city, yet it's average rating is only 3.3</b>

# ## 'rate' column

# In[ ]:


def clean_rate_data(zomato_rate_data):
    #Using square bracket two times returns a df.
    zomato_rate_data = zomato_rate_data[zomato_rate_data['rate']!='NEW']
    zomato_rate_data = zomato_rate_data[zomato_rate_data['rate']!='-']
    zomato_rate_data = zomato_rate_data.dropna()
    zomato_rate_data['rate'] = zomato_rate_data['rate'].apply(lambda x : float(x.split('/')[0])) 
    return zomato_rate_data


# In[ ]:


# Filter rate data and call cleaning function
zomato_rate_data = zomato[['rate']]
zomato_rate_data = clean_rate_data(zomato_rate_data)


# In[ ]:


zomato_rate_data.head()


# In[ ]:


sns.FacetGrid(zomato_rate_data, height=5, hue="rate").map(
    sns.distplot, "rate").add_legend()
plt.show()


# <b>Maximum restaurants have rating higher than 3.5 and upto 4, also restaurants with rating higher than 4.5 are very rare.</b>

# ## 'book_table' column

# In[ ]:


book_table_data = zomato["book_table"]


# In[ ]:


# It is a binary data
# Plot pie chart for above data
draw_pie_graph_for_this_data(book_table_data)


# <b>More than 85% restaurants provide the facility of booking table</b>

# ## 'votes' column

# In[ ]:


votes_data = zomato[['votes']]


# In[ ]:


"""A thing to note is that votes standalone can't provide any information. 
Since voting is not co-related to any other data. It's better to utilise it with any other data.
Like at a later stage you can actually visualize  relation between votes and rating."""


# ## 'phone' column

# In[ ]:


"""In particular phone can't provide us any kind of information , hence we will drop it from database
at a later stage"""


# ## 'location' column

# In[ ]:


#So what now? . 
#Think what you can extract from locations of restaurant ?
location_data = zomato[['location']]


# In[ ]:


location_data.head()


# In[ ]:


#Let's write code to get the latitude and longitude of a particular location
#This is going to take a lot of time.
#Since google API only allows you 2500 requests a day.
# geolocator = Nominatim()
# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
# zomato["latitude"]= zomato['location'].apply(geocode, timeout=20).apply(lambda x: (x.latitude))
# zomato["longitude"]= zomato['location'].apply(geolocator.geocode, timeout=20).apply(lambda x: (x.longitude))

#Problem here is , it is way too difficult to get the coordinates of all the unique cities in bangalore.
#Will work around it later.


# In[ ]:


#Let's draw bar graph for number of restaurant with respect to cities.
draw_bar_graph_for_this_data(location_data["location"])


# In[ ]:


#We can mark top 5 cities with max restaurants in map
top_ten_cities = location_data["location"].value_counts()[:10]
top_ten_cities = top_ten_cities.index.values


# In[ ]:


geolocator = Nominatim()
list_of_coordinates = []
for city in top_ten_cities:
    loc = geolocator.geocode("Bangalore "+ city, timeout=20)
    geo = (loc.latitude, loc.longitude)
    list_of_coordinates.append(geo)
print(list_of_coordinates)


# In[ ]:


map_of_bangalore = folium.Map(location = [12.9716, 77.5946], zoom_start=10)
map_of_bangalore


# In[ ]:


for index in range(len(list_of_coordinates)):
    folium.Marker(list_of_coordinates[index], tooltip = top_ten_cities[index]).add_to(map_of_bangalore)
map_of_bangalore


# <b>Maximum restuarants hence are available in central bangalore</b>

# ## 'rest_type' column

# In[ ]:


rest_type_data = zomato[["rest_type"]]
rest_type_data.head()


# In[ ]:


#Let's draw bar graph for top thirty type of restaurants with respect to count.
draw_bar_graph_for_this_data(rest_type_data["rest_type"])


# <b>Clearly quick bites restaurants wins the race.</b>
# <b>Hence culture of bangalore promotes quick bites restaurants.</b>

# ## 'dish_liked' column

# In[ ]:


dish_liked_data = zomato[["dish_liked"]]


# In[ ]:


dish_liked_data = dish_liked_data.dropna()


# In[ ]:


dish_liked_data.shape


# In[ ]:


array_of_dishes = np.array(dish_liked_data["dish_liked"])


# In[ ]:


#Function call
list_of_top_fav_dishes = split_data(array_of_dishes)


# In[ ]:


#Function call
list_of_items,respective_count_of_items = seperate_name_and_count(list_of_top_fav_dishes)


# In[ ]:


direct_bar_graph(respective_count_of_items,list_of_items)


# In[ ]:


list_of_top_fav_dishes


# ## 'cuisines' column

# In[ ]:


cuisines_data = zomato[["cuisines"]]


# In[ ]:


cuisines_data = cuisines_data.dropna()


# In[ ]:


array_of_cuisines = np.array(cuisines_data["cuisines"])


# In[ ]:


#Function call
list_of_top_fav_cuisines = split_data(array_of_cuisines)


# In[ ]:


#Function call
list_of_items,respective_count_of_items = seperate_name_and_count(list_of_top_fav_cuisines)


# In[ ]:


direct_bar_graph(respective_count_of_items,list_of_items)


# In[ ]:


list_of_top_fav_cuisines


# ## 'approx_cost(for two people)' column

# In[ ]:


cost_data = zomato[['approx_cost(for two people)']]


# In[ ]:


cost_data['approx_cost(for two people)'] = cost_data['approx_cost(for two people)'].apply(lambda x : str(x).replace(",", ""))


# In[ ]:


plt.figure(figsize=(10, 7))
top_thirty = cost_data['approx_cost(for two people)'].value_counts()[:5]
print(top_thirty)
print(top_thirty.index)
sns.barplot(x=top_thirty.index,y=top_thirty)
plt.title("Rate analysis")
plt.xlabel("Price")
plt.ylabel("Count")


# <b>Above analysis is informing us that maximum restaurants in town offers food at 300 cost for two people</b> 

# ## 'reviews_list' column

# In[ ]:


#Since it involves NLP portion, We will see it later.


# ## 'menu_item' column

# In[ ]:


menu_item_data = zomato[['menu_item']]


# In[ ]:


array_of_menus = np.array(menu_item_data["menu_item"])


# In[ ]:


menu_list = []
for menu in tqdm(array_of_menus):
    curr = menu.split(',')
    curr = [item.strip() for item in curr]
    menu_list.extend(curr)


# In[ ]:


count_menu_dict = Counter(menu_list)


# In[ ]:


list_of_top_five_menus = count_menu_dict.most_common(6)[1:]


# In[ ]:


list_of_menus = []
respective_count_of_menus = []
for ele in list_of_top_five_menus:
    list_of_menus.append(ele[0])
    respective_count_of_menus.append(ele[1])


# In[ ]:


plt.figure(figsize=(10, 7))
sns.barplot(x=respective_count_of_menus,y=list_of_menus)
plt.title("Number of favourite cuisine")
plt.xlabel("Count")
plt.ylabel("Name")


# ## 'listed_in(type)' column

# In[ ]:


type_data = zomato[['listed_in(type)']]


# In[ ]:


type_data["listed_in(type)"].unique()


# In[ ]:


direct_bar_graph(type_data["listed_in(type)"].value_counts(),type_data["listed_in(type)"].value_counts().index)


# <ol>
#     <li>Delivery and dine-out stands tall.</li>
#     <li>It is strange to believe that pubs and bars culture is not that popular </li>
# </ol>

# ## 'listed_in(city)' column

# In[ ]:


city_data = zomato[['listed_in(city)']]


# In[ ]:


city_data['listed_in(city)'].unique()


# In[ ]:


direct_bar_graph(city_data["listed_in(city)"].value_counts(),city_data["listed_in(city)"].value_counts().index)


# <b> Since the problem with locations is that it is hard to find the geolocation due to limited calls through google api hence it would be better to encode the cities on some priority basis </b>

# # Inferences as of now

# <ol>
#     <li> Food culture in bangalore prommotes restaurants accepting online orders</li>
#     
#     <li> CCD, Onesta, JustBake and others are few top restaurants chains in bangalore according to the count</li>
#     
#     <li>Restaurants with rating higher than 4.5 are very rare. Hence one thing is very clear , that it is impossible to win this game statisitically. You have to think of other factors too, (quality and type of chefs for instance) .Otherwise you can play safe and be in a rating of 3.5-4.0 which is safe and decent</li>
#     
#     <li>85% restaurants provide the facility of booking the table. Obviously staying in game needs that facility.</li>
#     
#     <li>Central bangalore is the prime location to open a restaurant. Although keep in mind that central bangalore is well explored. And regions otherwise are yet there to gain control.</li>
#     
#     <li>Quick bites and casual dining is the most popular type of restaurants</li>
#     
#     <li>No surprises that the demand of fast food like pasta and burger is top notch.  Hence every new restaurant must either serve top quality of these fast food, or else serve variety. </li>
#     
#     <li>Overall in bangalore north indian and chinese is the top favourite type of cuisines.</li>
#     
#     <li>300-400 is the best price range offered for two people by restaurants.Cost hence must be a feature taken care of. It may influence the business.</li>
#     
#     <li>Make sure rice is in your menu. That too of every type, veg and non veg.</li>
#     
#     <li>Delivery and dine outs are the most favourite trend . Pubs and bars dont have that much popularity. One reason might be the busy life of peoples in the city.</li>
#     
#     <li>To classify restaurants I believe it should be wise to divide rating in between a scale of 1-5 , then it will be a better scenario of classification</li>
# </ol>
