#!/usr/bin/env python
# coding: utf-8

# # Logistics and Travelling Santa
# 
# This started with an exploration of how to solve the problem as-is.
# But I became interested in how to make this more "real world" and consider many practical limitations and how to address those considerations in the scope of this problem.
# 
# #### Let's first import some modules that we'll be using

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import isprime
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib
import math
import doctest
from itertools import permutations
import seaborn as sns
import time
import os

files_directory_path = "../input/"


# #### Let's see what the data looks like

# In[ ]:


# Scatter Plot
def plot_scatter(x,y,title):
    fig = plt.figure(figsize=(15,15))
    plt.scatter(x, y, marker='o', s=1, color='b', linewidths=0)
    plt.scatter(x[0],y[0], marker='o', color='r', linewidths=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()
    
    
cities_df = pd.read_csv(files_directory_path + "cities.csv") 

# apply a function to add a column to our DataFrame indicating whether the city is a prime or not
cities_df['is_prime'] = cities_df.CityId.apply(isprime)
prime_cities_count = np.sum(cities_df['is_prime'])

# Detailing cities
print("The North Pole is located at: {:,.2f}, {:,.2f}".format(cities_df.X[0],cities_df.Y[0]))
print("There are {:d} cities".format(len(cities_df.index) - 1)) 
print("There are {:d} prime cities".format(prime_cities_count))
print("There are {:d} that are not prime".format(len(cities_df.index) - 1 - prime_cities_count))
print("First 10 cities:")
print(cities_df.head(10))

# Plot
plot_scatter(cities_df.X, cities_df.Y, "Scatter Plot with North Pole Highlighted")


# #### Define some functions to help us compute the score for a given route

# In[ ]:


# The function to get the distance between the cities.
def distance(x1, y1, x2, y2, prev_is_prime, is_10th):
    # Every 10th step is 10% more lengthy unless coming from a prime CityId.
    cost_factor = 1.1 if is_10th and not prev_is_prime else 1.0
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * cost_factor

# The function to calculate score.
# The beginning and end of the route must be City'0'.
def calculate_score(route, cities_df):
    cities_df_dict = cities_df.to_dict()

    sum_distance = 0
    prev_x, prev_y = cities_df_dict['X'][0], cities_df_dict['Y'][0]
    prev_is_prime = False

    for i, city in enumerate(route):
        x, y = cities_df_dict['X'][city], cities_df_dict['Y'][city]
        is_prime = cities_df_dict['is_prime'][city]

        sum_distance += distance(prev_x, prev_y, x, y, prev_is_prime, i % 10 == 0)
        prev_x, prev_y = x, y
        prev_is_prime = is_prime

    return sum_distance


# #### Let's create a really simple route that passes thru each city and compute the score for that route
# 

# In[ ]:


route = cities_df['CityId'].tolist()
route.append(route[0])
score = calculate_score(route, cities_df)
print("Score for route where we pass thru cities in their inital order: {:,.2f}".format(score))


# #### Polygon Route
# The shortest route should be one where the path does not cross itself.
# Thus let's try to create a polygon where each node is a city.
# To do this, we'll add a column to the data being the arctan2 of where that city is in comparison to the north pole.

# In[ ]:


def add_arctan2_to_data(cities_df, use_np_as_center):
    x = cities_df["X"]
    y = cities_df["Y"]
    if use_np_as_center:
        refX = x[0]
        refY = y[0]
    else:
        refX = int(sum(x for x in x) / float(len(cities_df.index)))
        refY = int(sum(y for y in y) / float(len(cities_df.index)))

    def angle(x, y):
        relX,relY = x - refX, y - refY
        return np.arctan2(relX, relY) # takes care of quadrant calculation

    # use vectorization, the angle function uses numpy internally to allow for list comprehension
    cities_df['arctan2'] = angle(cities_df['X'],cities_df['Y'])

add_arctan2_to_data(cities_df, use_np_as_center=True)
print(cities_df.head(10))


# #### Arctan2
# 
# Now we will use the arctan2 value to "sweep out" a polygon by traversing the cities by increasing arctan2.

# In[ ]:


# Create a simple polygon
def create_polygon_route(cities_df,use_np_as_center):
    sorted_cities = cities_df.sort_values(by='arctan2')

    # extract the "route" as the city IDs
    route = sorted_cities['CityId'].tolist()
    route.append(route[0])
    return route
    
# Path plotting
def plot_path(path,cities,title):
    coords = cities[['X', 'Y']].values
    ordered_coords = coords[np.array(path)]
    codes = [Path.MOVETO] * len(ordered_coords)
    path = Path(ordered_coords, codes)
    
    fig = plt.figure(figsize=(15,15))
    plt.rcParams['agg.path.chunksize'] = 1000
    ax = fig.add_subplot(111)
    xs, ys = zip(*ordered_coords)
    ax.plot(xs, ys,  lw=1., ms=10)
    
    north_pole = cities[cities.CityId==0]
    plt.scatter(north_pole.X, north_pole.Y, c='red', s=5)
    
    plt.title(title)
    plt.show()

    
route = create_polygon_route(cities_df,use_np_as_center=True)
score = calculate_score(route, cities_df)
print("Score for route where we create simple polygon: {:,.2f}".format(score))
plot_path(route,cities_df,"Simple Polygon - Score: {:,.2f} Path Length: {:d}".format(score,len(route)))


# #### >50% Improvement
# 
# We see a more than 50% improvement but it still doesn't look good.  In this case we have a big variance in distance between consecutive points as they vary in distance from the center (north pole).
# 
# Let's try to divide this route into "bands".  To do this, let's add a distance from the center (north pole) to our data.

# In[ ]:


def add_distance_from_center_to_data(cities_df, use_np_as_center):
    x = cities_df["X"]
    y = cities_df["Y"]
    if use_np_as_center:
        refX = x[0]
        refY = y[0]
    else:
        refX = int(sum(x for x in x) / float(len(cities_df.index)))
        refY = int(sum(y for y in y) / float(len(cities_df.index)))

    def angle(x, y):
        relX,relY = x - refX, y - refY
        return np.arctan2(relX, relY) # takes care of quadrant calculation

    # use vectorization to compute the values for this column.
    cities_df['dist_from_center'] = np.sqrt((refX - cities_df['X']) ** 2 + (refY - cities_df['Y']) ** 2)

add_distance_from_center_to_data(cities_df, use_np_as_center=True)
print(cities_df.head(10))


# #### Band Size?
# 
# Now we have the distance.  But what size should we make our "bands"?
# Let's do some exploration to find a good value.

# In[ ]:


# slice and assemble the path so it start at City 0 and ends with City 0
def order_path(path):
    # find index where 0 is
    np_index = path.index(0)
    reordered = path[np_index:] + path[:np_index]
    reordered.append(reordered[0])
    return reordered

# function to assemble a route using a "band_index"
def route_from_bands(cities_df):
    sorted_by_band_and_arctan2 = cities_df.sort_values(by=['band_index','arctan2'])
    
    b_min = sorted_by_band_and_arctan2['band_index'].min()
    b_max = sorted_by_band_and_arctan2['band_index'].max()
    route = []
    for b in range(b_min, b_max + 1):
        # select rows with current b
        band = sorted_by_band_and_arctan2.loc[cities_df['band_index'] == b]
        band_route = band['CityId'].tolist()
        
        # we want to alternate the direction of the band so
        # the next band start where the last band ended
        if (b % 2):
            band_route.reverse()
        route.extend(band_route)
    
    # make city 0 the first in the path
    route = order_path(route)
    return route

best_itr_size = -1
best_score = -1
best_route = []

for band_size in range(2,30,1):
    start = time.time()
    
    # use vectorization to compute values for this column
    cities_df['band_index'] = (cities_df['dist_from_center'] / band_size).astype(int)
    route = route_from_bands(cities_df)
    score = calculate_score(route,cities_df)
    
    if best_itr_size == -1:
        best_itr_size = band_size
        best_score = score
        best_route = route
    else:
        if score < best_score:
            best_itr_size = band_size
            best_score = score
            best_route = route
    delta = int(time.time() - start)
    print("Band Iteration with size: {:d}.  Took {:d} sec and has score: {:,.2f}".format(band_size, delta, score))
    
print("Best itr size", best_itr_size, "best score", best_score, "route length", len(best_route))
plot_path(best_route,cities_df,"Banded Route - Score: {:,.2f} Path Length: {:d}".format(best_score,len(best_route)))


# ####  >99% Improvement
# 
# This looks much better!  We see that the "best" band size is 15.  We've reduced the route length from 446M to under 2.3M.  It also has some interesting characteristics:
# 
# * Compared to optimal, as seen on leaderboard, this is about 53% worse.
# * We gain some structure to our route - it does not look random
#  * But we have some problems such as the lines across which is not efficient (but solved later)
# 

# #### Logistics Considerations
# 
# I'm more interested in seeing what happens if we make some assumptions about Santa doing the deliveries.  Many of these assumptions are things we would need to consider in Logistics.
# 
# * Santa only delivers at night.  This means he cannot travel to arbitrary locations.  Instead he needs to visit a time zone when it is night and avoid timezones when it is daylight.
#  * We can model this by dividing the delivery area into 24 "wedges".  Think of each wedge being an area that Santa needs to complete all deliveries within 1 hour.
#  * With Logisitics we may want to accomplish all deliveries at the same time.  Thus we may assign each "wedge" to a different delivery person so that they can be done in parallel.
# * Santa probably can't carry all presents in his bag/sleigh.  Yes, he is magical but he may need to return to the North Pole to get more presents.
#  * With this assumption, we'll center the "wedges" around the North Pole and we'll make each "wedge" a roundtrip where he goes out and back.
#  * With Logistics we'd want a "Hub" where all the delivery people start from.  Thus the aligns well with "wedges" around the North Pole (Hub).
#  
# For simplicity, let's assume that each citiy receives the same number of presents.  Thus we can divide the cities equally across the wedges (each wedge thus has 1/24th of the total cities)

# In[ ]:


def create_wedge_route_using_distances(wedge_index, wedge_cities, band_size):    
    dc_min = wedge_cities['dist_from_center'].min()
    dc_max = wedge_cities['dist_from_center'].max()
        
    wedge_route = []
    index = 0
    not_done = True
    while not_done:
        b_dist_start = index * band_size
        b_dist_end = b_dist_start + band_size
        if b_dist_end > dc_max:
            b_dist_end = dc_max
            not_done = False
        
        # select rows between these distances
        band = wedge_cities.loc[(wedge_cities['dist_from_center'] >= b_dist_start) & (wedge_cities['dist_from_center'] <= b_dist_end)]

        # sort them by angle to traverse
        sorted = band.sort_values(by=['arctan2'])
        band_route = sorted['CityId'].tolist()
        
        # reverse the direction
        if (index % 2):
            band_route.reverse()
        wedge_route.extend(band_route)
        index += 1
        
    return wedge_route

def route_from_arctan_with_even_gift_distribution(cities_df,band_size):
    city_count = len(cities_df.index) - 1
    sorted_by_arctan2 = cities_df.sort_values(by=['arctan2'])
    
    # we want 24 routes (one for each hour of day)
    # and those 24 routes should go out and back
    # so compute 48 wedges
    cities_per_wedge = int(city_count / 48)
    route = []
    for w in range(0,48):
        # determine range of cities to operate on
        start_city = w * cities_per_wedge
        last_city = start_city + cities_per_wedge
        if (w == 47):
            last_city = city_count
            
        # select this range of cities, which forms a wedge since we sorted by arctan2
        this_wedge = sorted_by_arctan2.iloc[start_city:last_city]
        wedge_route = create_wedge_route_using_distances(w, this_wedge, band_size)
        
        # done with wedge
        if (w % 2):
            wedge_route.reverse()
        route.extend(wedge_route)
                
    # make city 0 the first in the path
    route = order_path(route)
    return route

route = route_from_arctan_with_even_gift_distribution(cities_df,15)
score = calculate_score(route,cities_df)
plot_path(route,cities_df,"Even gift distribution per wedge: Score:{:,.2f} Path Length: {:d}".format(score,len(route)))


# #### More Improvement with more features!
# 
# This is interesting as we've made 24 round trips from the north pole and, at the same time, gotten ~10% improvement in the total distance.
# 
# We can see the improvements from 2 aspects:
# 
# * We see more "whitespace" which indicates  we aren't traversing areas that are empty
#    * This is due to using "wedges" and the "wedges" do not need to cross an empty area whereas a complete "band" does
# * We see less of the lines that cross the area
#    * This is also due to using "wedges" where we don't have cities in the corners that need to connect along a "band"
#  
# We've also reduced how bad we are compared to optimal from ~53% to ~35%.
# 
# #### More Improvement?
# 
# Can we improve on this more?
# 
# * Notice that some wedges are densely packed whereas other are less.
#   * Could we use different "band" sizes in different wedges instead of 15 on all wedges?
# * We still have lines that cross.
#   * This is due to the 1st wedge spanning the entire left edge as there is little city density in that area.
#      * To fix this we will further divide any wedges who span is too big

# In[ ]:


# divide a wedge into subwedges so that a wedge does not span a large arc
def create_wedge_route_using_subwedges(wedge_cities, band_size):
    a_min = wedge_cities['arctan2'].min()
    a_max = wedge_cities['arctan2'].max()
    delta_a = a_max - a_min
    avg_delta = 6.28 / 48     # 48 wedges from -3.14 to + 3.14

    count = int(delta_a / avg_delta)
    # make an even number of subwedges (out and back)
    if (count % 2):
        count -= 1
        
    sweep = delta_a / count
    
    wedge_route = []
    for sw in range(0,count):
        start_a = a_min + (sweep * sw)
        end_a = start_a + sweep
        if (sw == count - 1):
            # last wedge
            end_a = a_max
            
        sw_cities = wedge_cities.loc[(wedge_cities['arctan2'] >= start_a) & (wedge_cities['arctan2'] <= end_a)]
        sw_route = create_wedge_route_using_distances_2(sw, sw_cities, band_size)
        if (sw % 2):
            sw_route.reverse()

        wedge_route.extend(sw_route)
        
    return wedge_route
        
    
    
def create_wedge_route_using_distances_2(wedge_index, wedge_cities, band_size):
    a_min = wedge_cities['arctan2'].min()
    a_max = wedge_cities['arctan2'].max()
    delta_a = a_max - a_min
    avg_delta = 6.28 / 48     # 48 wedges from -3.14 to + 3.14
    
    if delta_a > (3 * avg_delta):
        print("Wedge at {:d} has sweep of {:f} compared to average {:f}".format(wedge_index, delta_a, avg_delta))
        return create_wedge_route_using_subwedges(wedge_cities, band_size)
    
    dc_min = wedge_cities['dist_from_center'].min()
    dc_max = wedge_cities['dist_from_center'].max()
        
    wedge_route = []
    index = 0
    not_done = True
    
    while not_done:
        b_dist_start = index * band_size
        b_dist_end = b_dist_start + band_size
        if b_dist_end > dc_max:
            b_dist_end = dc_max
            not_done = False
        
        # select rows between these distances
        band = wedge_cities.loc[(wedge_cities['dist_from_center'] >= b_dist_start) & (wedge_cities['dist_from_center'] <= b_dist_end)]

        # sort them by angle to traverse
        sorted = band.sort_values(by=['arctan2'])
        band_route = sorted['CityId'].tolist()
        if (index % 2):
            band_route.reverse()
        wedge_route.extend(band_route)
        index += 1
        
    return wedge_route
    

def route_from_arctan_with_even_gift_distribution_2(cities,band_sizes,search_for_best_band_size_per_wedge=False):
    city_count = len(cities.index) - 1
    sorted_by_arctan2 = cities.sort_values(by=['arctan2'])
    
    # we want 24 routes (one for each hour of day)
    # and those 24 routes should go out and back
    # so compute 48 wedges
    cities_per_wedge = int(city_count / 48)
    route = []
    best_wedge_band_size = []
    for w in range(0,48):
        # determine range of cities to operate on
        start_city = w * cities_per_wedge
        last_city = start_city + cities_per_wedge
        if (w == 47):
            last_city = city_count
            
        # select this range of cities, which forms a wedge since we sorted by arctan2
        this_wedge = sorted_by_arctan2.iloc[start_city:last_city]
        
        if (search_for_best_band_size_per_wedge):
            best_bs = -1
            best_score = -1
            
            # use a distribution that is 10 sizes lower and higher than 15
            for bs in range(5,25,1):
                wedge_route = create_wedge_route_using_distances_2(w, this_wedge, bs)
                score = calculate_score(wedge_route,cities)
                if ((best_score == -1) or score < best_score):
                    best_score = score
                    best_bs = bs
            best_wedge_band_size.append(best_bs)
            
            # use the best to calculate this wedge
            wedge_route = create_wedge_route_using_distances_2(w, this_wedge, best_bs)
            print("Wedge Route: Index: {:d}  Score:{:,.2f}  BS: {:f}".format(w, best_score, best_bs))
        else:
            band_size = band_sizes[w]
            wedge_route = create_wedge_route_using_distances_2(w, this_wedge, band_size)
            score = calculate_score(wedge_route,cities)
            print("Wedge Route: Index: {:d}  Score:{:,.2f}  BS: {:f}".format(w, score, band_size))

        # done with wedge
        if (w % 2):
            wedge_route.reverse()
        route.extend(wedge_route)
        
    if (search_for_best_band_size_per_wedge):
        print("Best Wedge Band Sizes")
        print(*best_wedge_band_size, sep = ", ")
        
    # make city 0 the first in the path
    route = order_path(route)
    return route

route = route_from_arctan_with_even_gift_distribution_2(cities_df, [], search_for_best_band_size_per_wedge=True)
score = calculate_score(route,cities_df)
plot_path(route,cities_df,"Even gift distribution per wedge: Score:{:,.2f} Path Length: {:d}".format(score,len(route)))


# #### ~24% Less than Optimal
# 
# We now see a total route size of 1,865,000 which is about 24% less than Optimal (compared to Kaggle Competition leaderboard).  But we have some benefits:
# 
# * Our route contains round trips to the "hub" so we can get more gifts or have elfs do some of the timezones/wedges
# * Our route is broken up into evenly sized distributions (by city count)
# * Our route has a predictable path
#    * This is important in Logistics where we'd want to load packages in a "first in, last out" order
#       * Imagine the label for a gift being printed at the fulfillment center.  This is 1 or 2 days before delivery.  At that time we can print the "wedge" and "band" on the label.  This information can then be used to sort the package in "first in, last out" order into the delivery truck.  The alternative, which is also popular, is to relabel the package at the delivery "hub".
#  
# #### Further Considerations and Improvements
# 
# There is still more to consider and improve!
# 
# * Could we use a variable band width inside a wedge?
# * We balance the wedges by cities (with assumption that each city has the same number of stops and gifts), but we should also balance by distance and time
#    * For Logistics we'd want to assign the same amount of "work" to each delivery person
#       * Equal "work" is a balance of city density, time, distance, gifts and addresses per stop
