#!/usr/bin/env python
# coding: utf-8

# # Helping Santa:
# 
# In this kernal we will attempt to help Santa make his path through a number of cities. However, there is one plot twist: The houses in prime cities always leave carrots for the Reindeers alongside the usual cookies and milk. These carrots are just the sustenance the Reindeers need to keep pace. In fact, Rudolph has found that if the Reindeer team doesn't visit a prime city exactly every 10th step, it takes the 10% longer than it normally would to make their next destination!
# 
# ## Contents:
# 1.  Imports
# 2. Load/Preview DataSet
# 3. Optimization Solutions:
#     * Simplest/Stupidest Solution: No Optimization
#     * Next Simplest Solution: Sorting cities by X-coordinate
#     * Nearest Neighbor Solution
#     * Nearest Neighbor Solution (with some attepmpted prime city optimization)
#     * Keep Adding More Solutions!
# 4. Save Route Data For Competition Submission

# ## Imports:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Data Handling:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import math

# Plotting:
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Import/Preview Data:

# In[ ]:


DF_cities = pd.read_csv('../input/cities.csv')
print(DF_cities.shape)
print(DF_cities.head())


# Let's also create a column to denote if the city is a prime city (that is, the CityID is a prime number)

# In[ ]:


# Function to determine prime numbers:
def is_prime(n):
    if n > 2:
        i = 2
        while i ** 2 <= n:
            if n % i:
                i += 1
            else:
                return False
    elif n != 2:
        return False
    return True

#Create column in DF_cities to flag prime cities
DF_cities['IsPrime'] = DF_cities.CityId.apply(is_prime)

# Ok, let's preview the edited DF:
print(DF_cities.head(5))


# Okay. We can see that there are 197,769 cities (tha'ts a lot!) and that for each city, we are given x and y coordinates (these do not seem to relate to typical latitude and longitude values).
# 
# ### Next, let's plot all the locations of all of the cities to see what we are working with:
# 
# we know that the first data point = the North Pole, so we will highlight that one so it stands out (red)...
# 
# it would also be nice to highlight the 'prime' cities as well (green)...

# In[ ]:


DF_cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
# Add north pole in red, and much larger so it stands out
north_pole = DF_cities[DF_cities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=20)
# Add in prime cities in green and slightly larger
DF_primes = DF_cities.loc[DF_cities['IsPrime'] == True]
plt.scatter(DF_primes.X, DF_primes.Y, c='green', s=0.5);
plt.show()


# Cool. We can see that the city coordinates create a picture of Sant'as reindeer and some trees. The prime cities seem to be relativley evenly distributed thorughout the "map". 
# 
# How many Prime cities are there compared to not prime cities?
# 

# In[ ]:


print(DF_cities['IsPrime'].value_counts())
sns.countplot(x='IsPrime', data = DF_cities);


# Ok, there is about one tenth the amount of prime cities as regular cities... makes sense given that each 10th stop should be to a prime city to avoid the 10% distance penalty...

# ## Solutions:

# In[ ]:


# First Let's define a function to calculate distance for our solutions:
# It'd also be nice if it told us how much of a distance penalty we accrued...
def total_distance(route):
    coords = DF_cities[['X', 'Y']].values
    summed_dist = 0
    summed_extra = 0
    for i in range(1, len(route)):
        extra = 0
        begin = route[i-1]
        end = route[i]
        distance = ((coords[end,0]-coords[begin,0])**2 + (coords[end,1]-coords[begin,1])**2)**0.5
        if i%10 == 0:
            # Edit this part
            if is_prime(begin):
                pass
            else:
                distance *= 1.1
                extra = distance * 0.1
            # if begin not in PRIMES:
                # distance *= 1.1
        summed_dist += distance
        summed_extra += extra
    print('Total Distance:  ' + str(summed_dist) + '\nPenalty: ' + str(summed_extra) + '(' + str((summed_extra/summed_dist)*100) + '%)')
    return summed_dist, summed_extra
print('done!')


# ### Simplest Solution (No Optimization)
# This is the dumbest solution we could do. It doesn't even take into acount any optimization. It purely goes down the list of cities we were given, in the order they were given, and calculates the distances between consecutive cities and sums them up (also adds the return trip to the North pole from the last city)
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "route = list(DF_cities['CityId'])\nroute.append(0)\ntotal_distance(route)")


# While this solution is dumb, at least it gives us an idea of a starting point. It also quickly highlights that the distance punishment is pretty minimal. This was somewhat expected since there is only a distance punishment every 10th city, and sometimes you get lucky and that city ends up being a prime city by chance, so the total penalty is <1% of the total distance.

# ### Next Simplest Solution (Sort by City X coordinate)
# Clearly it is going to be very inefficient to be going back and forth all over the map, like we did in the last un-optimized solution. However, if we start at the one side and go to the other, we should reduce our distance pretty dramatically, without much effort at all...
# 
# Let's do this by sorting the cities by the X location:

# In[ ]:


get_ipython().run_cell_magic('time', '', "DF_cities = pd.read_csv('../input/cities.csv') # to prevent confusion with notebooks... \ntemp = DF_cities.drop(DF_cities.index[0]).sort_values(['X'], ascending = 1)\n\nroute = [0] # start at North Pole\nroute.extend(list(temp['CityId'])) # All other stops\nroute.append(0)  # End at North Pole\n\ntotal_distance(route)")


# Sorting the cities by the X-coordinate lowered our total trip distance from ~447 million to ~196 million. Sweet! Not a bad improvement from such a simple solution.

# Now we can start working on some slightly more sophisticated solutions...
# 
# It makes sense that if we design an algorithm that always chooses the nearest remaining city, we should be able to greatly reduce our total trip distance. I'm sure we will run into some sub-optimal / long trips towards the end as the number of remaining cities is low, but it should still be pretty decent. This is commonly known as the [Nearest Neighbor Aproach](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm).
# ### Nearest Neighbor Approach:
# 
# 
# This will involve the following steps (via the Wiki link above):
# 1. start on an arbitrary vertex as current vertex  (Less arbitrarily, the North Pole for us).
# 2. find out the shortest edge connecting current vertex and an unvisited vertex V.
# 3. set current vertex to V.
# 4. mark V as visited (drop from the DF).
# 5. if all the vertices in domain are visited, then terminate.
# 6. In our case, add the North Pole as the final stop and add up the total distance traveled.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Simple Nearest Neighbor Model:\n\n# Re-load CD_cities (to avoid issues with it being modified in previous cells)\nDF_cities = pd.read_csv('../input/cities.csv')\n\nIDs = DF_cities.CityId.values\ncoords = np.array([DF_cities.X.values, DF_cities.Y.values]).T\n\n# Set intial position (north pole)\nposition = coords[0]\nroute = [0]\n\n# Remove initial position from list now that it has already been added to the route\nIDs = np.delete(IDs, 0)\ncoords = np.delete(coords, 0, axis=0)\n\ncount = 0\n# Loop through remaining cities to fill in route with the nearest remaining cities:\nwhile len(IDs) > 0:\n    # create matrix of distances from remaining cities to current city\n    distance_matrix = np.linalg.norm(coords - position, axis=1)\n    \n    # Find Nearest City (minimum distance)\n    idx_min = distance_matrix.argmin()  # np.argmin returns the index of the min value\n    \n    # Set position for next loop and remove this city from list\n    route.append(IDs[idx_min])\n    position = coords[idx_min]\n    IDs= np.delete(IDs, idx_min, axis=0)\n    coords = np.delete(coords, idx_min, axis=0)\n    \n    # print out updates on progress of loop every 10000 iterations:\n    if count % 10000 == 0:\n        print(str(len(IDs))+ ' cities left')\n    count += 1\n    \n# Finally, end back at the north pole:\nroute.append(0)\n\n# Use the function from above to calculate the total distance travelled:\ntotal_distance(route)")


# So going with a slightly smarter aproach (nearest neighbor) has definitely improved the total distance that Santa will need to travel. We went from ~196 million with the x-coordinate sorting down to ~ 1.81 million miles [took ~3.5 minutes through Kaggle Kernal). Pretty awesome improvement, but still worse than ~85% of other contest submissions...

# ### Nearest Neighbor Approach (With Prime City Optimization):
# What if we add in some of the logic to help optimize for prime city distance penalties?...
# 
# Here, for every 10th step, we multiply the non-prime members of the calculated distance_matrix of the remaining cities by 1.10. That way, the solution will only go out of the way to visit a prime city if it doesn't exceed the 10% distance penalty. My guess is that this will yield a fairly minimal benefit (on the order of a fraction of 1% improvement).

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Re-load CD_cities (to avoid issues with it being modified in previous cells)\nDF_cities = pd.read_csv('../input/cities.csv')\nDF_cities['IsPrime'] = DF_cities.CityId.apply(is_prime)\n\nIDs = DF_cities.CityId.values\ncoords = np.array([DF_cities.X.values, DF_cities.Y.values]).T\nprimes = np.array(DF_cities.IsPrime)\n\n# Set intial position (north pole)\nposition = coords[0]\nroute = [0]\n\n# Remove initial position from list now that it has already been added to the route\nIDs = np.delete(IDs, 0)\ncoords = np.delete(coords, 0, axis=0)\nprimes = np.delete(primes,0)\n    \ncount = 0\n# Loop through remaining cities to fill in route with the nearest remaining cities:\nwhile len(IDs) > 0:\n    \n    # add to count:\n    count += 1\n    \n    # create matrix of distances from remaining cities to current city\n    distance_matrix = np.linalg.norm(coords - position, axis=1)\n    \n    if count % 10 == 0:\n        idx_toPenalize = np.where(primes == False )[0]\n        distance_matrix[idx_toPenalize] *= 1.10\n\n    idx_min = distance_matrix.argmin()  # np.argmin returns the index of the min value\n    \n    # Find Nearest City (minimum distance)\n    idx_min = distance_matrix.argmin()  # np.argmin returns the index of the min value\n    \n    # Set position for next loop and remove this city from list\n    route.append(IDs[idx_min])\n    position = coords[idx_min]\n    IDs= np.delete(IDs, idx_min, axis=0)\n    coords = np.delete(coords, idx_min, axis=0)\n    primes = np.delete(primes, idx_min, axis=0)\n    \n    # print out updates on progress of loop every 10000 iterations:\n    if count % 10000 == 0:\n        print(str(len(IDs))+ ' cities left')\n    \n# Finally, end back at the north pole:\nroute.append(0)\n\n# Use the function from above to calculate the total distance travelled:\ntotal_distance(route)")


# Using this method of optimizing visits to prime cities, we reduced the total route distance from 1,812,602 down to 1,812,550. 
# 
# Hmmm... This only improved the distance traveled by 52 (miles?)... Lame. And somehow, the total distance incurred from penalties actually increased from 17892 to 18004. That means that I have either messed up this algorithm, or at some point the route had to make-up for passed-over cities...
# 
# (If you spot my mistake, please let me know in the comments!)

# ## Smarter Solutions:
# to be updated...I still haven't added in any logic to take into acount the penalty on prime cities. Though, given that doing this will likely only improve my solution by <1%, maybe initial efforts are better spent finding a different strategy.

# ## Save the Route Data For Competition Submission:

# In[ ]:


output = pd.DataFrame({'Path': route})
output.to_csv('submission.csv', index=False)

print('Output data file saved')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




