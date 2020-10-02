#!/usr/bin/env python
# coding: utf-8

# # All the Carrots: Traveling Santa 2018
# ## Santa and the Reindeer talk after the Rebellion
# 
# When the Reindeer heard of Rudolph's algorithm to score Santa's path they were not very happy and, as you know,
# this lead to the ["Reindeer Rebellion"](https://www.kaggle.com/dan3dewey/reindeer-rebellion-traveling-santa-2018).  
# 
# After the Rebellion, the Reindeer and Santa talked together to see what compromise might be made.  During their discussion, Santa revealed that the every-ten-cities designation is important to him since he collects cookies at each city and then only eats them every-10th city for efficiency (it takes some time to get the crumbs out of his beard).  In addition, the accompanying Elves take a brief rest break at the 10th city (as specified in the Elf-Union contract.) If the city before the 10th city has carrots (is prime) Santa grabs the carrots and gives them to the Reindeer to eat on the 10th city while he and the Elves take their break.
# 
# 
# With these new details the Reindeer proposed an alternate algorithm:<br>
# ** * Find the shortest path that has EVERY prime city occuring just before a 10th city* ** <br>
# (Actually, it's on every 10th city if the NP is counted as city 1.).<br>
# In this way the Reindeer will get all the carrots that have been left for them.  The Reindeer understand that not every 10th city will have carrots (17802 carrot cities vs 19776 10th cities), but, as long as Santa's path gets them all the available carrots they have agreed to work full speed between all cities charging no penalty.
# 
# **With the results below (v7), the Reindeer propose a "two-loop" path:<br>
# After leaving the North Pole, they are excited and energized and will do 19740 cities at top speed with no carrots. <br>
# Then for the rest of the night, the path will get them All the Carrots with carrots at each 10th city. <br>
# The total travel length is 2374798 . For reference its Rudolph Score is 2377086 (from a penalty fraction of 0.09 %)**
# <br>

# ## Notes/Diary:
# This kernel borrows from the Theo Viel's
# ["Greedy Reindeer"](https://www.kaggle.com/theoviel/greedy-reindeer-starter-code). <br>
# "Backbone" input files are made with a **companion kernel**:
# [Concorde - Primes only](https://www.kaggle.com/dan3dewey/concorde-primes-only) <br>
# which is based on William Cukierski's 
# ["Concorde solver"](https://www.kaggle.com/wcukierski/concorde-solver) kernel. <br>
# 
# **General Approach...** Given that the path will include all prime cities at 1/10th-city locations, one way to determine the path is by answering these questions (perhaps iteratively): <br>
# i) Which other cities will also be at 1/10th-city locations? , e.g. to make a complete backbone. <br>
# ii) How to determine the order of the backbone cities along the backbone? <br>
# iii) Which 9 cities are assigned to each backbone segment? , i.e., which 9 cities are put between backbone-adjacent 1/10th-cities?<br>
# iv) How is the path-order of the 9 cities within a segment decided? <br>
# 
# * 7 Dec 2018: (v1) Start exploring this idea using a Concorde-TSP NP+primes-only path as a backbone and adding (inserting) 9 cities before each prime city and return to the North Pole. The resulting path of 178021 cities does have primes at every 10th city scoring no additional penalty. (Of course this path does not have all the cities on it.)<br>
# * 8 Dec 2018: (v2) Determine and include the proper number of cities (9) on the segment from the last backbone city back to the NP so that the remaining number of un-connected cities is a multiple of 10. Select 1/10 of these to add to the prime backbone and write out a csv file that is the new, full backbone (i.e., has enough additional backbone cities to be able to make a penalty-free path that incudes all of the cities.)  Use this new backbone list as input to the
# [Concorde - Primes only](https://www.kaggle.com/dan3dewey/concorde-primes-only) kernel(v5)
# to output the full backbone in an optimum TSP order, this full backbone itself has length 382762. <br>
# * 9 Dec 2018: (pre-v3) Use the new full backbone to generate an All-Carrots path - it worked getting all 17802 Carrots. However, the Rudolph score is 8284825 from a total Length of: 8266128
# The 'answers' to the General Approach questions used at this point are: i) randomly pick 1/10th of 'remaining' cities (cities not assigned to a primes-only TSP backbone). ii) Backbone cities are put in TSP solution order. iii) For each segment, select/assign he 9 (remaining) cities that are closest to the midpoint of the backbone segment. iv) keep segment cities in their selected order.
# * 11 Dec 2018: Setup a "backbone database" to capture useful information about the backbone and path. Use a little better full-backbone TSP solution (from "Primes only" (v7) with t_bound = 5000). <br>
# * 11-15 Dec 2018: (v3) Implement a sequential insertion scheme to order the 9 cities within each backbone segment. <br>
# It might speed things up to use the efficient code developed in hirune924's kernel: [Random Insertion (No Concord Solver Solution)
# ](https://www.kaggle.com/hirune924/random-insertion-no-concord-solver-solution)  ;-) <br>
# * 15 Dec 2018: Apply the sequential insertion scheme to the segments. Some very large MaxRadius values... Start again with the t5000 primes only TSP to see how that is (without the remaining non-prime cities.) Does reasonably well, length = 2284014. This suggests that including the additional 19740 cities, if they had similar efficiency, would give a length increased by a factor (19740+178030)/(178030), or 1.111, for a total length around 2540000.
# * 16 Dec 2018: (v4) Did a TSP ("Concorde - Primes only" kernel) on the "Remaining cities" left after the primes-only backbone was used. Combined 1/10th of these with the Primes and NP to form a complete backbone (TSP'ed) that should better include all the cities... But still many large spacings are needed, length ~ 4300000.
# * 17-19 Dec 2018: Since a full path has to use all the cities, experimenting with less than 9 cities per segment shows that the problem (length) comes when getting to the last bb cities and having to pick segment-cities that are scattered and far away. <br>
# Try to reduce this problem by doing things like: adding one city at a time to each backbone segment, and then add a second to each, etc. Could also do the backbone cities going from least to most dense... Since the backbone may not be constructed and filled in in path order, generate the final path after-the-fact from the bb dataframe. To allow cities for each segment to be added incrementally, keep/update the CityListStr values in the dfbb (v5). <br>
# * 20 Dec 2018: Include information in the dfbb based on each segment in isolation (RadAlone, LenAlone) and write it to a csv file (to speed further processing.) Filling the prime backbone segments and then the non-prime ones mimics the way that non-prime backbone cities were chosen. <br>
# Gets all 17802 carrots with a Rudolph score and Length of 2993490 and 2983638 (v6).<br>
# * 21 Dec 2018: **What if the Reindeer did (essentially) two loops?** One with all primes cities and one with the remaining cities? All primes has length/score of 2151297.  Put the 19740 remaining cities into TSP order (include NP too) with Concorde... that has length ~ 220000. <br>
# So all together the complete path length is 2374798 (Rudolph score 2377086) - the best "All the Carrots" path so far ;-)  (v7)
# * 26-27 Dec 2018: Look into ways to improve the Primes path (this will change the Remaining): i) limiting distance sequentially across all segements; ii) using the Centroid of exisiting path to calculate distances; iii) do the Compact cities in stages to take advantage of the centroid; iv) adjust compact city selection ratio for best length. <br>
# Got down to **2208262**. (v8)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import Counter
from time import time
from matplotlib import collections  as mc

sns.set_style('whitegrid')


# ## Load Cities Data with prime indication

# In[ ]:


# Get all the cities:   Note: CityId is equal to the index
df_cities = pd.read_csv("../input/traveling-santa-2018-prime-paths/cities.csv")
##df_cities.head(10)


# In[ ]:


def sieve_eratosthenes(n):
    prime_flags = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (prime_flags[p] == True):
            for i in range(p * 2, n + 1, p):
                prime_flags[i] = False
        p += 1
    return prime_flags


# In[ ]:


max_city = max(df_cities.CityId)
print("Largest City number to visit : ", max_city)


# In[ ]:


prime_flags = np.array(sieve_eratosthenes(max_city)).astype(int)
df_cities['Prime'] = prime_flags

df_cities.head(5)


# In[ ]:


df_cities.tail()


# ## Various routines to use

# In[ ]:


# Assign a distance measure from xc,yc to all cities
# If remaining is present then only that subset are calculated (if few enough)
# and/but the return array is full size.
def dist_array(coords_in, xc, yc, remaining=[]):
    begin = np.array([xc, yc])[:, np.newaxis]
    # Doing all cities is generally faster, unless a small number (< 60000)
    if (len(remaining) < 1) or (len(remaining) > 60000):
        dists = np.linalg.norm(coords_in - begin, ord=2, axis=0)
    else:
        dists = np.zeros((len(coords_in[0])))
        dists[remaining] = np.linalg.norm(coords_in[: , remaining] - begin, ord=2, axis=0)
    return dists

# return the index of the nearest available city, and its distance
def get_next_city(dist, avail):
    min_avail = np.argmin(dist[avail])
    return avail[min_avail], dist[avail[min_avail]]

def plot_path(path, coordinates, np_star=True, end_stars=False, prime_vals=[]):
    # Plot tour
    lines = [[coordinates[: ,path[i-1]], coordinates[:, path[i]]] for i in range(1, len(path))]
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_aspect('equal')
    plt.grid(False)
    ax.add_collection(lc)
    ax.autoscale()
    if np_star:
        # add the North Pole location
        plt.scatter(coordinates[0][0], coordinates[1][0], s=150, c="red", marker="*", linewidth=3)
        # and first cities on the path
        ##plt.scatter(coordinates[0][path[1:10]], coordinates[1][path[1:10]], s=15, c="black")
    if end_stars:
        # add stars at first and last location
        plt.scatter(coordinates[0][path[0]], coordinates[1][path[0]], s=150, c="red", marker="*", linewidth=3)
        plt.scatter(coordinates[0][path[-1]], coordinates[1][path[-1]], s=150, c="red", marker="*", linewidth=3)
    if len(prime_vals) == len(coordinates[0]):
        # Mark the prime cities
        for this_city in path:
            if prime_vals[this_city] == 1:
                plt.scatter(coordinates[0][this_city], coordinates[1][this_city],
                            s=60, c="green", marker="*", linewidth=3)
    plt.show()
    
# Calculate the Score, Carrots, and Length for a path
def usual_score(path, coords, prime_flags):
    score = 0
    carrots = 0 
    length = 0
    for i in range(1, len(path)):
        begin = path[i-1]
        end = path[i]
        distance = np.linalg.norm(coords[:, end] - coords[:, begin], ord=2)
        length += distance
        if i % 10 == 0:
            # if the starting city is prime then a carrot and no penalties
            if prime_flags[begin]:
                carrots += 1
            # if not prime, no carrot and a penalty
            else:
                distance *= 1.1
        score += distance

    return score, carrots, length

def str_from_cities(cities_list):
    # Create a comma-separated string of cities_list
    cities_str = ''
    for icty in range(len(cities_list)):
        cities_str += str(int(cities_list[icty]))+', '
    return cities_str

def cities_from_str(cities_str):
    # extract a list of city ids from cities_str,
    pieces = cities_str.split(",")
    # start with an empty list
    city_list = []
    # if the first city has value >= 0 then get and return the cities
    if int(pieces[0]) >= 0:
        for ipiece in range(len(pieces)-1):
            city_list.append(int(pieces[ipiece]))
    return city_list


# ### Routine to select cities for a backbone segment

# In[ ]:


def add_to_segment(bb_next, n_between, mpdist_limit=1.e6):        
    # Routine to increase selected segment, bb_next, to have n_between cities,
    # will add/extend to have n_between nearest-to-mid-point cities in the path.
    # Will not add cities that are further than mpdist_limit.
    # If the segment is partially filled, the centroid of cities replaces backbone mid-point.
    #
    # Use: n_added = add_to_segment(1,8)  # add 8 from NP to first bb city
    
    # Assumes access to read these variables:
    #  backbone[]   coordinates[[][]]
    #
    # Set these to global to allow modifying values as well:
    global dfbb
    global remaining

    # Return the number that were added
    n_added = 0
    
    # Don't do bb 0:
    if bb_next == 0:
        return n_added
    
    # Check how many are in this segment already, return if there are enough already.
    seg_cities = cities_from_str(dfbb.loc[bb_next,'CityListStr'])
    if len(seg_cities) >= n_between:
        print("Segment already full for bb_next =", backbone[bb_next])
        return n_added
    
    # Need to add one or more, so ...
    # If this backbone segment is empty then start it with one city
    if len(seg_cities) == 0:
        # Calc midpoint (average location) between bb_last and bb_next (faster than getting it from the dfbb?)
        mid_pt = 0.5*(coordinates[:, backbone[bb_next - 1]] + coordinates[:, backbone[bb_next]])
        # Calc distances of all remaining points from this mid_pt
        dist_from_mid = dist_array(coordinates, mid_pt[0], mid_pt[1], remaining=remaining)
        
        # Start with the remaining city closest to midpoint
        add_this, test_dist = get_next_city(dist_from_mid, remaining)
        # Only start it if its within mpdist_limit:
        if test_dist < mpdist_limit:
            this_dist = 1.0*test_dist
            remaining = np.setdiff1d(remaining, add_this)
            n_added += 1
            # and create a path with that city in the middle
            built_path = [backbone[bb_next - 1], add_this, backbone[bb_next]]
            _dum1, _dum2, this_len = usual_score(built_path, coordinates, prime_flags)
            best_score = this_len
        else:
            # Too far away, forget it
            # Nothing to update
            return n_added
    else:
        # Assemble the already assigned cities into built_path with backbone endpoints
        built_path = [backbone[bb_next - 1]] + seg_cities + [backbone[bb_next]]
        #
        # Calc the centroid of all cities on this path (use same mid_pt variable)
        mid_pt = 1.0*coordinates[:, built_path[0]]
        for seg_city in built_path[1:]:
            mid_pt += coordinates[:, seg_city]
        mid_pt = mid_pt/len(built_path)
        # Calc distances of all remaining points from this mid_pt
        dist_from_mid = dist_array(coordinates, mid_pt[0], mid_pt[1], remaining=remaining)
        
    # Now, add additional cities... Add them in order from closest to farthest from midpoint
    # and/but choose where in the segment path is best when inserting each city.
    for iadd in range(len(built_path)-1,n_between+1):
        add_this, test_dist = get_next_city(dist_from_mid, remaining)
        # Only add it if its within mpdist_limit:
        if test_dist < mpdist_limit:
            this_dist = 1.0*test_dist
            remaining = np.setdiff1d(remaining, add_this)
            n_added += 1
            # Loop over insertion location
            # keep track of shortest path and where the insertion for it was
            best_score = 1.e9
            best_insert_path = []
            for iinsert in range(1, len(built_path)):
                test_path = built_path[:iinsert] + [add_this]+ built_path[iinsert:]
                _dum1, _dum2, this_len = usual_score(test_path, coordinates, prime_flags)
                # check the length for this path
                if this_len < best_score:
                    best_score = this_len
                    best_insert_path = test_path
            # OK, put the added city in best location in the path:
            built_path = best_insert_path
        else:
            # too far away, stop adding here
            break  # out of the for loop adding cities
        # Go back to get next city for the segment
        
    # Done getting and arranging the desired number of cities for the segment
    # 
    if n_added > 0:
        # save these cities (without bb end points) in the dfbb
        dfbb.loc[bb_next, 'CityListStr'] = str_from_cities(built_path[1 : -1])
        # save the number of cities assigned to the segment
        dfbb.loc[bb_next, 'NumCities'] = len(built_path[1 : -1])
        # and the (max) distance of the last selected city
        dfbb.loc[bb_next, 'MaxRadius'] = this_dist
        # and the current length of the backbone segment
        dfbb.loc[bb_next, 'Length'] = best_score
    
    # end of add_to_segment()
    return n_added


# ## Get the backbone path
# This path has been pre-computed by [Concorde - Primes only](https://www.kaggle.com/dan3dewey/concorde-primes-only).
# That code may be included here to have a single kernel for the problem.

# In[ ]:


# Get the Backbone path

# TSP solution for the North Pole and the Prime cities:
##df_bbtsp = pd.read_csv("../input/santa-18-primes-tsp60/primes_path_t60.csv")
# Slightly improved all primes TSP
df_bbtsp = pd.read_csv("../input/santa-18-primes-tsp60/primes_path_t5000.csv")
bb_name = "bb_primesTSP5000"

df_bbtsp.head()


# In[ ]:


df_bbtsp.tail()


# In[ ]:


# Generate the backbone list of CityIds, starting and ending at NP
# Note that the 'Path' column values are prime-city-indices in path order,
# that is, the loc of the CityId in the df.
backbone = list(df_bbtsp.loc[list(df_bbtsp['Path']),'CityId'])

# loop back to the start of the bb (e.g. the NP for a usual bb)
backbone += [backbone[0]]
len_bb = len(backbone)
print("Backbone length is", len_bb, "\nStarts with cities:\n  ", backbone[0:10],
     "\nEnds with cities:\n    ", backbone[len_bb-10:])


# In[ ]:


# Show all the cities and the backbone cities
plt.figure(figsize=(15, 10))
# All cities:
plt.scatter(df_cities.X, df_cities.Y, s=1, c="green", alpha=0.3)
# Prime cities:
plt.scatter(df_cities.loc[backbone, "X"], df_cities.loc[backbone, "Y"], s=1, c="red")
plt.scatter(df_cities.iloc[0: 1, 1], df_cities.iloc[0: 1, 2], s=30, c="black")
plt.grid(False)
plt.show()


# In[ ]:


# Put all the city coordinates in an np array
coordinates = np.array([df_cities.X, df_cities.Y])


# In[ ]:


# Look at the backbone (red star is NP)
# whole thing
plot_path(backbone, coordinates)


# In[ ]:


# Can calculate the path-length of the backbone-path (NP to NP) itself
_dum1, _dum2, bb_len = usual_score(backbone+[0], coordinates, prime_flags)

print("Backbone path-length is ", int(bb_len))


# ## Setup a backbone dataframe
# Put useful information about each backbone segment in a dataframe.

# In[ ]:


# * * * Can use pre-made file * * *
# Read in previously saved backbone dataframe:
# Full backbone with Primes and non-primes
##dfbb = pd.read_csv("dfbb_primesTSPremaining_v6_SAVE.csv", sep=";", index_col=0)
# Backbone of just the Primes
##dfbb = pd.read_csv("dfbb_primesTSP5000_v8_empty.csv", sep=";", index_col=0)
# * * to Skip the following cells * * * *


# In[ ]:


#  Or, make and fill the dfbb...


# In[ ]:


# Create a dataframe to keep track of information about the various segments of the path
#(segment = piece = cities between adjacent bb cities = cities on path between this and previous bb city).
dfbb = df_cities.loc[backbone].copy().reset_index()

# drop the new "index" column (same as CityId)
dfbb = dfbb.drop(columns='index')

dfbb.head()


# In[ ]:


# Create a dataframe to keep track of information about the various segments of the path
#(segment = piece = cities between adjacent bb cities = cities on path between this and previous bb city).
dfbb = df_cities.loc[backbone].copy().reset_index()

# drop the "index" column
dfbb = dfbb.drop(columns='index')

# Coordinates of the segment midpoint
dfbb["MidX"] = 0.0
dfbb["MidY"] = 0.0

# The direct length from previous bb city to this bb city
dfbb['BBtoBBlen'] = 0.0

# Define these other values in dfbb - they give information on the solution status
# The maximum radial distance from the segment midpoint of the cities assigned to this segment
dfbb['MaxRadius'] = 0.0
# Define and initialize "RadAlone" in dataframe (min midpoint radius with 9 cities)
dfbb['RadAlone'] = 0.0
# The length of the segment as currently constructed
dfbb['Length'] = 0.0
# Define and initialize "LenAlone" in dataframe (est. of min length with 9 closest cities)
dfbb['LenAlone'] = 0.0
# The number of cities assigned to this segment (e.g. in CityListStr)
dfbb['NumCities'] = 0
# A string entry to keep track of the cities that are assigned to this segment
dfbb['CityListStr'] = '-1,'

# Fill the BBtoBBlen and Mid point values (they don't change)
for bb_next in range(1,len(backbone)):
    xy_next = coordinates[:, backbone[bb_next]]
    xy_last = coordinates[:, backbone[bb_next-1]]
    dfbb.loc[bb_next, 'BBtoBBlen'] = np.linalg.norm(xy_next - xy_last, ord=2)
    mid_pt = 0.5*(xy_last + xy_next)
    dfbb.loc[bb_next, "MidX"] = mid_pt[0]
    dfbb.loc[bb_next, "MidY"] = mid_pt[1]
    if bb_next % 1000 == 0:
        print(" ... ", bb_next, "bb lengths, mid-points calculated.")


# ### Find and save each backbone's best, "all alone", length

# In[ ]:


# Keep track of time...
t0 = time()

# Start with all cities:
remaining = np.array(df_cities.CityId)
# Remove the ones in the backbone
all_remaining = np.setdiff1d(remaining, np.array(backbone))

# First segment (special)
bb_next = 1
remaining = all_remaining.copy()
n_added = add_to_segment(bb_next, 8)

# all the rest
for bb_next in range(2,len(backbone)):
    remaining = all_remaining.copy()
    n_added = add_to_segment(bb_next, 9)
    # show the progress
    if bb_next % 500 == 0:
        print( int(time() - t0), "seconds   - ...added bb number", bb_next)
        
# This takes longer because there are the maximum remaining number of cities for every segment.
# Could imagine writing out the dfbb dataframe at this point,
# its values are useful to a particular backbone.


# In[ ]:


# Transfer/Save these values
dfbb['LenAlone'] = dfbb['Length']
dfbb['RadAlone'] = dfbb['MaxRadius']
# and reset these values
dfbb['MaxRadius'] = 0.0
dfbb['Length'] = 0.0
dfbb['CityListStr'] = '-1,'


# In[ ]:


# Save the dataframe since it takes a while to create...
# Use ";" as separation character since I use commas in CityListStr.
dfbb.to_csv("df"+bb_name+"_v8_empty.csv", sep=";")


# In[ ]:


# * * * Below assumes dfbb was either made (as above) or read in (farther above) * * *


# ## Look at Backbone Dataframe

# In[ ]:


# Look at the dfbb
dfbb.head(5)


# In[ ]:


dfbb.tail(5)


# In[ ]:


# The total LenAlone, roughly the (optimistic?) best one could get with this backbone and these cities.
dfbb['LenAlone'].sum()


# In[ ]:


# Show the "alone" properties of the segments
pd.plotting.scatter_matrix(dfbb[['BBtoBBlen','RadAlone','LenAlone']], figsize=(14,9),
                           diagonal='hist', range_padding=0.05, hist_kwds={'bins':100},
                           grid=True, alpha=0.5)
plt.show()


# ## Add cities between the backbone cities
# The backbone gives the every 1/10th cities, need to assign 9 other cites to each backbone segment.  Current method to assign them is to find the 9 that are closest to the midpoint of the backbone segment.

# In[ ]:


# (Re)Start by resetting the city-selection values in dfbb:

def reset_segments():
    global bdfbb, remaining
    
    # The maximum radial distance from the segment midpoint of the cities assigned to this segment
    dfbb['MaxRadius'] = 0.0
    # The length of the segment as currently constructed
    dfbb['Length'] = 0.0
    # The number of cities assigned to this segment (e.g. in CityListStr)
    dfbb['NumCities'] = 0
    # A string entry to keep track of the cities that are assigned to this segment
    dfbb['CityListStr'] = '-1,'

    # Initialize the "remaining" cities to be the ones not in the backbone
    # Start with all cities:
    remaining = np.array(df_cities.CityId)
    # Remove the ones in the backbone
    remaining = np.setdiff1d(remaining, np.array(backbone))

    print("Total number of cities (includes NP):", max_city+1,
          "\nUnique cities in the backbone:", len_bb-1,
          "\nNumber of cities not in the backbone: ", len(remaining),
          "\n   Total primes among these:", sum(prime_flags[remaining]),  "<-- should be zero!")

# And do it
reset_segments()

# Calculate how many cities will be between the last non-NP bb city and the (return to the) NP 
# so that there will be a multiple of 10 remaining after adding between the bb cities.
n_last_bb_to_NP = ((max_city+1) - 10*(len_bb-2)) % 10
print("\nPut", n_last_bb_to_NP, "cities in the final stretch back to the NP.")


# In[ ]:


# Generate a path that has 9 non-backbone cities put before/between each backbone city;
# except: 8 between NP and first bb city,
# and a calculated number, n_last_bb_to_NP, between next-to-last bb city and NP.
#

# Keep track of the clock time
t0 = time()

# number per backbone segment, 9 unless doing an experiment
n_per_seg = 9


# Fill the first and last segment
if True:
    # First do NP to first city (special number between):
    n_added = add_to_segment(1, 8)

    # Do the last bb city back to NP (could be a special number, it's 9 actually):
    n_added = add_to_segment(len(backbone)-1, n_last_bb_to_NP)


# Do the "Compact" Primes
# - Even and then Odd bb index ones ("every other"-ish spacing)
# - Select the 9 in sets: first 4, then 3 more, then last 2
# - Centroid used for distance determination
# - No distance limit in adding cities
if True:

    # - - - "compact" ones have the RadAlone/BBtoBBlen ratio less than a value
    compact_ratio = 1.60
    # Do ones that are 'compact':
    ibb_compact = dfbb[(dfbb['RadAlone'] < compact_ratio*dfbb['BBtoBBlen']) & (dfbb['NumCities'] < n_per_seg) &                   (dfbb.index % 2 == 0) & (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_compact),"Compact Even Prime cities...")
    ndone = 0
    for bb_next in ibb_compact:
        # Do all 9 at once...
        ##n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        # Do some, then some more, then do the remaining ones (uses centroid of the some+2)
        n_added = add_to_segment(bb_next, 4, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, 7, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        #
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))

    ibb_compact = dfbb[(dfbb['RadAlone'] < compact_ratio*dfbb['BBtoBBlen']) & (dfbb['NumCities'] < n_per_seg) &                   (dfbb.index % 2 == 1) & (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_compact),"Compact Odd Prime cities...")
    ndone = 0
    for bb_next in ibb_compact:
        # Do all 9 at once...
        ##n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        # Do some, then some more, then do the remaining ones (uses centroid of the some+2)
        n_added = add_to_segment(bb_next, 4, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, 7, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        #
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))


mpdist_limit = 15.0     
while mpdist_limit < 400.0:

    print("\n\n--- Doing dist limit of ", mpdist_limit, "---")
    # Do the un-finished ODD and EVEN PRIMES
    
    # Do the  - EVEN ones -
    ibb_even = dfbb[(dfbb.index % 2 == 0) & (dfbb['NumCities'] < n_per_seg) &                (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_even),"Even Prime cities...")
    ndone = 0
    for bb_next in ibb_even:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))
            
    # Do the  - ODD ones ( > 1 ) -
    ibb_odd = dfbb[(dfbb.index % 2 == 1) & (dfbb.index > 1) & (dfbb['NumCities'] < n_per_seg) &               (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_odd),"Odd Prime cities...")
    ndone = 0
    for bb_next in ibb_odd:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))

    mpdist_limit += 5.0 + 0.1*mpdist_limit

    
if True:
    # All/Any remaining cities added
    # distance limit (1.e6 if none)
    mpdist_limit = 1.e6
    
    # Do the un-finished ODD and EVEN PRIMES
    
    # Do the  - EVEN ones -
    ibb_even = dfbb[(dfbb.index % 2 == 0) & (dfbb['NumCities'] < n_per_seg) &                (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_even),"Even Prime cities...")
    ndone = 0
    for bb_next in ibb_even:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))
            
    # Do the  - ODD ones ( > 1 ) -
    ibb_odd = dfbb[(dfbb.index % 2 == 1) & (dfbb.index > 1) & (dfbb['NumCities'] < n_per_seg) &               (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_odd),"Odd Prime cities...")
    ndone = 0
    for bb_next in ibb_odd:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))

            
# All done selecting cities within each backbone segment
print( int(time() - t0), "seconds   - Finished to bb number", bb_next)
print("            Number remaining:", len(remaining))

print(f"Loop lasted {(time() - t0) // 60} minutes ")


# In[ ]:


# Show the dfbb
dfbb.head(5)


# In[ ]:


dfbb.tail(5)


# [](http://) ## Look at the Final [Primes every 10th city] Path

# In[ ]:


# Assemble the path from the dfbb
# Start with zeroth bb (NP usually)
path_df = [backbone[0]]
# add the rest
for ibb in range(1,len(backbone)):
    path_df += cities_from_str(dfbb.loc[ibb,'CityListStr'])
    path_df.append(backbone[ibb])
##path_df


# In[ ]:


# Show the whole path
plot_path(path_df, coordinates)


# In[ ]:


print("Number of cities in the path:", len(path_df), ";    Number left to add:", len(remaining))


# In[ ]:


# Show the Rudolph Score results  *** Got All the Carrots ! ***
score, carrots, length = usual_score(path_df, coordinates, prime_flags)
# and without going back to the NP
score_noNP, dummy1, dummy2 = usual_score(path_df[:-1], coordinates, prime_flags)

print("Rudolph Score:", int(score), "   Carrots:", carrots, "   Length:", int(length), ".\n" +
      " Penalty frac:", int(10000*(score-length)/length)/100,
      "%   Final step to NP has distance ", int(score - score_noNP))


# In[ ]:


# The length from the dataframe values
dfbb['Length'].sum()


# In[ ]:


# Save the dataframe with the assigned cities.
# Use ";" as separation character since I use commas in CityListStr.
dfbb.to_csv("df"+bb_name+"_v8x_filled.csv", sep=";")


# ## Look at how the insert method does within a segment
# Look at the backbone-to-backbone path at a selected segment

# In[ ]:


# Look at particular segment of the bb, bb_last to bb_next
if True:
    bb_next = 106

    
    # For testing, etc.
    # Start with empty segment for testing...
    ##reset_segments()
    ##print("")
    #
    # For testing, add to the segment
    ##print("Added", add_to_segment(bb_next, 2, mpdist_limit=1.e6), " Remaining cities:", len(remaining))
    #   Segment from,to : 196117 88513 
    #        has path:  [142958, 80942]
    #   Segment from,to : 196117 88513 
    #      has path:  [85808, 142958, 80942, 81345]
    ##print("")
    ##print("Added", add_to_segment(bb_next, 4, mpdist_limit=1.e6), " Remaining cities:", len(remaining))
    
    seg_cities = cities_from_str(dfbb.loc[bb_next,'CityListStr'])
    print("Segment from,to :", backbone[bb_next - 1], backbone[bb_next], 
         "\n   has path: ", seg_cities)

    segment_path = [backbone[bb_next - 1]] + seg_cities + [backbone[bb_next]]


# In[ ]:


plot_path(segment_path, coordinates, np_star=False, end_stars=True, prime_vals=prime_flags)
# Total length of these:
seg_len = 0.0
for iseg in range(1,len(segment_path)):
    seg_len += np.linalg.norm(df_cities.loc[segment_path[iseg-1]][['X','Y']] - 
                              df_cities.loc[segment_path[iseg]][['X','Y']], ord=2)
seg_len


# In[ ]:


# closeup of the backbone near the NP
n_close = 30
plot_path(backbone[-1*n_close:]+backbone[0:n_close+1], coordinates, prime_vals=prime_flags)


# In[ ]:


# closeup of the path near the NP
plot_path(path_df[-10*n_close:-1]+path_df[0:10*n_close], coordinates, prime_vals=prime_flags)


# ## Looking at Cities not (yet) in the backbone (if any)
# If the above path only includes the NP+prime backbone cities and 9 additional segment cities for each segment (including to final NP), then total number of cities in the path is: `10*17802 + 9+1`. In this case that leaves 19740 cities not yet in the path.  Those are shown here along with the backbone cities.

# In[ ]:


# Show the remaining cities (if any) and ALL the backbone cities
if len(remaining) > 0:
    plt.figure(figsize=(15, 10))
    plt.scatter(df_cities.loc[remaining, "X"], df_cities.loc[remaining, "Y"], s=1, c="green", alpha=0.5)
    # Backbone cities:
    plt.scatter(df_cities.loc[backbone, "X"], df_cities.loc[backbone, "Y"], s=1, c="red", alpha=0.2)
    # North pole
    plt.scatter(df_cities.iloc[0: 1, 1], df_cities.iloc[0: 1, 2], s=30, c="black")
    plt.grid(False)
    # remind what the backbone file is
    print("Backbone (red) cities are from file: ",bb_name)
    print("Remaining (green) cities will be saved and put in TSP order by themselves.")
    plt.show()


# ## Choose more backbone cities 

# In[ ]:


if len(remaining) > 0:
    print("There are", len(remaining), "cities remaining to add to the path;")
    n_add_bb = len(remaining)//10
    n_at_end = len(remaining) % 10
    print("add them with", n_add_bb, "additional cities in the backbone.")
else:
    print("  ***  All done - the path is complete.  ***  ")


# In[ ]:


if len(remaining) > 0:
    
    # All remaining cities:
    remaining_cities = pd.DataFrame({"CityId" : remaining})
    # Write them out to a file...
    remaining_cities.to_csv("remaining_cities_v8x.csv", index=None)

    # OK, will put these remaining cities in their own TSP path...


# ## Generate output file to submit

# In[ ]:


# Output the path to a file that we can submit (This does include the final NP !)
if len(remaining) == 0:
    submission = pd.DataFrame({"Path": path_df})
    submission.to_csv("submission.csv", index=None)
    print("Complete path, submission file written.")
else:
    print("  ***  Not a complete path, so no submission.  ***  ")


# ## Look at the BB efficiency with plots

# In[ ]:


dfbb.head(5)


# In[ ]:


dfbb.iloc[1:].plot.scatter("BBtoBBlen","Length",s=3, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's actual Path-Length vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()


# In[ ]:


dfbb.iloc[1:].plot.scatter("BBtoBBlen","LenAlone",s=3, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's Alone-Length vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()


# In[ ]:


dfbb.iloc[1:].plot.scatter("BBtoBBlen","MaxRadius",s=5, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's actual MaxRadius vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()


# In[ ]:


dfbb.iloc[1:].plot.scatter("BBtoBBlen","RadAlone",s=5, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's Alone-Radius vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()


# In[ ]:


# Compare 
pd.plotting.scatter_matrix(dfbb[['BBtoBBlen','LenAlone','Length','RadAlone','MaxRadius']], figsize=(14,14),
                           diagonal='hist', range_padding=0.05, hist_kwds={'bins':100},
                           grid=True, alpha=0.5)
plt.show()


# 

# ## Final version 8:
# # Combine the All-Primes path with TSP of Remaining cities

# In[ ]:


# This cell is run after the file:
#     remaining_v8C_t2500rand
# has been generated by "Concorde - Primes only" from the file created above.
if True:

    # Instead of doing all the processing above, can read in the result of the above processing,
    # will not use this when doing the Kaggle "commit"
    if False:
        dfbb = pd.read_csv("../input/santa-18-primes-tsp60/dfbb_primesTSP5000_v7B_filled.csv",
                           sep=";", index_col=0)
        # the backbone cites in this df:
        backbone = list(dfbb['CityId'])
        # Assemble the path from the dfbb
        # Start with zeroth bb (NP usually)
        path_df = [backbone[0]]
        # add the rest
        for ibb in range(1,len(backbone)):
            path_df += cities_from_str(dfbb.loc[ibb,'CityListStr'])
            path_df.append(backbone[ibb])

        
    # Get the remaining cities in TSP order, NP + 19740 others
    dftsp = pd.read_csv("../input/santa-18-primes-tsp60/remaining_v8C_t3000rand.csv")
    # Generate this backbone list of CityIds
    # Starts at the NP and ends at final city in path (close to NP).
    path_remain = list(dftsp.loc[list(dftsp['Path']),'CityId'])

    
    # Combine the two paths: Remaining path and then the Primes path.
    # The NP starts the remaining path, so skip the NP that starts the primes path
    full_path = path_remain + path_df[1:]
 
    # and zero-out remaining
    remaining = []

    print("Number of cities in the path:", len(full_path), ";    Number left to add:", len(remaining))
    
    # and output it
    submission = pd.DataFrame({"Path": full_path})
    submission.to_csv("submission.csv", index=None)
    print("Complete path, submission file written.")

    # Show the Rudolph Score results  *** Got All the Carrots ! ***
    score, carrots, length = usual_score(full_path, coordinates, prime_flags)
    # and without going back to the NP
    score_noNP, dummy1, dummy2 = usual_score(full_path[:-1], coordinates, prime_flags)

    print("Rudolph Score:", int(score), "   Carrots:", carrots, "   Length:", int(length), ".\n" +
          " Penalty frac:", int(10000*(score-length)/length)/100,
          "%   Final step to NP has distance ", int(score - score_noNP))
    
    # Show the whole combined path
    plot_path(full_path, coordinates)


# In[ ]:


# This is the final proposed path.


# ### Summary of versions and outputs:
# * **version 1**:<br>
# `Number of cities in the path: 178021 ;    Number left to add: 19749` <br>
# `Rudolph Score: [4036831]    Carrots: 17802    Length: 4036831 .` <br>
# ` Penalty frac: 0.0 %   Final step to NP has distance  22` <br>
# * **version 2**:<br>
# `Number of cities in the path: 178030 ;    Number left to add: 19740` <br>
# `Rudolph Score: [4036261]    Carrots: 17802    Length: 4036261 .` <br>
# ` Penalty frac: 0.0 %   Final step to NP has distance  22` <br>
# * pre version 3 - B, out of the v2 box:<br>
# `Number of cities in the path: 197770 ;    Number left to add: 0` <br>
# `Rudolph Score: 8284825    Carrots: 17802    Length: 8266128 .` <br>
# ` Penalty frac: 0.22 %   Final step to NP has distance  58` <br>
# Submitting csv gave score: 8284873.4 - corrected my scoring, was't including final NP, now get: <br>
# `Rudolph Score: 8284873    Carrots: 17802    Length: 8266176 .` <br>
# ` Penalty frac: 0.22 %   Final step to NP has distance  47` <br>
# Using the 5000 second TSP backbone: <br>
# `Rudolph Score: 7760343    Carrots: 17802    Length: 7744692 .` <br>
# ` Penalty frac: 0.2 %   Final step to NP has distance  20` <br>
# * **version 3**: <br>
# Demonstrate optimizing within a segment using (kludgily coded) insertion-addition method. <br>
# `Rudolph Score: 7760343    Carrots: 17802    Length: 7744692 .` <br>
# ` Penalty frac: 0.2 %   Final step to NP has distance  20` <br>
# * pre version 4 - A, using insertion ordering on previous backbone: <br>
# `Rudolph Score: 4960633    Carrots: 17802    Length: 4944794 .` <br>
# ` Penalty frac: 0.32 %   Final step to NP has distance  30` <br>
# 4-B Use just the primes-only (t5000) TSP backbone and get: <br>
# `Rudolph Score: [2284014]    Carrots: 17802    Length: 2284014 .` <br>
# ` Penalty frac: 0.0 %   Final step to NP has distance  11` <br>
# This suggests that including the additional 19740 cities, if they had similar efficiency, would give a length increased by a factor (19740+178030)/(178030), or 1.111, for a total length about 2537265.<br>
# 4-C Well that wasn't as expected (with usual `n_end_portion = int(0.05*len_bb)`): <br>
# `Rudolph Score: 4734260    Carrots: 17802    Length: 4719618 .` <br>
# ` Penalty frac: 0.31 %   Final step to NP has distance  8` <br>
# * **version 4**: <br>
# A little better, but many large MaxRadius values above 1000 ...with `n_end_portion = int(0.5*len_bb)`: <br>
# `Rudolph Score: 4310788    Carrots: 17802    Length: 4299253 .` <br>
# ` Penalty frac: 0.26 %   Final step to NP has distance  8 ` <br>
# * pre version 5 - A, B  <br>
# Lengths for not-full backbones (going from bb 1 to last one):<br>
# `remaining bb (mod 1): Backbone length is  122266; Rudolph Score:  246535` <br>
# `remaining bb (mod 6): Backbone length is  121372; Rudolph Score:  244658` <br>
# `NP+Primes-only bb   : Backbone length is  351826; Rudolph Score: 2285676` <br>
# Full backbones: <br>
# `Full bb, rand remain: Backbone length is  381843; Rudolph Score: 5180089` <br>
# `Full bb, 1/10thTSP  : Backbone length is  387598; Rudolph Score: 4959111` <br>
# Look at effect of including fewer cities (i.e., some left over)<br>
# `Full1/10 4 between  : Backbone length is  387598; Rudolph Score:  952928  ( 98895 cities)` <br>
# `Full1/10 7 between  : Backbone length is  387598; Rudolph Score: 1675916  (158220 cities)` <br>
# `Full1/10 8 between  : Backbone length is  387598; Rudolph Score: 2040849  (177995 cities)` <br>
# `Full1/10 9 between  : Backbone length is  387598; Rudolph Score: 4959111  (197770 cities)` <br>
# * pre version 5 - C Add 4,7,8, then to 9 cities per segment in bb order... hmmm, maybe greedy is good? <br>
# `Rudolph Score: 7666203    Carrots: 17802    Length: 7661590 ` <br>
# ` Penalty frac: 0.06 %   Final step to NP has distance  8` <br>
# D - function allows segments to be populated in any order and incrementally...
# * **version 5**: <br>
# Assign cities to the Odd then the Even backbone segments - does better than doing them sequentially (less segment-segment competition for cities, in a good way...) <br>
# `Rudolph Score: 3834178    Carrots: 17802    Length: 3824060 ` <br>
# ` Penalty frac: 0.26 %   Final step to NP has distance  4478 ` <br>
# * pre version 6 - A, add dfbb columns of LenAlone and RadAlone to characterize the segments; save the dfbb to csv for further use. <br>
# B - Keep track in dfbb of the number of cities assigned to each bb city - can select on this to avoid ones that are finished...
# * **version 6**: <br>
# First populate the prime segments (odd, even), then populate the non-prime segments, mimicing how the backbone cities were selected: <br>
# `Rudolph Score: 2993490    Carrots: 17802    Length: 2983638 ` <br>
# ` Penalty frac: 0.33 %   Final step to NP has distance  8 ` <br>
# * pre version 7 - A, Slight improvement by doing the "compact" odd/even before the rest of odd/even for primes/not-primes:<br>
# `Rudolph Score: 2868849    Carrots: 17802    Length: 2861015 ` <br>
# ` Penalty frac: 0.27 %   Final step to NP has distance  8 ` <br>
# B, Use just the primes-only TSP-5000 backbone and assign cities as in 7A; this gives a no-penalty primes path:<br>
# `Rudolph Score: [2151297]    Carrots: 17802    Length: 2151297 ` <br>
# ` Penalty frac: 0.0 %   Final step to NP has distance  9 ` <br>
# and leaves 19740 remaining (non primes) cities.
# * **version 7**: <br>
# Combine the no-penalty Prime path with the TSP'ed Remaining cities to create a full path with lower score.<br>
# `Rudolph Score: 2377086    Carrots: 17802    Length: 2374798 ` <br>
# ` Penalty frac: 0.09 %   Final step to NP has distance  9 ` <br>
# * pre version 8 - C, Improved adding cities to prime-backbone segments; the changes give a no-penalty primes path:<br>
# `Rudolph Score: [1982465]    Carrots: 17802    Length: 1982465 ` <br>
# ` Penalty frac: 0.0 %   Final step to NP has distance  9 ` <br>
# * **version 8**: <br>
#  `Rudolph Score: 2208262    Carrots: 17802    Length: 2206101 ` <br>
# ` Penalty frac: 0.09 %   Final step to NP has distance  9 ` <br>

# In[ ]:




