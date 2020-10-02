#!/usr/bin/env python
# coding: utf-8

# # Cuisine popularity
# 
# The most obvious measure of popularity of each cuisine type is the number of restaurants which serve that type of food.

# In[ ]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import operator

# Pull out the fields we care about
data = pd.read_csv("../input/zomato.csv")[['rate', 'votes', 'cuisines']]

# Find all the types of cuisine
cuisines = {}
for i, j in data.iterrows():
    
    # Each restaurant can have multiple types of cuisine
    types = [x.strip() for x in str(j[2]).split(',')]
    for cuisine in types:
        cuisines[cuisine] = cuisines[cuisine] + 1 if cuisine in cuisines.keys() else 1

# Bar chart!
x_pos = np.arange(len(cuisines))
plt.bar(x_pos, list(cuisines.values()))
plt.show()


# Looks like the data has some sparsely used tags. Let's take the 40 most-used.

# In[ ]:


sorted_cuisines = sorted(cuisines.items(), key=operator.itemgetter(1), reverse=True)
sorted_cuisines = sorted_cuisines[:40]

# Bar chart!
plt.subplots(figsize=(20, 6.4))
x_pos = np.arange(len(sorted_cuisines))
plt.bar(x_pos, [x[1] for x in sorted_cuisines], width=0.8)
plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)
plt.show()


# As we might expect, Bangalore has a lot of Indian restaurants. The other options are pretty varied with consistent representation.
# 
# Average review scores could be another obvious definition of popularity:

# In[ ]:


# New counters for the same types
cuisine_ratings = {}
for cuisine in sorted_cuisines:
    cuisine_ratings[cuisine[0]] = 0
    
# Count em up
for i, j in data.iterrows():
    
    # Parse the rating - invalid ratings get mean score which could affect emerging cuisines
    try:
        rating = float(str(j[0])[:str(j[0]).find('/')])
    except ValueError:
        rating = 2.5
    
    # Each restaurant can have multiple types of cuisine
    types = [x.strip() for x in str(j[2]).split(',')]
    for cuisine in types:
        if cuisine in cuisine_ratings.keys():
            cuisine_ratings[cuisine] += rating
            
# Average em out
for cuisine in sorted_cuisines:
    cuisine_ratings[cuisine[0]] /= cuisine[1]
    
# Bar chart!
plt.subplots(figsize=(20, 6.4))
x_pos = np.arange(len(sorted_cuisines))
plt.bar(x_pos, [cuisine_ratings[y] - 3 for y in [x[0] for x in sorted_cuisines]], width=0.8, bottom=3)
plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)
plt.show()


# The ratings are pretty favorable, so let's baseline it to 3 stars in order to force it to be more significant.
# 
# Notice the slight increasing trend in average review scores as the restaurant count decreases. Survival of the fittest? Or something else?
# 
# We can think of the average review score as the total "review points" normalized by the number of reviewers. Let's multiply the reviewer counts back in to the the raw points as another possible popularity definition. First, the reviewer counts themselves:

# In[ ]:


# New counters for the same types
cuisine_votes = {}
for cuisine in sorted_cuisines:
    cuisine_votes[cuisine[0]] = 0
    
# Count em up
for i, j in data.iterrows():
    
    # Each restaurant can have multiple types of cuisine
    types = [x.strip() for x in str(j[2]).split(',')]
    for cuisine in types:
        if cuisine in cuisine_votes.keys():
            cuisine_votes[cuisine] += j[1]

# Bar chart!
plt.subplots(figsize=(20, 6.4))
x_pos = np.arange(len(sorted_cuisines))
plt.bar(x_pos, [cuisine_votes[y] for y in [x[0] for x in sorted_cuisines]], width=0.8)
plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)
plt.show()


# Looks like our original trend makes another showing with some exceptions. Seems people are less likely to review fast food restaurants, but more likely to review "continental" for example. 
# 
# However, let's be careful about offering this as a potential popularity definition. It's more like a "view count" on a web page, and doesn't necessarily give an indication of favoribility. That's why reviews have scores in the first place! Multiplying these in as discussed:

# In[ ]:


# Bar chart!
plt.subplots(figsize=(20, 6.4))
x_pos = np.arange(len(sorted_cuisines))
plt.bar(x_pos, [(cuisine_ratings[y] - 3) * cuisine_votes[y] for y in [x[0] for x in sorted_cuisines]], width=0.8)
plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)
plt.show()


# Here's a potential definiton of cuisine popularity. We can see the large number of reviews of North Indian cuisine are weighted down by the review score, putting continental cuisine into first place. There's also a solid showing by cafes, Italian and American cuisines.
# 
# But how about restaurant popularity? There were a large number of North Indian restaurants, so it's suspicious that our review-count based score isn't heavily favoring cuisines with more options for reviewing. Let's normalize by restaurant count:

# In[ ]:


# Bar chart!
plt.subplots(figsize=(20, 6.4))
x_pos = np.arange(len(sorted_cuisines))
plt.bar(x_pos, [((cuisine_ratings[y] - 3) * cuisine_votes[y])/cuisines[y] for y in [x[0] for x in sorted_cuisines]], width=0.8)
plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)
plt.show()


# This exaggerates the slight upward trend in review score that was noted before. 
# 
# In general, it seems folks are more likely to rate (and give favorable ratings) towards european and american-style restaurants. This likely reflects some bias where people are more likely to remember and rate foreign experiences than the typical options.
