#!/usr/bin/env python
# coding: utf-8

# # Fast Food in America EDA
# #### Or More Aptly, How Does Ohio Compare to the Typical
# 
# Lets explore some patterns in fast food in the USA. 
# (And get some experience with Seaborn --- I typically use ggplot in R for EDA.)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 

import os

#import and split out to US only
df = pd.read_csv("../input/FastFoodRestaurants.csv")

df.head()


# ## Plots
# 
# Lets first look at what fast food chains America seems to like the most.

# In[ ]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9

sns.barplot(x=df.name.value_counts().index, y=df.name.value_counts(),
           order = df.name.value_counts().iloc[:10].index)


# How about top states for fast food?

# In[ ]:


sns.barplot(x=df.province.value_counts().index, y=df.province.value_counts(),
           order = df.province.value_counts().iloc[:10].index)


# First two are not too much of a surprise given their populations in comparison to other states.. Ohio (my home state) is a surprise.
# 
# Does ohio tend to the typical fast foods of the nation?

# In[ ]:


ohio_subset = df[df['province'] == 'OH']

sns.barplot(x=ohio_subset.name.value_counts().index, y=ohio_subset.name.value_counts(),
           order = ohio_subset.name.value_counts().iloc[:10].index)


# ### What's up with Gold Star Chili!?
# 
# SO the top 5 or so fast food restaurants in the state are pretty proportional to the country as a whole. Not too surprisingly there is a jump of Gold Star Chili into the top 10. Additionally, subway is exchanged for jimmy johns, and white castle also makes an appearence while sonic and Dominos disapear. 
# 
# Let look more into this Ohio Chili relationship.

# In[ ]:


GSC_sub = df[df['name'] == 'Gold Star Chili']
GSC_OH_sub = GSC_sub[GSC_sub['province'] == 'OH']

GSC_count = GSC_sub.name.count()
GSC_OH_count = GSC_OH_sub.name.count()

print(float((GSC_OH_count)/(GSC_count)))

sns.barplot(x=GSC_sub.province.value_counts().index, y=GSC_sub.province.value_counts())


# So as I expected Gold Star Chili is contained in a very small radius.. Two states total. Impressive that a local chain such as this can make it into the top 10 for the state of Ohio. 
# 
# Lets prep for one more.. possibly weird.. chart.

# In[ ]:


ohio_id_subset = ohio_subset.copy()
ohio_id_subset['gsc_id'] = ohio_id_subset['name'].apply(lambda i: 1 if i == 'Gold Star Chili' else 0)

ohio_id_subset['gsc_id'].unique()


# OK  this might not work out how I'm hoping.. Lets see if a lat/long plane with restaurants plotted over it make sense.. Color identifying if its a Gold Star or not..  I'm hoping it gets close to mapping it onto a grid of Ohio coordinates.  I don't think Seaborn has any in house geo plotting functions?
# 
# NOTE: I'm not too much of a chili fan.. I actually don't even eat meat.. Just an interesting abnormality in the data I noticed and dove into ^.^

# In[ ]:


lm = sns.lmplot(x="latitude", y="longitude", data=ohio_id_subset, hue = 'gsc_id',fit_reg = False,
          legend = False, markers=["o", "x"])
axes = lm.axes
axes[0, 0].set_title('Coords of Ohio Fast Food Restaurants - (Gold Star in Orange)')
axes[0,0].set_ylim(-86, -79,)


# I think it turned out okay!  We can see relatively easily that Gold Star is a big thing in Cincinatti and thats about it.  
# 
# ## Waffle House
# I have to say, while I'm not a chili fan, I'm a die hard Waffle House consumer..  
#  
#  (I ran out of time.. I'll explore WH asap!
