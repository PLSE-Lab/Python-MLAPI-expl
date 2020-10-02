#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import folium

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Missing values

# In[ ]:


df = pd.read_csv('/kaggle/input/egyptianpyramids/pyramids.csv')
df.isna().sum()


# Here, we can see that there are some missing values, especially in height, slope and volume. Intuitively, we would think of building as much of them as possible using some basic maths. The formula of the volume *V* of a pyramid is *V = (B . h) / 3*, with *B* the base area, and *h* the height of the pyramid. But, let's first check if this rule is followed by the pyramids whose data is complete.

# In[ ]:


df_full = df.loc[:,['Base1 (m)', 'Base2 (m)', 'Height (m)', 'Slope (dec degr)', 'Volume (cu.m)', 'Type']].dropna()
diff = []
for idx, row in df_full.iterrows():
    exp_vol = row['Base1 (m)'] * row['Base1 (m)'] * row['Height (m)'] / 3.
    diff_to_exp = (abs(float(row['Volume (cu.m)'].replace('.','')) - exp_vol))
    diff.append(diff_to_exp / exp_vol)      # We work with ratios
fig, ax = plt.subplots()
ax.set_xlabel('Difference with expected value')
ax.set_ylabel('Count')
plt.hist(diff, range = (0, max(diff)), bins = 10)


# In[ ]:


print(len(df_full))


# Most of the volumes are close to what we were expecting. Nonetheless, there are 5 out 27 of them that are totally different. This is pretty disturbing, because we can not be sure that filling mathematically the NaN fields would be a good idea. We miss the information on how these volumes were obtained, and the method is likely to differ from a pyramid to another. One thing we can do, is to check if such differences are bound to the type of pyramid.

# In[ ]:


for idx, row in df_full.iterrows():
    exp_vol = row['Base1 (m)'] * row['Base1 (m)'] * row['Height (m)'] / 3.
    diff_to_exp = (abs(float(row['Volume (cu.m)'].replace('.','')) - exp_vol))
    dif = diff_to_exp / exp_vol 
    print(row['Type']+' ->  '+str(dif)) 


# Every type of pyramid has some volumes close from what was expected, and some volumes far from what was expected. 'True' pyramids seem to be the closest from what was expected, but even with them, we have some 20% - 50% "errors". It is almost certain that the methodology to obtain the volume varies from a pyramid to another, the two pyramids with a perfect 0.0 difference had probably their volume computed mathematically (but such a perfection is very unlikely to be found in real life), and those with high "errors" were probably measured with other media (maybe, e.g., taking only the cavities into account, which could explain high differences). So, it does not seem pertinent to fill the voids with the math approach. We will assume that, taking into account the inconsistency in the methodology to obtain them, volume values are not reliable. And thus, we will part them in the further analysis.

# ## Height over time

# In this section, we will focus on the evolution of the height of pyramids over time. In the dataset, time is expressed in terms of pharaoh's dynasties.

# In[ ]:


print(set(df['Dynasty']))


# To be perfectly accurate, we would need to find the exact dates of each dynasty. But, it is not mandatory. Dynasties are classified in their order of apparition, so we can easily sort them chronogically.

# In[ ]:


dfh = df[df['Height (m)'] > 0.]
dyn = set(df['Dynasty'])
h_by_dyn = {}
for d in dyn:
    h_by_dyn[d] = dfh['Height (m)'][dfh['Dynasty'] == d]
    print(str(d)+' -> '+str(len(h_by_dyn[d])))


# We can see that there are more pyramids in the eldest dynasties.

# In[ ]:


fig, ax = plt.subplots()
ax.boxplot(h_by_dyn.values(), dyn)
ax.set_xlabel('Dynasties')
ax.set_ylabel('Height (m)')


# No clear rule appear there. In the eldest dynasties, with more pyramids, there is a high variety of heights. Anyway, the very highest pyramids were all built in the ancient times.

# ## Impact of the location
# 
# In the following section, we will focus on how the location of a pyramid impacts its characteristics. First, let's build a map of the dataset's pyramids.

# In[ ]:


m = folium.Map(location=[28.04, 30.71], zoom_start=7, tiles='Stamen Terrain')
for idx, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']]).add_to(m)
m


# On the above map, we can see that the pyramids are distributed along the Nile river. Nile is approximatively following a meridian, so they are almost all at the same longitude. Thus, we will focus on latitude, and see if we can discover some schemes based on it.

# In[ ]:


lat = []
h = []
for idx, row in df.iterrows():
    if row['Height (m)'] > 0:
        h.append(row['Height (m)'])
        lat.append(row['Latitude'])
fig, ax = plt.subplots()
ax.set_xlabel('Latitude')
ax.set_ylabel("Pyramid's height")
ax.scatter(lat, h)


# This graph shows clearly that the very high pyramids are all concentrated at the highest latitudes (i.e. in the north of Egypt, known as *Lower* Egypt). There are small pyramids everywhere, but in the south (*High* Egypt), there are only small pyramids.
# Let's now do the same experience with the slope.

# In[ ]:


lat = []
sl = []
for idx, row in df.iterrows():
    if row['Slope (dec degr)'] > 0:
        sl.append(row['Slope (dec degr)'])
        lat.append(row['Latitude'])
fig, ax = plt.subplots()
ax.set_xlabel('Latitude')
ax.set_ylabel("Pyramid's slope")
ax.scatter(lat, sl)


# This new graph seems to be the exact opposite of the previous one. Lowest slopes are only in the north, whereas high slopes are present everywhere. So the temptation is high to check for a correlation between height and slope.

# In[ ]:


h = []
sl = []
for idx, row in df.iterrows():
    if row['Slope (dec degr)'] > 0 and row['Height (m)'] > 0:
        sl.append(row['Slope (dec degr)'])
        h.append(row['Height (m)'])
fig, ax = plt.subplots()
ax.set_ylabel("Pyramid's height")
ax.set_xlabel("Pyramid's slope")
ax.scatter(sl, h)

Here, one thing appears, there is no pyramid combining high height and high slope (at least in the dataset). High height implies a low slope, and high slope implies a short height. Low slopes are acceptable for any height.
# ## Conclusion
# 
# This dataset in the only one about Egyptian pyramids I found on kaggle at the time I started this kernel. It does not include all pyramids, and some data is missing. Nonetheless, it is complete enough to find very interesting insights. In this short exploration, we saw that the highest pyramids were built during the eldest ages, and that they specifically belongs to the northern area of Egypt. We also saw that all configurations height/slope are not allowed (probably for physical reasons). So there is a lot to learn from such a dataset. There are also interrogations that remain, for example the strange values we observed about the volumes of the pyramids. Maybe an egyptologist would be able to explain them, which could open new perspectives on the dataset. So, as last words, I would like to quote the author of the dataset:

# *DEAR EGYPTOLOGISTS, PLEASE HELP ME UPDATE THE DATASET IF YOU HAVE MORE INFO (let's include temples also)*

# I totally agree with that. Historical datasets are really hard to obtain, because they require strong expertise to be built. Maybe that's why there are so few of them (combined with no direct economic benefits). That is a pity, because I am deeply convinced that, working with experts, data science and data analysis have a lot to bring to those fields of knowledge.
