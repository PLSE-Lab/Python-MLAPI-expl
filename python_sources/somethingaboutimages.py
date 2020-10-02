#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from scipy.stats import itemfreq
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns # visualizations


# # This Notebook is an exploration of the ["Painter By Numbers" dataset](https://www.kaggle.com/c/painter-by-numbers)
# 
# # Observations
# 1. This dataset has a lot of images
# 
# # Questions / Hypotheses
# 1. Is it possible to identify the artist of a painting by features related to the painting? 
#   * Would the following features be helpful?
#     * color palette
#     * dominant color
#     * etc ...

# In[ ]:


df = pd.read_csv("../input/train_info.csv")
df[df['title'].notnull()]


# In[ ]:


# https://www.kaggle.com/getting-started/39426
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install webcolors')


# In[ ]:


# https://stackoverflow.com/a/9694246/5411712
import webcolors


# In[ ]:


# This code helps identify color names

# https://stackoverflow.com/a/9694246/5411712
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


# # Here is 1 image from the training set

# In[ ]:


img = cv2.imread('../input/train_2/2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)


# # Let's find the ***Color Palette*** of the image, and the ***Dominiant Color***
# Resource: https://stackoverflow.com/q/43111029/5411712

# In[ ]:


# average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
#===============
arr = np.float32(img)
pixels = arr.reshape((-1, 3))

n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS
_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

palette = np.uint8(centroids)
quantized = palette[labels.flatten()]
quantized = quantized.reshape(img.shape)
#===============
dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
# `itemfreq` is deprecated and will be removed in a future version. Use instead `np.unique(..., return_counts=True)`


# In[ ]:


actual_name, closest_name = get_colour_name(dominant_color)

print("palette = ")
palette_names = []
for color in palette:
    meh, name = get_colour_name(color)
    palette_names.append(name)
    print(str(color) + " " + str(name))
print('\n\n')

dc = dominant_color

print("dominant_color (rgb)\t=\t" + str(dominant_color))
print("dominant_color (name)\t=\t" + str(closest_name))
print("dominant_color (hex)\t=\t" + str('#%02x%02x%02x' % (dc[0], dc[1], dc[2])))


# # SO [***goldenrod***](https://www.google.com/search?q=goldenrod+color) is the dominant color of the image ```../input/train_2/2.jpg```
# 
# #### a limitation of this technique/code is that ```#d2d43b``` is not necessarily the same as ```goldenrod```, but it's close enough
# #### maybe just the hex code is enough/better, but using color names would help reduce the feature size. 

# In[ ]:


plt.imshow(img)
plt.show()


# # Here are all the image files and their features
# ### * displaying just the first 10 with ```df.head(10)```

# In[ ]:


df = pd.read_csv("../input/train_info.csv")
df.head(10)


# # Let's look at the features related to ```../input/train_2/2.jpg```

# In[ ]:


df[(df['filename'] == '2.jpg')]


# # just two lists of all the ***unique*** ```styles``` and ```genres```

# In[ ]:


#sns.boxplot(x='style', y='idk', data=df)

df['style']

styles = []

for s in df['style']:
    if (s not in styles):
        styles += [s]

print("\nSTYLES\n")
print(styles)

genres = []

for g in df['genre']:
    if (g not in genres):
        genres += [g]

print("\nGENRES\n")
print(genres)


# # The artist names are hashed... werid, but how many artists are there? Also how many images are there per artist? 

# In[ ]:


artists = {} # holds artist hash & the count
for a in df['artist']:
    if (a not in artists):
        artists[a] = 1
    else:
        artists[a] += 1

print("\nHow many artists? \n")
print(len(artists))

# convert unique hashes to unique numbers 
# conversion helps matplotlib actually plot
new_dict = {} 
i = 1
for each in artists:
    new_dict[i] = artists[each]
    i += 1

lists = sorted(new_dict.items(), key=lambda kv: kv[1], reverse=True) # sorted by value, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.plot(y, x, 'r.')
plt.title(s='Plot of Number of Images vs Artist ID')
plt.ylabel(s='Artist ID (not hash but is unique number related to the hash)')
plt.xlabel(s='Number of Images')
plt.show()


# # How many artists have over 300 images? 

# In[ ]:


over_200 = 0
over_300 = 0
over_400 = 0
under_100 = 0
under_50 = 0
under_25 = 0
under_5 = 0
under_1 = 0

for a in artists:
    over_200 = over_200 + 1 if artists[a] >= 200 else over_200
    over_300 = over_300 + 1 if artists[a] >= 300 else over_300
    over_400 = over_400 + 1 if artists[a] >= 400 else over_400
    under_100 = under_100 + 1 if artists[a] <= 100 else under_100
    under_50 = under_50 + 1 if artists[a] <= 50 else under_50
    under_25 = under_25 + 1 if artists[a] <= 25 else under_25
    under_5 = under_5 + 1 if artists[a] <= 5 else under_5
    under_1 = under_1 + 1 if artists[a] <= 1 else under_1

print("OVER?")
print("over_200 = " + str(over_200))
print("over_300 = " + str(over_300))
print("over_400 = " + str(over_400))

print("UNDER?")
print("under_100 = " + str(under_100))
print("under_50 = " + str(under_50))
print("under_25 = " + str(under_25))
print("under_5 = " + str(under_5))
print("under_1 = " + str(under_1))


# # some stats plots to show the same info

# In[ ]:


# https://stackoverflow.com/a/20026448/5411712
import scipy.stats as stats
fit = stats.norm.pdf(y, np.mean(y), np.std(y))  #this is a fitting indeed
plt.plot(y,fit,'-o')
plt.hist(y, normed=True)
plt.show()

# https://stackoverflow.com/a/15419072/5411712
values, base = np.histogram(y, bins=40)
cumulative = np.cumsum(values)
plt.plot(base[:-1], cumulative, '-o', c='blue')
plt.plot(base[:-1], len(y)-cumulative, '-o', c='green')
plt.hist(y, color='orange')
plt.show()

