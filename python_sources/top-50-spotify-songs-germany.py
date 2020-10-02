#!/usr/bin/env python
# coding: utf-8

# I gonna make a short overview about the spotify data. Also i wanna know what are the things, which make songs popular. I start with an analysis on german songs, since i'm german and interested ;) Let's see if i continue by comparing those on eu and worlwide level.

# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#load data and glance on them 
data = pd.read_csv("../input/top-50-spotify-songs-by-each-country/top50contry.csv", encoding="iso-8859-1",index_col=0)

print(data.head(1),"\n\n",data.shape)


# In[ ]:


data_germany = data[data.country == "germany"]

print("\n",data_germany.head(1))


# In[ ]:


#sort for popularity
data_popularity = data_germany.sort_values(by=["pop"], ascending = False)

data_popularity[0:9]


# so let's try to find some information, i want to know which genre is the most popular. since there are not much samples it's not that meaningful ;)
# 
# 
# 

# In[ ]:


#calculate the average of the popularity and the size of each genre 
genre_popularity = data_germany.groupby("top genre").agg([np.mean, np.size])["pop"]

#sort
genre_popularity = genre_popularity.sort_values(by=["mean"], ascending = False)


print(genre_popularity)


# I did not knew so much pop and rap genres exists, well you can check them by yourself. Let's do the same with the artist!

# In[ ]:


#calculate the average of the popularity and the size of each genre 
artist_popularity = data_germany.groupby("artist").agg([np.mean, np.size])["pop"]

#sort
artist_popularity = artist_popularity.sort_values(by=["mean"], ascending = False)

print(artist_popularity)


# In[ ]:


#Alright this shows

#to be detailed the top 50 hast a mean of:
mu, std = norm.fit(data["pop"])

print("\b","mean:",mu,"\n","standard deviation:",std)

#if we take the average of the popularity the average musician we are nearly close.
mu, std = norm.fit(artist_popularity["mean"])

print("\n","mean:",mu,"\n","standard deviation:",std)


# In[ ]:


#let's plot the popularity by artist.
value_to_plot = artist_popularity["mean"]
plt.hist(value_to_plot, density=True)
plt.plot(value_to_plot, norm.pdf(value_to_plot, mu, std))
plt.title('german musicians popularity')
plt.show()


# after a short overview let's check if some of the variables are correlating the popularity, a heatmap shows a visualized overview of the correlating variables

# In[ ]:





correlations = data_germany.corr()

ax = sns.heatmap(
    correlations, 
    vmin=-1, vmax=1, center=0,
    square=True,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_title("correlation matrix of the variables (germany)")


# In[ ]:


print(correlations["pop"][:-1])


# Germans seem to dislike speechness and danceable music. But those are just slight correlations. I guess older music is less popular since it can't get a "trending" effect anymore. 

# Alright let's get to the big stage: the World!

# In[ ]:


#calculate the average of the popularity and the size of each genre 
genre_popularity_worlwide = data.groupby("top genre").agg([np.mean, np.size])["pop"]

#sort
genre_popularity_worlwide = genre_popularity_worlwide.sort_values(by=["mean"], ascending = False)

print(genre_popularity_worlwide.head(30))


# Apperently the australians are pretty popular on christmans. It seems they got multiple first places, since the average is so high. Let's check which song won our hearts. (Also this should be number one of the "world" argument of the countries.

# In[ ]:


data[data['top genre']=='australian pop'].groupby("title").size()


# THE WINNER IIIIIIS: Dance Monkey from Tones and I
# 
# 
# 
# I also want to know if everyone "dislikes" speechness like the germans apparently do. Let's do the correlation on worlwide level.

# In[ ]:


print(data.corr()["pop"][:-1])


# Wow, we got some different results. Speechness does not matter in average, but the duration is correlating negativly. Make short songs ;)!
# 
# 
# Let's check how we germans decide our favourite songs compared worlwide.

# In[ ]:


#let's plot the popularity distribution by country.
country_data = data.groupby("country").agg([np.mean])["pop"]

value_to_plot = country_data["mean"]
mu, std =  norm.fit(value_to_plot)
print ("average worlwide: ", mu, " standard deviation worldwide: ", std)

plt.hist(value_to_plot, density=True)
plt.plot(value_to_plot, norm.pdf(value_to_plot, mu, std))
plt.title('worlwide musicians popularity')
plt.show()


# Our average Popularity is 81, which is clearly the average worlwide. Our standard deviation is much higher, this means we are a more picky with our songs. (At least at xmas time) Also sadly i have no idea what went wrong by plotting the normal distribution. If someone knows there mistake feel free to tell me.
