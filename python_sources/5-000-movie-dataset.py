#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # for visuals
sn.set(style="white", color_codes = True) #customizes graphs
import matplotlib.pyplot as mp #for visuals
get_ipython().run_line_magic('matplotlib', 'inline')
#how graphs are printed out
import warnings #suppress certain warnings from libraries
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

movie = pd.read_csv("../input/movie_metadata.csv", sep=",", header=0)


# In[3]:


print(movie.shape)


# In[4]:


movie.head(10)


# ## Questions to Consider ##
#  1. Will the budget be higher if a specific director is involved?
#  2. Does the duration of the film have a correlation to the IMDB score?
#  3. Is the gross higher if there are multiple actors faces on the poster?

# In[5]:


movie.corr()


#  1. There is no correlation between the director and the budget of the film
#  2. The duration of a film has a small correlation to the IMDB score. Could this be because it keeps people from watching it? Does the duration mean some people are less likely to want to watch the film?
#  3. There is no correlation between the amount of faces on a poster and the film gross.
# 

# Under correlation, if a number is negative, it implies that as the x value increases, the y value will decrease. There are no numbers about that are negative which would imply that as the column subject increases, the row subject decreases.

# In[6]:


correlation = movie.corr()
mp.figure(figsize = (10,10))
sn.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')

mp.title("Correlation between Movie Info")


# Other information that can be taken from the above correlation table is that as the number of critic reviews increases, the higher the gross of the film. This demonstrates that there is a strong correlation between the two bits of data.

# Let's now take a look at which plot keywords were used most in the reviews.

# In[7]:


plot_keywords = movie.plot_keywords.map(lambda x:str(x).split(","))
empty_array = []
for i in plot_keywords:
    empty_array = np.append(empty_array, i)


# In[8]:


from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=['title']).generate(" ".join(empty_array))
mp.figure(figsize=(20, 15))
mp.imshow(cloud)
mp.axis('off')
mp.show()


# From the above image, we can conclude that words such as "friend", "love", "murder" were used more than words such as "student" or "violence".

# Let's take a look at how the budget correlates specifically to the gross.

# In[9]:


correlation["budget"].corr(correlation["gross"])


# We can conclude from the above amount, that there is no correlation. Had the amount been higher, we could have assumed that as the budget increased, the gross decreased.

# ## Total Gross of Director's Films ##
# Let us now take a look at how each director's film gross.

# In[14]:


chart = movie.groupby(["director_name"])['gross'].count()
ax=mp.xticks(rotation=90)
chart.plot()

