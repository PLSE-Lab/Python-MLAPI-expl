#!/usr/bin/env python
# coding: utf-8

# Hey everyone! This is my first kernel so any suggestions would be very helpful.
# 
# I've done some analysis on horror movies to test some common assumptions about them. Here is what I found. Enjoy!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Import other libraries
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/movie_metadata.csv')

#Some column info for reference
data.info()


# Now, I want to first check the general notion I hear often. Which is, Horror movies are poorly made compared to other genres and thus, should be rated badly. We will base this claim on the IMDB score.

# In[ ]:


#Gives us all the horror films (or films that contain horror elements)
df_horror = data.loc[data['genres'].str.contains('Horror')]


# In[ ]:


sns.set(style="whitegrid")


f, ax = plt.subplots(figsize=(9, 6))

sns.set_color_codes("pastel")
sns.barplot(x="title_year", y="imdb_score", data=data,
            label="All Movies", color="b")

sns.set_color_codes("muted")
sns.barplot(x="title_year", y="imdb_score", data=df_horror,
            label="Horror Movies", color="b")

ax.legend(loc="lower right", frameon=True)
ticks = plt.setp(ax.get_xticklabels(),rotation=90)


# So for nearly every year, horror movies score worse on average than all movies in the dataset. Although these ratings don't necessarily speak to a movie's production quality, horror movies tend to get a worse rating overall. Now lets compare how successful horror movies are in terms of earnings. For this, we will check how many movie flops there are compared to all movies.

# In[ ]:


#Lets compare how many flops horror movies received compared to all movies.
num_flops = data[data['budget'] > data['gross']]
num_horror_flops = df_horror[df_horror['budget'] > df_horror['gross']]


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2)

labels = ['Flops', 'Successful']
colors = ['red', 'green']
total_flops = [len(num_flops.index), str((len(data.index) - len(num_flops.index)))]
patches, texts, autotext = ax1.pie(total_flops, colors=colors, autopct='%1.1f%%',startangle=90)
ax1.set_title('Total Box Office Flops')
ax1.legend(patches, labels, loc="best")
ax1.axis('equal')

horror_flops = [len(num_horror_flops.index), str((len(df_horror.index) - len(num_horror_flops.index)))]
patches, texts, autotext = ax2.pie(horror_flops, colors=colors, autopct='%1.1f%%',startangle=90)
ax2.set_title('Horror Film Box Office Flops')
ax2.legend(patches, labels, loc="best")
ax2.axis('equal')


# Interesting to see that Horror films have more success at the box office compared to all movies in the dataset. One explanation is that many horror movies take place in smaller sets, which would lower production costs.
# 
# While we're at it, let's get a visual of the production costs.

# In[ ]:




df_budget = data
df_budget['genre_horror'] = np.where(data['genres'].str.contains('Horror'), 'Horror', 'All')
#Removes korean film that has a budget that is not converted to dollars.
df_budget = df_budget[df_budget['budget'] < 10000000]

ax = sns.stripplot(x="genre_horror", y="budget", data=df_budget, jitter=True)
ax.set(xlabel='Genre', ylabel='Production Budget', ylim=(0,None))
ax.yaxis.set_major_formatter(FormatStrFormatter('$' + '%.2f'))

