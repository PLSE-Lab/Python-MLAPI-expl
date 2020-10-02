#!/usr/bin/env python
# coding: utf-8

# Guys I have just started with data science.
# Please upvote, this will encourage me.

# # Covid 19 Old data

# imports

# In[ ]:


# imports
# for visualising data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Working for numericals
import pandas as pd
import numpy as np


# Arranging data to be plotted.

# In[ ]:


# Data to plot
df = pd.read_csv('../input/cv.csv')
labels = df['State/UT']
sizes = df['Confirmed']
sizes_rec = df['Recovered']
sizes_dis = df['Deceased']
ind = np.arange(29)    # the x locations for the groups
width = 0.5      # the width of the bars: can also be len(x) sequence


# Visualising The data

# In[ ]:


p1 = plt.barh(ind, sizes, width)
p2 = plt.barh(ind, sizes_rec, width, left=0)
p3 = plt.barh(ind, sizes_dis, width, left=sizes_rec)
plt.ylabel('Data')
plt.xlabel('States')
plt.title('Stats of India (COVID-19)')
plt.text
plt.yticks(ind, labels)
plt.xticks(np.arange(0, 400, 20))
plt.legend((p1[0], p2[0], p3[3]), ('Active', 'Recovered', 'Deceased/Dead'))

plt.show()


# ![wadn](https://github.com/pranayteaches/stacked_bar_corona/blob/master/Figure_1.png?raw=true)

# Hope you enjoyed my first notebook.
# It's a small notebook but an upvote will encourage me.
# Thank you!!!
