#!/usr/bin/env python
# coding: utf-8

# ## Property Loss and Tornado Analysis by State##
# This analysis aims to look at total property loss and tornado counts by state from 1996 through 2015

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# We start by importing the **Tornadoes_SPC_1950to2015.csv** file and slice out the data items we need

# In[ ]:


df = pd.read_csv('../input/Tornadoes_SPC_1950to2015.csv')
tonadoes_1996 = df[df['yr'] >= 1996][['st','loss']]


# Let's take a look at the first 10 rows

# In[ ]:


tonadoes_1996.head(10)


# Now we will index and aggregate the data by state, providing a count of the n umber of tornadoes and a sum of the total property damage

# In[ ]:


tonadoes_1996_damage = pd.DataFrame({'count': tonadoes_1996.groupby(['st'])['loss'].count(), 'total_loss': tonadoes_1996.groupby(['st'])['loss'].sum()})


# Looking at the data, we see

# In[ ]:


tonadoes_1996_damage


# Now I'll plot this data to see what we have

# In[ ]:


#Separate the dataframe into series
count = tonadoes_1996_damage['count'] 
loss = tonadoes_1996_damage['total_loss']

#Find the state with maximum number of tornadoes and then get the index label and plot points
max_tornado_count = tonadoes_1996_damage['count'].max()
max_tornado_cnt_label = tonadoes_1996_damage[tonadoes_1996_damage['count'] == max_tornado_count].index.tolist()[0]
max_tornado_cnt_x = tonadoes_1996_damage[tonadoes_1996_damage['count'] == max_tornado_count]['count']
max_tornado_cnt_y = tonadoes_1996_damage[tonadoes_1996_damage['count'] == max_tornado_count]['total_loss']

#Find the state with maximum amount of property damage and then get the index label and plot points
max_tornado_loss = tonadoes_1996_damage['total_loss'].max()
max_tornado_loss_label = tonadoes_1996_damage[tonadoes_1996_damage['total_loss'] == max_tornado_loss].index.tolist()[0]
max_tornado_loss_x = tonadoes_1996_damage[tonadoes_1996_damage['total_loss'] == max_tornado_loss]['count']
max_tornado_loss_y = tonadoes_1996_damage[tonadoes_1996_damage['total_loss'] == max_tornado_loss]['total_loss']

#Prepare our plot
colors = np.random.rand(51)
area = count
plt.scatter(count, loss,s=area,c=colors,alpha=.5)

#Provide axis labels and a title
xlab = "Number of Tornadoes [in thousands]"
ylab = "Total Loss [in million USD]"
title = "Total Property Loss Since 1996"

plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)

#set the axis limits
plt.xlim(0, 3500)
plt.ylim(0, 6000)

#Apply grid lines for good measure
plt.grid(True)

#Plot the max values for count and loss
plt.text(max_tornado_cnt_x, max_tornado_cnt_y, max_tornado_cnt_label)
plt.text(max_tornado_loss_x, max_tornado_loss_y, max_tornado_loss_label)

plt.show()


# ## More to Come
# I plan to develop this more. Things I would like to do
# 
# - Change the scale of the plot to spread things out a bit
# - Possibly show two plots based on the count of tornadoes
# - Couple this data with median income levels of counties and do some analysis at the county
#    level
