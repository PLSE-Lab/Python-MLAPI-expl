#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


state_data = pd.read_csv("../input/murder-rates-by-states/state.csv")


# Lets start by getting some information about the dataset using the info() method

# In[ ]:


state_data.info()


# We will start with Percentiles
# 
# **Percentiles** are valuable to summarize the entire distribution. Here we use the quantile method from pandas to display some percentiles of murder rate by state.Percentiles are also valuable to summarize the entire distribution. Here we use the quantile method from pandas to display some percentiles of murder rate by state.

# In[ ]:


state_data['Murder.Rate'].quantile([.05, .25, .5, .75, .95])


# As you can see, for 0.50 percentile, the murder rate is 4, and since 50 pencentile essentially mean median, the median is 4 murders per 100,000 people.

# Now, lets move on to our first tool in exploring this data.
# 
# **Boxplots**, 
# They based on percentiles and give a quick way to visualize the distribution of data. In python, this task can be accomplished using Pandas, Seaborn or Matplotlib. Lets see how.

# **Using Pandas**

# In[ ]:


state_data.Population = state_data.Population/1000000
state_data.boxplot(column = ['Population'], )


# The top and bottom of the box are the 75th and 25th percentiles, respectively. The dashed lines, referred to as whiskers, extend from the top and bottom to indicate the range for the bulk of the data. Any data outside of the whiskers is plotted as single points.

# **Using Seaborn**

# In[ ]:


import seaborn as sns
ax = sns.boxplot(y = state_data["Population"])


# You can altenatively use Matplotlib for the same purpose

#  **Frequency Table and Histograms**

# In[ ]:


state_data.Population = state_data.Population * 1000000
pd.DataFrame(pd.Series.value_counts(state_data.Population, bins = 11))


# The above code uses Value count to publish the population frequency within different ranges. The bin parameter in values helps divide the range into a given number of input(11 in this case).

# Histogram can be plotted using either Matplotlib or Pandas library. Lets see how both of them perform for the Column Murder.Rate

# In[ ]:


from matplotlib import pyplot as plt
plt.hist(state_data['Murder.Rate'], color = 'blue', edgecolor = 'black')


# In[ ]:


state_data['Murder.Rate'].hist()
state_data.hist()


# In case of Matplotlib, you need to specify which Numeric columns you want to plot the histogram of. Which is not the case with pandas. For a full description and parameters list, you can use the following links:
# 
# * Pandas: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
# * Matplotlib: https://pythonspot.com/matplotlib-histogram/

# **Density Plots:**
# A density plot is a smoothed, continuous version of a histogram estimated from the data. The most common form of estimation is known as kernel density estimation. In this method, a continuous curve (the kernel) is drawn at every individual data point and all of these curves are then added together to make a single smooth density estimation.

# In[ ]:


import seaborn as sns
sns.distplot(state_data['Murder.Rate'], hist=True, kde=True, color = 'blue')

