#!/usr/bin/env python
# coding: utf-8

# Let's import the necessary packages

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt  # importamos pyplot para hacer graficas
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(['fivethirtyeight'])
mpl.rcParams['lines.linewidth'] = 3


# Reading the dataset. We parse the 'Date' column.

# In[ ]:


df = pd.read_csv('../input/comptagesvelo2015.csv', header=0,
                 sep=',', parse_dates=['Date'], dayfirst=True,
                 index_col='Date')


# Let's inspect the data we just imported and get an idea of the fields and size.

# In[ ]:


df.head()


# In[ ]:


df.shape


# We notice a column that should not be included, so we remove it using the method 'drop'.

# In[ ]:


df = df.drop('Unnamed: 1', 1)


# In[ ]:


df.head(2)


# Let's plot one of the records to get an idea.

# In[ ]:


df['Berri1'].plot(figsize=(12,6))


# We can see the rise of activity in the warmer months of the year. Montreal is a cold city!. Let's try now to plot all of the lanes together and add a legend to the plot.

# In[ ]:


df.plot(figsize=(12,6))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# We can't really get much information from this graph. Let's see if we can get better insights by using a boxplot.

# In[ ]:


df.plot(kind='box', sym='gD', vert=False, xlim=(-100,11000))


# We notice some outliers. Some really high values out of nowhere and some lanes that it's range of values don't include low values, meaning they should have regular activity throughout the whole year (like Saint-Laurent U-Zelt Test). That seems odd so let's take a closer look at those cases.

# In[ ]:


df['Saint-Laurent U-Zelt Test'].plot(figsize=(12,6))


# As we could have guessed from the name, this lane seems to be undergoing a test so it only has activity recorded since the end of September. Let's take a look at the rest of the lanes that could have a similar problem.

# In[ ]:


df[['Saint-Laurent U-Zelt Test','Maisonneuve_1','Pont_Jacques_Cartier','Parc U-Zelt Test']].plot(figsize=(12,6))


# As we can see they only have info recorded for shorts periods of time. We would have to investigate. Now, let's look at the big outlier in the 'Maisonneuve_3' lane.

# In[ ]:


df[['Maisonneuve_3']].plot(figsize=(15,6))


# Let's filter that specific event 

# In[ ]:


df[df['Maisonneuve_3'] > 6000]


# So, it looks like either there was a massive event, like a race of some sort, or some sort of technical problem. We would have to investigate that too, but for now let's get ride of that lane, and of all the lanes that contain missing values, and move on with our overview.

# In[ ]:


clean_df = df.dropna(axis=1, how='any')
clean_df = clean_df.drop('Maisonneuve_3',1)


# This is it. Just a little short inspection of the dataset. I hope some of the techniques used in here are of use to somebody. Comments are very welcome. Enjoy! 
