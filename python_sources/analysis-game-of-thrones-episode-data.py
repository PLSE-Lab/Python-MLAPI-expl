#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Exploring episodes data for Game of Thrones

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. 

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})


# There is 1 csv file in the current version of the dataset but it doesn't include the last seson data, so added that dataset
# 

# In[ ]:


print(os.listdir('../input/game-of-thrones-episode-data-full'))
df_ep = pd.read_csv('../input/game-of-thrones-episode-data-full/got_csv_full.csv')
df_ep.head(10)
print('Number of episodes in the dataset : ' , len(df_ep))


# In[ ]:


df_ep_clean = pd.read_csv('../input/game-of-thrones-episode-data-cleaned/got_csv_full_clean.csv')
df_ep_clean.head(10)
print('Number of episodes in the dataset : ' , len(df_ep))


# In[ ]:


#plotPerColumnDistribution(df_ep, 10, 5)
print(df_ep.dtypes)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})

import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.figure_factory as ff
import cufflinks as cf


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df_ep, 10, 5)


# ## Death Count Percentage

# In[ ]:


number_of_deaths_in_category = df_ep['Notable_Death_Count'].value_counts().sort_values(ascending=True)

data = [go.Pie(
        labels = number_of_deaths_in_category.index,
        values = number_of_deaths_in_category.values,
        hoverinfo = 'label+value'
    
)]

plotly.offline.iplot(data, filename='Notable_Death_Count')


# > Most episodes have either 2 (19.2%, 14 episodes) or 4 o1 (17.8%, 13 episodes) notable deaths

# ## Average Rating of Episodes
# 
# Do any episodes perform really bad or really good?

# In[ ]:


data = [go.Histogram(
        x = df_ep.Imdb_Rating,
        xbins = {'start': 1, 'size':0.5, 'end' :10}
)]

print('Average episode rating = ', np.mean(df_ep['Imdb_Rating']))
plotly.offline.iplot(data, filename='overall_rating_distribution')


# ## Which Season is most Popular?

# In[ ]:


sns.set_style("darkgrid")
ax = sns.jointplot(df_ep['Season'], df_ep['Imdb_Rating'])


# > Season 1 has most consistent ratings, Season 3 and 6 have highest ratings overall

# ## Notable Death - Impacts Ratings?
# 
# How do notable death count impact the episodes ratings

# In[ ]:


sns.set_style("darkgrid")
ax = sns.jointplot(df_ep['Notable_Death_Count'], df_ep['Imdb_Rating'])


# Most top rated episodes have optimally sized deaths 0 to 4 - neither too less nor too much.

#  ## Is there are link between Ratings and Viewer figures?

# In[ ]:


sns.set_style("darkgrid")
ax = sns.jointplot(df_ep['US_viewers_million'], df_ep['Imdb_Rating'])


# > Consistent ratings until ~8 million viewers

# ## Another way to look at Same Analysis of Viewers vs. Ratings

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
p = sns.stripplot(x="Imdb_Rating", y="US_viewers_million", data=df_ep, jitter=True, linewidth=1)
title = ax.set_title('Viewers vs. Ratings')


# ## Do certain Writers and Directors make better episodes?

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
p = sns.stripplot(x="Imdb_Rating", y="Writer", data=df_ep_clean, jitter=True, linewidth=1)
title = ax.set_title('Writers vs. Ratings')


# In[ ]:


#df_ep_clean_1 = df_ep_clean
#df_ep_clean_1['Writer'] = df_ep_clean['Writer'].apply(lambda x: x.replace(' ', ' ') if ',' in str(x) else x)


# ## Do certain Directors make better Episodes?

# In[ ]:



fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
p = sns.stripplot(x="Imdb_Rating", y="Director", data=df_ep_clean, jitter=True, linewidth=1)
title = ax.set_title('Directors vs. Ratings')


# ## Viewers and Ratings by Writers and IMDB Voters

# In[ ]:


#!pip install bubbly


# In[ ]:


from __future__ import division
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
from bubbly.bubbly import bubbleplot


# In[ ]:


# Adding Year column
df_ep_clean['Original_Air_Date'] = pd.to_datetime(df_ep_clean['Original_Air_Date'], format='%B %d, %Y')
df_ep_clean['Original_Air_Year'] = df_ep_clean['Original_Air_Date'].dt.year # df_ep_clean['year']


# In[ ]:


df_ep_clean.head(10)


# In[ ]:


figure = bubbleplot(dataset=df_ep_clean, x_column='US_viewers_million', y_column='Imdb_Rating'
                   ,     bubble_column='Season' 
                  ,  time_column='Original_Air_Year'
                    , size_column='IMDB_votes'
                  #  , color_column='Writer'
                     , color_column='Number_in_Season'
                       ,x_title="Viewers (millions)", y_title="IMDB Ratings", title='Viewers and Ratings by IMDB Voters over Years',
    x_logscale=True, scale_bubble=3, height=650, show_colorbar=True)

iplot(figure, config={'scrollzoom': True})


# ## Conclusion
# This concludes my inital analysis! 
