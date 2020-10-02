#!/usr/bin/env python
# coding: utf-8

# In this notebook I intend to practice my newly acquired insights into seaborn visualization. I am training Data Science, Python, ML, AI at the moment in the Udemy course: https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp. For maybe other learners or those that want to get a first insight into seaborn I will document, as I go, step-by-step, my process to get insights into the dataset and meaningful visualisations.
# 
# I would be very thankful for any hints for improvement you may give me.
# 
# So let's dive into music streaming:
# ![](https://www.top5z.com/wp-content/uploads/2017/04/xmusic-streaming-apps.jpg.pagespeed.ic.w45ndw6BtU.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Reading the dataset into a pandas DataFrame

# In[ ]:


data=pd.read_csv("../input/data.csv",encoding = 'utf8')


# So let's get first insights into the data at hand:

# In[ ]:


data.info()


# In[ ]:


# First line makes the float numbers readable (not e^something)
pd.options.display.float_format = '{:,.2f}'.format
data.describe()


# In[ ]:


data.head()


# **Observations:**
# * We have the first 200 positions in the streaming charts per region.
# * Besides position only # of streams as quantitative position, will depend strongly on region, i.e. potential audience in region which will depend on # of people in given region and market penetration of spotify
# 
# **Questions: Still descriptive statistics**
# * How many regions?
# * Do we have a time series per region?

# In[ ]:


data["Region"].nunique()


# In[ ]:


data["Date"].nunique()


# 45 regions and 226 time points per region. Time Series is nothing I have worked myself into as of now.
# 
# **Questions:**
# * Size of regions
# * How correlated are regions to each other in their taste?
# * Distribution of views over the 200 places

# In[ ]:


maxi_views=data.groupby("Region").describe().Streams["max"].sort_values(ascending=False)
maxi_views.head(10)


# Actually interesting if these were the same songs

# In[ ]:


data[(data["Streams"].isin(maxi_views.head(10)))&(data["Position"]==1)].sort_values("Streams", ascending=False)


# Well,... Despacito it is:
# ![](https://instrumentalfx.co/wp-content/uploads/2017/09/Luis-Fonsi-Daddy-Yankee-%E2%80%93-Despacito-ft.-Justin-Bieber-Instrumental-mp3-300x300.jpg)
# 
# This leads me to a hypothesis to be proven:
# 
# **Hypothesis 1:**
# * Streaming over time increases

# In[ ]:


type(data["Date"][0])


# In[ ]:


data["Date"][20000]


# What is actually the time horizon of the dataset?

# In[ ]:


data["Date"].min()


# In[ ]:


data["Date"].max()


# Ok. For this short time horizon (6 month) I do not expect streaming to increase. Seasonality overrides underlying trend. But lets see.

# **Learning**:
# 
# When trying to plot at first arrange the data which you want to plot in a variable, then try to plot this. Otherwise, as I now painfully learned, you don't know whether your error lies with how you try to plot or how you try to arrange your data.

# In[ ]:


# For proper plotting time needs to be in a non-string format
from datetime import datetime

data["Date2"]=data["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[ ]:


type(data["Date2"][10])


# In[ ]:


data["Date2"][10]


# In[ ]:


global_total_streams=data[(data["Region"]=="global")].groupby("Date2").sum().drop("Position",axis=1)
global_total_streams.head()


# In[ ]:


plt.plot(global_total_streams)


# **Observations:**
# 
# * Peaks and Lows surely have to do with weekends
# * Declining trend in summer month. Due to vacation?
# 
# ![](http://raverrafting.com/wp-content/uploads/weekend_music.jpg)
# 
# Now back to original questions:
# 1) Size of regions
# 2) Correlation between regions

# In[ ]:


total_views_regions=data.groupby("Region").sum().drop("Position",axis=1).drop("global",axis=0)
total_views_regions.head()


# In[ ]:


regions=data["Region"].unique()
regions


# In[ ]:


ax=sns.distplot(total_views_regions)
ax.set(xlabel="Total Streams per Region")


# US is a clear outlier. Lets have a closer look at the smaller regions.

# In[ ]:


total_views_regions_wo_US=data.groupby("Region").sum().drop("Position",axis=1).drop(["global","us"],axis=0)


# In[ ]:


ax=sns.distplot(total_views_regions_wo_US, bins=20)
ax.set(xlabel="Total Streams",ylabel="% of regions within respective cluster of streams")


# Lots of very small regions. "Long-Tail"-logic., 10-15 core markets.
# 
# Speaking of long-tail: Lets quickly check Hypothesis 2: That the # of streams in any given region between 1-200 follows a long-tail logic.

# In[ ]:


streams_per_position=data.groupby("Position").sum()
streams_per_position.head()


# In[ ]:


ax=sns.distplot(streams_per_position)
ax.set(xlabel="Total Streams",ylabel="% of positions within respective cluster of streams")


# Beautiful. Exactly as expected
# ![](http://www.tenerifenews.com/wp-content/uploads/2014/02/p-18-liberty.jpg)
# 
# Now lets see if regions are correlated. For this I need some quantitative measure. Qualitative base would is always the title. As base quantitative measure I have place and stream per day.
# 
# Idea 1: Compare relative stream-weight of titles
# 
# Idea 2: Compare difference in places (The smaller the more correlated)

# In[ ]:


#Group by titles per region to have one list of titles per region
tracks_per_region=data.groupby(["Region","Track Name"]).sum()
tracks_per_region.head()


# In[ ]:


tracks_per_region=tracks_per_region["Streams"]
tracks_per_region.head()


# In[ ]:


region_matrix=pd.DataFrame(index=regions,columns=regions)


# In[ ]:


for region1 in region_matrix.index:
    for region2 in region_matrix.index:
        count=0
        for element in tracks_per_region[region1].index:
            if element in tracks_per_region[region2].index:
                count+=1
        region_matrix[region1][region2]=count
region_matrix


# In[ ]:


region_matrix = region_matrix.astype(int)


# In[ ]:


sns.heatmap(region_matrix)


# Looks like we have a couple of strong outliers (black rows/columns) and some highlights of very similar markets (bright dots). Let's see what the clustermap of seaborn makes out of it.

# In[ ]:


sns.clustermap(region_matrix)


# Maybe let's do all this analysis for the Core Markets

# In[ ]:


core_markets=total_views_regions.sort_values("Streams",ascending=False)[0:14].index


# In[ ]:


core_markets


# In[ ]:


region_matrix_core=region_matrix[list(core_markets)]


# In[ ]:


region_matrix_core=region_matrix_core.loc[list(core_markets)]


# In[ ]:


sns.heatmap(region_matrix_core)


# In[ ]:


#Normalizing the Dataset to abstract from size
region_matrix_core_norm=(region_matrix_core-region_matrix_core.mean())/region_matrix_core.std()


# In[ ]:


sns.heatmap(region_matrix_core_norm)


# **Observations**
# * "Ar" is very different from many. Except Mx (Mexico) and "ES" (Spain). Seems to be a Hispanic cluster. Lets check on clustermaps later
# * "Us" and "Ca" are most similar

# In[ ]:


sns.clustermap(region_matrix_core_norm)


# **Observations**
# * Very clearly we have a hispanic cluster with Spain, Mexiko and Argentina
# ![](http://www.institutohispania.com/uploads/images/Spanish-Speaking-Countries.jpg)
# * Brazil is clustered into a different cluster. right next to the hispanic cluster but still.
# * North America with US and Canada stick out, next comes the other english-speaking countries Great Britain and Australia
# * Philippines next closes to English-native cluster (I wasn't aware but now I know: Official language in Philippines is English)
# ![](https://image.slidesharecdn.com/english-speaking-countries-miriam-mascaras4703/95/english-speaking-countries-miriam-mascaras-1-728.jpg)
# * Italy and France on the one side and Germany, Sweden and Netherlands as European cluster on the other side

# Next really interesting question is of course how do the songs spread?
# 
# **Assumption**
# * Songs spread from the US throughout the world (as indicated by clustermap at first through English-speaking countries, then rest of Europe and Philippines before penetrating Hispanic cluster)
# 
# **To be tested:**
# * Is there a similar pattern within the spanish-speaking world?

# 
