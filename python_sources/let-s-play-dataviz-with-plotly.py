#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to see what can be done with [plotly](https://plot.ly/python/), as I am a beginner with this library.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/hostel-world-dataset/Hostel.csv')
df


# In[ ]:


print(len(df))
print(df.isna().sum())
df.describe()


# In[ ]:


for idx, row in df.iterrows():
    c = 0
    if "km from city centre" not in row['Distance']:
        c += 1
print(c)


# Let's transform the values in the "Distance" column in digital values.

# In[ ]:


dist = []
for d in df['Distance']:
    dist.append(d.replace('km from city centre',''))
dist[:5]


# In[ ]:


del df['Distance']
df['Distance'] = dist
df.head(5)


# In[ ]:


set(df['rating.band'])


# In[ ]:


fabulous = df[df['rating.band'] == 'Fabulous']
good = df[df['rating.band'] == 'Good']
rating = df[df['rating.band'] == 'Rating']
superb = df[df['rating.band'] == 'Superb']
verygood = df[df['rating.band'] == 'Very Good']


# ## Graphs

# Let's start with some boxplot to see if the "rating.band" value depends on the summary score.

# In[ ]:


fig = px.box(df, x="rating.band", y="summary.score")
fig.show()


# As we can see, there is no overlap between the boxes, meaning that rating band is a traduction for the summary score in natural language.

# In[ ]:


roten = []
for idx, row in df.iterrows():
    if row['rating.band'] is np.nan:
        roten.append(idx)
df_corr = df.drop(roten)


# Now, let's look at the number of hostels in each category.

# In[ ]:


import plotly.graph_objects as go
X = ('Rating', 'Good', 'Very Good', 'Fabulous', 'Superb')
#X = set(df_corr["rating.band"])
Y = [len(df_corr[df_corr["rating.band"] == x]) for x in X]
fig = go.Figure([
    go.Bar(x=X, y=Y)
])

fig.show()


# Most of the hostels are high rated. The better a category, the more hostels it contains.

# Does the quality of a hostel depends on its distance to the center. Let's look at it.

# In[ ]:


fig = px.scatter(df_corr, x="Distance", y="summary.score", color="rating.band")
fig.show()


# Most of the hostels are close to the center of the city. And so there are hostels of each type of quality in such areas. But it is also interesting to note that poorly rated hostels ('Rating'), are mainly close to the center. On the other hand, hostels far from the center are all rated as superb.
# So maybe, in the center, there are so many customers that, even being bad, you can have some, whereas far from the center, you have to be good to survive.
# Or maybe customers far from the center are in a better mood (holidays vs work)  and so give better notes to the hostel they are standing.

# Now, let's look if rating are quite equivalent in each category for a given hostel. We will focus on "atmosphere", "facilities" and "staff" values and plot them in a 3D scatter.

# In[ ]:


fig = px.scatter_3d(df_corr, x="atmosphere", y="facilities", z="staff", color="rating.band")
fig.show()


# I suggest you turn and zoom on the plot to have a better understanding of it. If the rating were equivalent, points would placed along a line. This is almost the case on best scores, but not for the "Rating" hostels. Some look very bad rated, but there is also a point, for example, where staff has the best rate (10) whereas facilities and atmosphere received a 2. There are some other points from other categories, lost in the plan where staff = 10, that indicate hostels where staff was really appreciated while the "material" part of the hostel was more disappointing.

# ### Conclusion on 1st part
# 
# I am quite impressed by how easy it is to plot with plotly. All of my plots requiring only columns of the dataset were done in one line (plus fig.show()). The possibility to assign the columns directly to the different aspects of figure is pretty convenient. It is a bit longer when you have to transform some of the data beforehand, but it is still OK. I appreciate to have info when I pass the mouse pointer on the figure.

# ## Maps
# 
# I am a great fan of folium. But plotly also have function for creating maps, so I am going to try them right now.

# In[ ]:


roten = []
for idx, row in df_corr.iterrows():
    if row['lat'] is np.nan or row['lon'] is np.nan:
        roten.append(idx)
df_corr2 = df_corr.drop(roten)


# In[ ]:


fig = px.scatter_geo(df_corr2, lat='lat', lon='lon', hover_name='hostel.name', color='rating.band', projection="natural earth", center={'lat':35.42, 'lon':139.43}, scope='asia')
fig.show()


# Well, the map is absolutely ugly, and really not convenient, as the scope is limited to a continental view. So a manual zoom on Japan is required.

# In[ ]:


fig = px.scatter_mapbox(df_corr2, lat='lat', lon='lon', hover_name='hostel.name', color='rating.band', zoom=15)
fig.show()


# No matter how hard I try, the map does not appear, only a kind of legend of the colors.
# Maybe I should use a [graph object](https://plot.ly/python/scattermapbox/#multiple-markers), but it does not seem that simple, which was a very good point of plotly.

# ## Conclusion
# 
# Plotly appears to be a powerful tool, that shows very good plots in a very easy way. It is probably the best we can imagine to visualize a dataset. By contrast, I am not convinced at all by the map systems, that looks really poor compared to specialized libraries. So, I will keep using folium to draw my maps.
