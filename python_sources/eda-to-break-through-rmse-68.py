#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/aDVymEW.jpg" width="550" height="350"/>
# ## Intro
# This notebook explores the dynamics of single intersections. I like to take this sort of micro-level look to learn about the data and get ideas for feature engineering. The potential features might be derived from the given data or they might require data from other sources. Using such features and/or ones like them will help improve model performance.
# 
# Here's a quick look at the data with the target-like columns separated.

# In[ ]:


import numpy as np
import pandas as pd
import hvplot.pandas
import geoviews as gv
import holoviews as hv
hv.extension('bokeh')


# In[ ]:


train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv', 
                    index_col='RowId')
test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv', 
                    index_col='RowId')
target_cols = [col for col in train.columns.tolist() if
                    col not in test.columns.tolist()]
target_df = train[(target_cols)]
train = train.drop(target_cols, axis=1)


# In[ ]:


tt = pd.concat([train,test], keys=['train', 'test'], sort=False)

if tt.columns[-1:][0] == 'City': #Move city column to where it belongs
    ttcols = tt.columns.tolist() 
    ttcols_moved = ttcols[-1:] + ttcols[:-1] 
    tt = tt[ttcols_moved].reset_index(level=0)
display(tt.head(), target_df.head())
#del train, test


# In[ ]:


# Optional function to change column name format from CamelCase to snake_case
def snakify(camel_list):
    snake_list = []
    for c in camel_list:
        underscored = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', c)
        refined = re.sub('([a-z0-9])([A-Z])', r'\1_\2', underscored).lower()
        snake_list.append(refined)
    return snake_list

# tt.columns = snakify(tt.columns)


# ## Area Recon
# 
# 

# #### Example 1- Boston
# Let's first look at Boston from a high-flying drone. Here are the intersections contained in the data. Even without looking at target distributions we can guess that the times and distances are quite spread across a city, especially at the \_p80 end. EDIT: Train and test geographies are similar, but the points are not evenly distributed. For instance, only test intersections extend north into Somerville.
# 
# Technical note: I used [GeoViews](http://geoviews.org/) to make the graph below. It's an easy way to make scalable, interactive geo plots.
# 
# Style note: Sorry about the confusing colors. The darker orange is for intersections seen in both train and test sets.

# In[ ]:


tt_bos = tt[tt.City == 'Boston'].drop_duplicates(['level_0','IntersectionId'])

points_bos = gv.Points(tt_bos, kdims=['Longitude', 'Latitude'],
                      vdims=['level_0']).opts(color='level_0', cmap=['dodgerblue', 
                      'darkorange'], width=500, height=450, alpha=0.5)

# points_bos_train = gv.Points(tt_bos[tt_bos.level_0=='train'], kdims=['Longitude', 'Latitude'], 
#                       vdims=['level_0']).opts(color='dodgerblue', width=500, height=450, 
#                       fill_alpha=0.1, line_width=1.5, size=3)

# points_bos_test = gv.Points(tt_bos[tt_bos.level_0=='test'], kdims=['Longitude', 'Latitude'], 
#                       vdims=['level_0']).opts(color='darkorange', width=500, height=450, 
#                       line_alpha=0.1, size=3)

tiles = gv.tile_sources.CartoLight()
display(points_bos * tiles)


# Let's pick a single intersection and bring the drone in close. I think of intersections as a collection of related traffic paths passing through a common point. In this dataset, the 'Paths' feature lists the unique paths at each intersection. Each traffic path has it's own dynamics and can affect other paths. There may be an advantage to stacking and/or using the target variables that aren't part of the LB score.
# 
# We'll first look at the intersection of Land Blvd and Cambridgeside Place, not far from where I lived several years ago. This area has intersections from both train and test. I pulled the street view image manually from Google Maps vs. using their wonderful APIs.
# 
# <img src="https://i.imgur.com/buKIbkr.png" width="350" height="350"/>
# <img src="https://i.imgur.com/PIbDbtJ.png" width="550" height="350"/>
# 
# Here are some things to notice and potential features that might help:
#   - It's a T-intersection: it can have up to 6 paths plus any u-turns.
#   - One street is a major artery connecting two bridges: the road name ends in Boulevard
#   - The other street is a small side street: the road name ends in Place
#   - The intersection is surrounded by businesses that drive evening and weekend traffic: area zoning might be useful (industrial, commercial, retail, residential). 
#   - It's in the city of Cambridge, not Boston proper: try density or distance from a common point
#   - There's a stoplight with protected left turns: control mechanism (and rare for Boston - hence the [Boston Left](https://www.urbandictionary.com/define.php?term=Boston%20Left).

# #### Example 2 - Philadelphia
# Let's now look at Philadelphia the same way. Here are the intersections contained in the data. 
# EDIT: As before, geographies are generally similar. 

# In[ ]:


tt_phi = tt[tt.City == 'Philadelphia'].drop_duplicates(['level_0','IntersectionId'])
points_phi = gv.Points(tt_phi, kdims=['Longitude', 'Latitude'],
                      vdims=['level_0']).opts(color='level_0', cmap=['dodgerblue', 
                      'darkorange'], width=500, height=450, alpha=0.5)
tiles = gv.tile_sources.CartoLight()
display(tiles*points_phi)


# Let's pick a single intersection like before. Here's the intersection of North 5th Street and W Cambria. This is Rocky Balboa's neighborhood in North Philly and around the way from Max's steaks, where Adonis Creed likes to take his dates.
# 
# <img src="https://i.imgur.com/1IFmj6p.png" width="350" height="350"/>
# <img src="https://i.imgur.com/fjXWk56.png" width="550" height="350"/>
# 
# Movie references aside, here are some things to notice.
#   - Intersection of two one-way streets: a 4-way intersection with only 4 paths. Restrictions or potential vs. actual might help.
#   - Three similar intersections in the area: nearest neighbors as a feature.
#   - Both streets are medium capacity and run across neighborhoods: not sure how to use this one...
#   - Streets are longer streets: likely if name contains N/S/E/W.
#   - Urban location: likely if a street is numbered.
#   - It's daytime and sunny: of course this is a picture, but day/night and weather affect traffic.

# You might come up with ideas for good features after looking at a few intersections. Whether the data is available or not I have no idea - it's more of a wish list at this point. Having met with city traffic managers in my area, I can tell you they have all sorts of helpful data: traffic volumes, lane geometry, signal timings and so on. Wouldn't that be great for us to have?

# ## Wait Times
# 
# ### Distribution of Means and Variation
# OK, let's look at something we have right now - the wait times - and see what inferences we might make.  I changed the format of the "Path" column in the original data. It reflects how I've seen traffic people label paths, which is based on the direction of entry and the type of turn. Plus it's more intuitive and concise (and leads to more features!). Here's how the data is organized.

# In[ ]:


index_cols = ['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading']
value_cols = ['TimeFromFirstStop_p' + str(i) for i in [20, 40, 50, 60 , 80]]

def get_times(city, iid, pathlist):
    intersect = train[(train.City == city) & (train.IntersectionId == iid)]
    targets = target_df.loc[intersect.index, :]
    intersect = intersect.join(targets)
    paths = intersect.groupby(index_cols)[value_cols].agg(['mean', 'std']).fillna(0)
    paths['Path'] = pathlist
    paths.columns = paths.columns.swaplevel()
    return paths.sort_index(axis=1)

pathlist_bos = ['E_left', 'E_right', 'NE_left', 'SW_right', 'NE_thru', 'SW_u', 'SW_thru']
bos = get_times('Boston', 2, pathlist_bos)

pathlist_phi = ['N_thru', 'N_right', 'E_left', 'E_thru']
phi = get_times('Philadelphia', 1824, pathlist_phi)

display(bos, phi)


# Let's look at some plots and see if we can spot anything.

# In[ ]:


import hvplot.pandas
opts = {'invert_yaxis': False,
        'yticks': list(range(0,100,20)),
        'padding': 0.1,
        'width':450,
        'height': 300,
           }

# df = bos
# aggfunc='mean'
def make_plot(df, aggfunc):
    assert (aggfunc == 'mean') | (aggfunc == 'std')
    paths = df.set_index(('', 'Path')).loc[:, aggfunc].reset_index()
    paths.columns = [paths.columns[0][1]] + [c[-4:] for c in paths.columns[1:]]
    plot = hvplot.parallel_coordinates(paths, 'Path', **opts)
    if aggfunc == 'mean':
        return plot.options(ylabel='Mean Wait Time')
    else:
        return plot.options(ylabel='STD of Wait Times', show_legend=False)

land_cambridge = make_plot(bos, 'mean').options(title="Land & Cambridgeside") +    make_plot(bos, 'std')
fifth_cambria = make_plot(phi, 'mean').options(title="5th & Cambria") +    make_plot(phi, 'std')

display(land_cambridge, fifth_cambria)


# There appear to be some things that make sense and other things that might depend on interactions with features we don't yet have. Looking at the Land intersection we can see that the left turns take the most time on average. The right turn off Cambridgeside is the quickest and has the lowest variation. We have the direction of turn data which should help a model. Going straight through on Land or u-turning has the highest variation. We'll look at that in a minute. 
# 
# For the 5th and Cambria intersection, It looks like going east on Cambria has the highest average times, whether one is going straight or turning left. Let's look at that also to see what's up.

# ### P-80 Deep Dive
# Here's a quick look at the p80 data to get more insight into high mean wait times and variations for the through traffic headed NE on Land.

# In[ ]:


opts = {'cmap': 'Paired',
        'yticks': list(range(0,300,50)),
        'colorbar': False,
       'grid': True,
         }

land_ne = tt[(tt.IntersectionId == 2) &
             (tt.EntryStreetName == 'Land Boulevard') &
             (tt.ExitHeading == 'NE')
             ].join(target_df)
landplot = land_ne.hvplot.scatter('Hour', 'TimeFromFirstStop_p80', 
                                    c='Weekend', **opts)

cambria_e = tt[(tt.IntersectionId == 1824) &
             (tt.EntryStreetName == 'West Cambria Street') &
             (tt.ExitHeading == 'E')
             ].join(target_df)
cambplot = cambria_e.hvplot.scatter('Hour', 'TimeFromFirstStop_p80', 
                                    c='Weekend', **opts)

display(landplot.options(title='Land_NE_thru'), cambplot.options(title='Cambria_E_thru'))


# In[ ]:


# alternate plot with seaborn to trigger viz output on the Notebooks page
import matplotlib.pyplot as plt
import seaborn as sns
cambria_e.plot(kind='scatter', x='Hour', y='TimeFromFirstStop_p20')


# Traffic on Land looks typical for a metro area, peaking at afternoon rush hour and dying down. Cambria on the other hand, has a relatively steady flow at p80 with a peak at around 2pm. We don't have an obvious cause, but time diffs could be useful to include.
# 
# I hope the notebook gave some ideas for features to add to your model. Using these features with your own and those in other puclic kernels should get your RMSE down toward 68 if not below. Good luck!
