#!/usr/bin/env python
# coding: utf-8

# # When did mixtapes become so popular?
# 
# First, load up the data:

# In[ ]:


# core libraries
import sqlite3
import pandas as pd
import numpy as np

# plotting libraries
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import Scatter, Figure, Layout
init_notebook_mode()

# get the data...
con = sqlite3.connect('../input/database.sqlite')
torrents = pd.read_sql_query('SELECT * from torrents;', con)
con.close()

# define mixtape and album subset
mixtapes = torrents.loc[torrents.releaseType == 'mixtape']
albums = torrents.loc[torrents.releaseType == 'album']


# The 'popularity' of a release type is not  an objective measure. The best metric we have is the number of *snatches* from each release: the number of times each release has been downloaded by users of What.CD. The popularity of a release *type* (mixtape vs. album) can be measured as average number of snatches per release. 
# 
# Problematically, the distribution of the number of snatches per release is *very* skewed: most releases have fewer than 5 snatches, but there are several releases with over 20,000. To make the data a little more normally distributed, I use a log transform.

# In[ ]:


# define year range
years = np.arange(1991, 2017)

# get average and standard error of log snatches per release 
snatches = pd.DataFrame(0, index = years, columns = ['Mixtapes','Albums'])
stderror = pd.DataFrame(0, index = years, columns = ['Mixtapes','Albums'])

# compute data for each year
for i in years:
    
    # index releases from current year
    year_mixtapes = mixtapes.loc[mixtapes.groupYear == i]
    year_albums = albums.loc[albums.groupYear == i]
    
    # take log transform -- add one to prevent log(0) error
    year_mixtapes = np.log(year_mixtapes.totalSnatched + 1)
    year_albums = np.log(year_albums.totalSnatched + 1)
    
    # get average snatches per release
    snatches.loc[i,'Mixtapes'] = year_mixtapes.mean() 
    snatches.loc[i,'Albums'] = year_albums.mean() 
    
    # get standard error
    stderror.loc[i,'Mixtapes'] = year_mixtapes.std() / np.sqrt(len(year_mixtapes))
    stderror.loc[i,'Albums'] = year_albums.std() / np.sqrt(len(year_albums))    


# ## Plotting

# In[ ]:


linespecs = {
    'Mixtapes':  dict(color = 'blue', width = 2),
    'Albums':  dict(color = 'red', width = 2),
    }


handles = []
for k in linespecs.keys():
    handles.append( Scatter(
            x = years, 
            y = snatches[k],
            name = k, 
            hoverinfo = 'x+name',
            line = linespecs[k],
            error_y = dict(type='data', 
                           array = stderror[k], 
                           color = linespecs[k]['color'])
               )
        )

    
layout = Layout(
    xaxis = dict(
                tickmode = 'auto',
                nticks = 20, 
                tickangle = -60, 
                showgrid = False
            ),
    yaxis = dict(title = 'Log Snatches Per Release'),
    hovermode = 'closest',
    legend = dict(x = 0.55, y = 0.15),
)


fh = Figure(data=handles, layout=layout)
iplot(fh)


# ## Results
# 
# Whereas albums are the dominant format throughout the '90s, mixtapes rise in popularity from 1997 to 2001 and are competitive with albums thereafter.
# 
# Another unique pattern here is the sharp decline in snatches per release starting in 2009. Since the popularity score (log snatches) is an *average*, the decline could be due to:
# 
# 1.  An increase in the number of releases without an increase in the number of snatches.
# 2. A decline in the number of snatches without a decline in the overall number of releases.
# 
# These possibilities can be straightforwardly evaluated:

# In[ ]:


# aggregate over mixtapes and albums
releases = torrents.loc[torrents.releaseType.isin(['mixtape', 'album'])]
data = pd.DataFrame(index = years, columns = ['Snatches', 'Releases'])

# compute data for each year
for i in years:
    year_releases = releases.loc[releases.groupYear == i]
    data.loc[i,'Snatches'] = np.sum(np.log(year_releases.totalSnatched + 1))
    data.loc[i,'Releases'] = year_releases.shape[0] 

    
# plot as scatter
labels = ["'" + str(i)[2:] for i in years]
sh = Scatter(
    x = data.Releases, y = data.Snatches,
    mode = 'text', text = labels,
    textposition='center',
    hoverinfo = 'none',
    textfont = dict( family='monospace', size=14, color='red'),
    name = None
)

# a quick reference line
slope = 4.1
lh = Scatter(
    x = [min(data.Releases), max(data.Releases)], 
    y = [slope*min(data.Releases), slope*max(data.Releases)],
    mode = 'lines', line = dict(color = 'gray', width = 1),
    name = '2009 Extrapolation'
)

    
layout = Layout(
    yaxis = dict(title = 'Total Log Snatches'),
    xaxis = dict(title = 'Number of Releases'),
    hovermode = 'closest',
    showlegend=False,
    annotations= [dict(
        x = 4000, y = 4000 * slope,
        text = 'log(Snatches) = 4.1 * Releases',
        font = dict(family = 'serif', size = 14),
        showarrow = False, bgcolor = 'white'
        )]
)

fh = Figure(data=[lh, sh], layout=layout)
iplot(fh)


# Clearly, there is an increase in the number of releases starting at about 2008. The increases look nonlinear: whereas there are only incremental gains from 1991 to 2002, we get huge increases between 2008 and 2015. 
# 
# On the figure I have plotted as a reference a linear model: *log(Snatches) = 4.1 x Releases*, where the slope (4.1) reflects the ratio between log snatches and the number of releases in 2009. This identifies the source of the decline: the snatches have not kept up with the releases. 
# 
# I think there's a couple things going on here. The boring explanation is that snatches are *cumulative*: people have had more time to download and listen to older records, which inflates their snatch rate.
# 
# A more interesting dynamic (interesting to me, anyway) is that it takes time for any record's impact to be felt on the genre. Not only were there fewer hip hop records being released in the '90s, but these records were also foundational in the genre, and they're likely to be very popular torrents. So, if we take these data at face value, it would then appear that it takes about 7 years (2009-2016) for hip hop records to enjoy this degree of seniority.
