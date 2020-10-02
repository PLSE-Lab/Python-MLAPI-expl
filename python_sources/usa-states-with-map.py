#!/usr/bin/env python
# coding: utf-8

# # USA States
# 
# Dataset Description: This data set provides all USA Sates name, Longitude and Latitude.
# 
# <table>
#   <tr>
#     <th>File Name : </th>
#     <th>USA_States.csv</th> 
#   </tr>
#   <tr>
#     <th>File Size :</th>
#     <th>Approx. 1kb</th> 
#   </tr>
#   <tr>
#     <th>Total Records :</th>
#     <th>51</th> 
#   </tr>
#     <tr>
#     <th>File Updated :</th>
#     <th>October 18, 2018</th> 
#   </tr>
# </table>
# 
# #### Note: We need import plotly offline pakage

# # Import Packages

# In[ ]:


import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# # Import File and Print File Data

# In[ ]:


df = pd.read_csv('../input/USA_States.csv')
df


# In[ ]:


scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df['Longitude'],
        lat = df['Latitude'],
        text = df['State'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'round',
            line = dict(
                width=2,
                color='rgba(102, 102, 102)'
            ), 
        ))]
layout = dict(
        title = 'USA States Capital',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
fig = dict( data=data, layout=layout )
py.offline.iplot( fig, validate=False, filename='d3-airports' )

