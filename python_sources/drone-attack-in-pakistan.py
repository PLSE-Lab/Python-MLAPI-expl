#!/usr/bin/env python
# coding: utf-8

# Data Import
# ------------------

# In[ ]:


import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
pakistan = pd.read_csv('../input/PakistanDroneAttacks.csv', encoding='ISO-8859-1')

pakistan.columns=pakistan.columns.str.lower()
pakistan=pakistan.dropna(subset=['date'])
separate=pakistan['date'].str.split(',')
day,month,years=zip(*separate)
pakistan['years']=years


# Died and Injured by Drone Acttack (2004-2017)
# -----------------------------------------------

# In[ ]:


pakistan_years=np.asarray(pakistan['years'].unique())
pakistan_died=pakistan.groupby('years')['total died mix'].count()
pakistan_injured=pakistan.groupby('years')['injured max'].count()

labels = ['DIED', 'INJURED']
colors = ['rgb(255, 51, 0)', 'rgb(0, 51, 204)']
x_data = pakistan_years
y_data = [pakistan_died, pakistan_injured]

traces = []
for i in range(0, 2):
    traces.append(go.Scatter(
        x = x_data,
        y = y_data[i],
        mode = 'splines',
        name = labels[i],
        line = dict(color = colors[i], width = 1.5)
    ))

layout = { 
  "title": "Died and Injured by Drone Acttack (2004-2017)", 
  "xaxis": {"title": "Years"}, 
  "yaxis": {"title": "People"}
}

figure = dict(data = traces, layout = layout)
iplot(figure)



# Drone Attacks by Latitude/Longitude in Pakistan (2004-2017)
# ------------------------------------------------------------

# In[ ]:


pakistan['injured max']=pakistan['injured max'].fillna(0)
pakistan['total died mix']=pakistan['total died mix'].fillna(0)

pakistan['text'] = pakistan['date'] + '<br>' +                     pakistan['total died mix'].astype(str) + ' Killed, ' +                     pakistan['injured max'].astype(str) + ' Injured' + '<br>'+ 'City: ' + pakistan['city'].astype(str) + '<br>' + 'Location: ' + pakistan['location'].astype(str)
                    

died = dict(
           type = 'scattergeo',
           locationmode = 'Pakistan',
           lon = pakistan[pakistan['total died mix'] > 0]['longitude'],
           lat = pakistan[pakistan['total died mix'] > 0]['latitude'],
           text = pakistan[pakistan['total died mix']  > 0]['text'],
           mode = 'markers',
           name = 'DIED',
           hoverinfo = 'text+name',
           marker = dict(
               size = pakistan[pakistan['total died mix']  > 0]['total died mix'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
        
injuries = dict(
         type = 'scattergeo',
         locationmode = 'Pakistan',
         lon = pakistan[pakistan['total died mix']  == 0]['longitude'],
         lat = pakistan[pakistan['total died mix'] == 0]['latitude'],
         text = pakistan[pakistan['total died mix']  == 0]['text'],
         mode = 'markers',
         name = 'INJURIES',
         hoverinfo = 'text+name',
         marker = dict(
             size = (pakistan[pakistan['total died mix']  == 0]['injured max'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )

layout = go.Layout(
    title = 'Drone Attacks by Latitude/Longitude in Pakistan (2004-2017)',
     showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
    geo = dict(
        resolution = 50,
        scope = 'Pakistan',
        showframe = False,
        showcoastlines = True,
        showland = True,
        showcountries = True,
        landcolor = "rgb(200,200,200)",
        countrycolor = "rgb(1, 1, 1)" ,
        coastlinecolor = "rgb(1, 1, 1)",
        projection = dict(
            type = 'Mercator'
        ),
        lonaxis = dict( range= [ 62.0,78.0 ] ),
        lataxis = dict( range= [ 24.0,35 ] ),
        domain = dict(
            x = [ 0, 1 ],
            y = [ 0, 1 ]
        )
    )
    )

data = [died, injuries]
figure = dict(data = data,layout = layout)

iplot(figure)

