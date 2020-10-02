# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Plot a thematic map (choropleph map)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


init_notebook_mode(connected=True)

df = pd.read_csv("../input/top_terror_countries.csv")
locations = df["country_name_iso2"]


data = [ dict(
        type = 'choropleth',
        locationmode ="country names", 
        locations = df["country_name_iso2"],
        z = df['fatalities'],
        text = df["country_name_iso2"],

        colorscale = 'Electric'  , 
        
        autocolorscale = False,
        showscale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'fatalities<br>per 1.000'),
      ) ]

layout = dict(
    title = '1970-2015 Terrorism Attack<br>Source:\
            <a href="http://start.umd.edu/gtd/">\
            GTD Global Terrorism Database</a>',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showcountries = True,
        projection = dict(
            type = 'Mercator'
    
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )


print ('finished...')
print(df)