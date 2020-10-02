#!/usr/bin/env python
# coding: utf-8

# ## In this tutorial, we'll try to do data visualization using Plotly. Plotly, just like Seaborn and Matplotlib is another Python library to make graphs or charts, it can make interactive, publication-quality graphs online. We are using the [Pokemon dataset](https://www.kaggle.com/gurarako/pokemon-all-cleaned) that have been cleaned beforehand.
# 
# ## Table of Content
# * [Plotly polar chart](#polar)
# * [Using polar chart to compare Pokemon](#compare)
# 
# As usual, we'll start by importing all the packages and take a peek at the first 5 rows of the dataframe

# In[ ]:


import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 

plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 100


# In[ ]:


pokedata = pd.read_csv("../input/pokemon-all-cleaned/pokemon_cleaned.csv")


# In[ ]:


pokedata.head()


# <a id='polar'></a>
# ## Plotly polar chart 
# 
# 
# Polar charts or radar charts are a way of comparing multiple quantitative variables. This makes them useful for seeing which variables have similar values or if there are any outliers amongst each variable. Polar charts are also useful for seeing which variables are scoring high or low within a dataset, making them ideal for displaying performance.
# 
# Now let's use plotly to draw a polar chart showing Charizard stats 
# 

# In[ ]:


x = pokedata[pokedata["Name"] == "Charizard"]
data = [go.Scatterpolar(
  r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
  fill = 'toself',
     line =  dict(
            color = 'orange'
        )
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200]
    )
  ),
  showlegend = False,
  title = "{} stats distribution".format(x.Name.values[0])
)
fig = go.Figure(data=data, layout=layout)
fig.layout.images = [dict(
        source="https://rawgit.com/guramera/images/master/Charizard.png",
        xref="paper", yref="paper",
        x=0.95, y=0.3,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      )]

iplot(fig, filename = "Single Pokemon stats")


# According to the above chart, there doesn't seem to be any particular stats that Charizard truly excels or lacks at. Charizard seems to have mostly above average stats. 
# 
# Maybe now we'll try some Pokemon that has extreme difference in their stats, let's find which Pokemon has the highest speed stat first and then use  that Pokemon data in the polar chart.

# In[ ]:


pokedata.loc[[pokedata['Speed'].idxmax()]]


# In[ ]:


x = pokedata.loc[[pokedata['Speed'].idxmax()]]
data = [go.Scatterpolar(
  r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
  fill = 'toself',
     line =  dict(
            color = 'darkkhaki'
        )
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200]
    )
  ),
  showlegend = False,
  title = "{} stats distribution".format(x.Name.values[0])
)
fig = go.Figure(data=data, layout=layout)
fig.layout.images = [dict(
        source="https://rawgit.com/guramera/images/master/Ninjask.png",
        xref="paper", yref="paper",
        x=0.95, y=0.3,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      )]

iplot(fig, filename = "Single Pokemon stats")


# Now it's clearer how polar chart are especially good for visualizing comparisons of quality data. We can clearly see how **Ninjask** has extremely high **speed**, but is very lacking in other stats, especially **defense, sp. attack, and sp. def**

# <a id='compare'></a>
# ##  Using polar chart to compare Pokemon
# 
# Polar charts maybe are the most effective to compare two or more products to see the quality difference between them. It will be interesting if we're able to compare more than one Pokemon in the polar chart to see how they compare. Let's see if we can do that by adding another Pokemon data to the code

# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) 

a = pokedata[pokedata["Name"] == "Blastoise"]
b = pokedata[pokedata["Name"] == "Charizard"]

data = [
    go.Scatterpolar(
        name = a.Name.values[0],
        r = [a['HP'].values[0],a['Attack'].values[0],a['Defense'].values[0],a['Sp. Atk'].values[0],a['Sp. Def'].values[0],a['Speed'].values[0],a["HP"].values[0]],
        theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
        fill = 'toself',
        line =  dict(
                color = 'cyan'
            )
        ),
    go.Scatterpolar(
            name = b.Name.values[0],
            r = [b['HP'].values[0],b['Attack'].values[0],b['Defense'].values[0],b['Sp. Atk'].values[0],b['Sp. Def'].values[0],b['Speed'].values[0],b["HP"].values[0]],
            theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
            fill = 'toself',
            line =  dict(
                color = 'orange'
            )
        )]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200]
    )
  ),
  showlegend = True,
  title = "{} vs {} Stats Comparison".format(a.Name.values[0], b.Name.values[0])
)

fig = go.Figure(data=data, layout=layout)

fig.layout.images = [dict(
        source="https://rawgit.com/guramera/images/master/blastoise.jpg",
        xref="paper", yref="paper",
        x=0.05, y=-0.15,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      ),
        dict(
        source="https://rawgit.com/guramera/images/master/Charizard.png",
        xref="paper", yref="paper",
        x=1, y=-0.15,
        sizex=0.6, sizey=0.6,
        xanchor="center", yanchor="bottom"
      ) ]


iplot(fig, filename = "Pokemon stats comparison")


# By looking at the above chart, we can understand whether certain Pokemon have any particular stats that they excel or lack at using polar chart, . We can also add more Pokemon to the chart, but we must be mindful of the chart readability if we put too many variables.

# **That's it for this post, thank you so much for reading it, special thanks to Chinmay Upadhye for providing the original Kernel and also Alberto Barradas and Rounak Banik for providing excellent datasets.**
# 
