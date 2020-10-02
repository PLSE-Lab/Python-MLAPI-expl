#!/usr/bin/env python
# coding: utf-8

# # Visualization of the Iris Species via plotly
# 
# Iris Species is a simple and beautiful dataset which can be easily visualized with a strong visualization library, **plotly**.  
# 
# I will be using plotly's appropriate functions to visualize the iris species data, and then make some conclusions.

# In[ ]:


import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/Iris.csv")


# In[ ]:


data.info()


# Dataframe contains 150 rows and 6 columns. Seems like each row corresponds to an individual flower. The columns in this dataset are:  
# 
# * `Id`
# * `SepalLengthCm`
# * `SepalWidthCm`
# * `PetalLengthCm`
# * `PetalWidthCm`
# * `Species`

# Let's take a glimpse at the samples in this dataset:

# In[ ]:


data.head()


#  ## Scatter Plot
# Scatter plot is a good way to visualize the correlations among features. I will be examining the correlation of `SepalLengthCm` with other features. So Sepal Length will be our y-axis, others will be laying on the x-axis. And I sorted and kept them in distinct dataframes to see correlations clearly.

# In[ ]:


data_sorted_bySW = data.sort_values('SepalWidthCm')
data_sorted_byPL = data.sort_values('PetalLengthCm')
data_sorted_byPW = data.sort_values('PetalWidthCm')


# In[ ]:


df = data.iloc[:100, :]

bySW = go.Scatter(
                    x = data_sorted_bySW.SepalWidthCm,
                    y = data_sorted_bySW.SepalLengthCm,
                    mode = "markers",
                    name = "Sepal Width (cm)",
                    marker = dict(color = 'rgba(255, 0, 0, 0.9)'),
                    text = data_sorted_bySW.Species
)

byPL = go.Scatter(
                    x = data_sorted_byPL.PetalLengthCm,
                    y = data_sorted_byPL.SepalLengthCm,
                    mode = "markers",
                    name = "Petal Length (cm)",
                    marker = dict(color = 'rgba(0, 255, 0, 0.9)'),
                    text = data_sorted_byPL.Species
)

byPW = go.Scatter(
                    x = data_sorted_byPW.PetalWidthCm,
                    y = data_sorted_byPW.PetalWidthCm,
                    mode = "markers",
                    name = "Petal Width (cm)",
                    marker = dict(color = 'rgba(0, 0, 255, 0.9)'),
                    text = data_sorted_byPW.Species
)

layout = dict(title = 'Change of Sepal Length by Other Properties',
              xaxis= dict(title= 'centimeters',ticklen= 5,zeroline= False)
             )
u = [bySW, byPL, byPW]
fig = dict(data = u)
iplot(fig)


# * Seems like Petal Width and Sepal Length has a very strong correlation.  
# * We can say there is a correlation between Petal Length and Sepal Length, but not like the one above.  
# * There is no correlation between the Sepal Length and Sepal Width.

# ## Bar Plot
# Let's visualize each species' average lengths, so we will be able to see how features change as genre of the flower changes.

# In[ ]:


data1 = data.groupby(data.Species).mean()
data1['Species'] = data1.index

t1 = go.Bar(
            x = data1.Species,
            y = data1.SepalLengthCm,
            name = "Sepal Length (cm)",
            marker = dict(color = 'rgba(160, 55, 0, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

t2 = go.Bar(
            x = data1.Species,
            y = data1.SepalWidthCm,
            name = "Sepal Width (cm)",
            marker = dict(color = 'rgba(0, 55, 160, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

t3 = go.Bar(
            x = data1.Species,
            y = data1.PetalLengthCm,
            name = "Petal Length (cm)",
            marker = dict(color = 'rgba(20, 55, 30, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

t4 = go.Bar(
            x = data1.Species,
            y = data1.PetalWidthCm,
            name = "Petal Width (cm)",
            marker = dict(color = 'rgba(70, 55, 160, 0.8)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text = data1.Species
)

b = [t1,t2,t3,t4]
layout_bar = go.Layout(barmode = "group")
fig_bar = go.Figure(data = b, layout = layout_bar)
iplot(fig_bar)


# As we can clearly observe from the barplot above; Sepal Length, Petal Length and Petal Width features grow as we walk in species Iris-setosa, Iris-versicolor and Iris-virginica, respectively.

# ## Bubble Chart
# Now I want to visualize all four attributes in one chart. Bubble chart is an appropriate way of visualization for this kind of purposes.  
# Let's decide what each parameter corresponds to:  
# `x` : `PetalLengthCm`  
# `y` : `PetalWidthCm`  
# `color` : `SepalWidthCm`  
# `size` : `SepalLengthCm`

# In[ ]:


fig_bubble = [
    {
        'x' : data.PetalLengthCm,
        'y' : data.PetalWidthCm,
        'mode' : 'markers',
        'marker' : {
            'color' : data.SepalWidthCm,
            'size' : data.SepalLengthCm,
            'showscale' : True
        },
        'text' : data.Species
    }
]
iplot(fig_bubble)


# ## Boxplot
# Boxplot is always the best choice, if we want to get some statistical information from the data.

# In[ ]:


t1_box = go.Box(
                name = 'Sepal Length (cm)',
                y = data.SepalLengthCm,
                marker = dict(color = 'rgba(160,160,50,0.7)')
)

t2_box = go.Box(
                name = 'Sepal Width (cm)',
                y = data.PetalWidthCm,
                marker = dict(color = 'rgba(50,160,150,0.7)')
)

t3_box = go.Box(
                name = 'Petal Length (cm)',
                y = data.PetalLengthCm,
                marker = dict(color = 'rgba(160,60,150,0.7)')
)

t4_box = go.Box(
                name = 'Petal Width (cm)',
                y = data.SepalWidthCm,
                marker = dict(color = 'rgba(150,160,150,0.7)')
)

fig_box = [t1_box, t2_box, t3_box, t4_box]

iplot(fig_box)


# * Petal Width has a short range from 2 to 4.4, with some outliers.
# * Petal Length has a wide range between 1 cm and 6.9 cm, but without outliers.

# ## Scatterplot Matrix
# Now let's examine the relations by crosschecking each feature by scatterplot matrix.

# In[ ]:


import plotly.figure_factory as ff
data_ff = data.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
data_ff['index'] = np.arange(1, len(data_ff)+1)

fig_ff = ff.create_scatterplotmatrix(data_ff, diag = 'box', index = 'index', colormap = 'Blues', colormap_type = 'cat', height = 800, width = 800)
iplot(fig_ff)


# ## 3D Scatter
# Let's jump into third dimension, and decide what each variable correspond to:  
# `x` : `SepalLengthCm`  
# `y` : `SepalWidthCm`  
# `z` : `PetalLengthCm`  
# `color` : `PetalWidthCm`

# In[ ]:


trace_3d = go.Scatter3d(
                        x = data.SepalLengthCm,
                        y = data.SepalWidthCm,
                        z = data.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        #name = data.Species,
                        marker = dict(
                                    size = 5,
                                    color = data.PetalWidthCm
                        )
)

list_3d = [trace_3d]

fig_3d = go.Figure(data = list_3d)
iplot(fig_3d)


# Since trying to use `name` parameter with `data.Species` giving `ValueError`, I decided to eliminate the color part of 3D graph, and plot them as different traces.  
# 
# `Iris-setosa` : pink  
# `Iris-versicolor` : lime  
# `Iris-virginica` : blue

# In[ ]:


i_setosa = data[data['Species']  == 'Iris-setosa']
i_versicolor = data[data['Species']  == 'Iris-versicolor']
i_virginica = data[data['Species']  == 'Iris-virginica']


# In[ ]:


# Iris-setosa
trace_setosa = go.Scatter3d(
                        x = i_setosa.SepalLengthCm,
                        y = i_setosa.SepalWidthCm,
                        z = i_setosa.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        name = "Iris-setosa",
                        marker = dict(
                                    size = 5,
                                    color = 'rgba(255,102, 255,0.8)'
                        )
)

# Iris-versicolor
trace_versicolor = go.Scatter3d(
                        x = i_versicolor.SepalLengthCm,
                        y = i_versicolor.SepalWidthCm,
                        z = i_versicolor.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        name = "Iris-versicolor",
                        marker = dict(
                                    size = 5,
                                    color = 'rgba(102, 255, 51, 0.8)'
                        )
)

# Iris-virginica
trace_virginica = go.Scatter3d(
                        x = i_virginica.SepalLengthCm,
                        y = i_virginica.SepalWidthCm,
                        z = i_virginica.PetalLengthCm,
                        mode = 'markers',
                        opacity = 0.7,
                        name = "Iris-virginica",
                        marker = dict(
                                    size = 5,
                                    color = 'rgba(51, 102, 255, 0.8)'
                        )
)

list_3d = [trace_setosa, trace_versicolor, trace_virginica]

fig_3d = go.Figure(data = list_3d)
iplot(fig_3d)


# As we can conclude from the graph above, clustering flowers due to their features would make sense at its finest!

# In[ ]:


data.head()


# In[ ]:




