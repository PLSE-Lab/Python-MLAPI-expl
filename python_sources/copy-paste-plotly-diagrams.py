#!/usr/bin/env python
# coding: utf-8

# # Copy/Paste Plotly Diagrams
# 
# In this notebook you can find nearly all interactive plots of the **plotly library to copy/paste** them into your notebook.<br>
# Have fun!
# 
# **Basics:**<br>
# Every graph consists of a **data-dictionary** and a **layout-dictionary.**
# 
# plotly.graph_obs.Figure( data = [ data_dict ], layout = layout_dict )
# 
# Most of the keys in the dictionaries can accept a single value (one value for all instances) or a list of values (one value for each instance). Removing a key results in the default parameters being loaded (A good choice for most of the time).<br>
# A Figure can have a list of data-dictionaries. Each dictionary will create a single graph on your canvas.
# 
# [1. Import The Libraries](#1)    
# [2. Load The Data](#2)    
# [3. Single Barplot](#3)    
# [4. Multiple Barplots](#4)    
# [5. Scatter Plots](#5)    
# [6. Boxplot](#6)    
# [7. Histogram](#7)    
# [8. Distplots](#8)    
# [9. Split Violin](#9)    
# [10. Donut](#10)    
# [11. Tree Map](#11)    
# [12. Sankey Diagram](#12)    
# [13. 2D Histogram](#13)    
# [14. 2D Contour](#14)    
# [15. Ternary](#15)    
# [16. Radar Chart](#16)    
# [17. Parallel Coordinates](#17)    
# [18. Subplots](#18)    
# 
# # <a id=1>Import The Libraries</a>

# In[ ]:


# To store the data
import pandas as pd

# To create interactive plots
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)

# To support tree-maps
import squarify


# # <a id=2>Load The Data</a>

# In[ ]:


df = pd.read_csv('../input/Pokemon.csv')
df.head()


# # <a id=3>Single Barplot</a>

# In[ ]:


# DataFrame
plot_df = df[['Type 1', 'Type 2', 'HP']].set_index('HP').stack().rename('Type').reset_index().groupby('Type').HP.median().sort_values(ascending=False)

title = 'Single Barplot: Pokemon median HP grouped by Type'

data = go.Bar(x = plot_df.index,
              y = plot_df,
              base = 0,
              text = plot_df,
              textposition = 'auto',
              width = 0.8,
              name = 'Pokemon',
              marker = dict(color = '#abcdef',
                            line=dict(color = '#000000',
                                      width = 2.0)),
              opacity = 1.0)

layout = dict(title = title,
              xaxis = dict(title = 'Type',
                           tickangle = -60,
                           titlefont = dict(size = 16,
                                            color = '#000000'),
                           tickfont = dict(size = 14,
                                           color = '#000000')),
              yaxis = dict(title = 'HP',
                           tickangle = 0,
                           titlefont = dict(size = 16,
                                            color = '#000000'),
                           tickfont = dict(size = 14,
                                           color = '#000000')),
              legend = dict(x = 0,
                          y = 1.0,
                          bgcolor = '#ffffff',
                          bordercolor = '#ffffff'),
              barmode = 'group',
              bargap = 0.15,
              bargroupgap = 0.1)

# Create the plot
fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=4>Multiple Barplots</a>

# In[ ]:


# DataFrame
plot_df_1 = df[['Type 1', 'Type 2', 'Attack']].set_index('Attack').stack().rename('Type').reset_index().groupby('Type').Attack.median().sort_values(ascending=False)
plot_df_2 = df[['Type 1', 'Type 2', 'Defense']].set_index('Defense').stack().rename('Type').reset_index().groupby('Type').Defense.median().sort_values(ascending=False)

title = 'Multiple Barplots: Pokemon median Attack and Defense grouped by Type'

data_1 = go.Bar(x = plot_df_1.index,
                y = plot_df_1,
                base = 0,
                text = plot_df_1,
                textposition = 'auto',
                width = 0.35,
                name = 'Attack',
                marker = dict(color = '#dd1c77',
                              line=dict(color = '#000000',
                                        width = 2.0)),
                opacity = 1.0)

data_2 = go.Bar(x = plot_df_2.index,
                y = plot_df_2,
                base = 0,
                text = plot_df_2,
                textposition = 'auto',
                width = 0.35,
                name = 'Defense',
                marker = dict(color = '#2ca25f',
                              line=dict(color = '#000000',
                                        width = 2.0)),
                opacity = 1.0)

layout = dict(title = title,
              xaxis = dict(title = 'Type',
                           tickangle = -60,
                           titlefont = dict(size = 16,
                                            color = '#000000'),
                           tickfont = dict(size = 14,
                                           color = '#000000')),
              yaxis = dict(title = 'Attack / Defense',
                           tickangle = 0,
                           titlefont = dict(size = 16,
                                            color = '#000000'),
                           tickfont = dict(size = 14,
                                           color = '#000000')),
              legend = dict(x = 0.92,
                          y = 1.0,
                          bgcolor = '#ffffff',
                          bordercolor = '#ffffff'),
              barmode = 'group',
              bargap = 0.2,
              bargroupgap = 0.0)

# Create the plot
fig = go.Figure(data=[data_1, data_2], layout=layout)
iplot(fig)


# # <a id=5>Scatter Plots</a>

# In[ ]:


title = 'Scatter Plot: Pokemon Attack And Defense'

data = go.Scatter(x = df.Defense,
                  y = df.Attack,
                  text = df.Name,
                  mode = 'markers', #'lines+markers', 'lines'
                  textposition = 'auto',
                  name = 'Pokemon',
                  marker = dict(size = 5,
                                color = '#ff0000',
                                colorscale = 'Viridis',
                                showscale = False,
                                opacity = 1.0,
                                line = dict(width = 0,
                                            color = '#000000')),
                  opacity = 1.0)

layout = dict(title = title,
              xaxis = dict(title = 'Defense',
                           tickangle = 0,
                           zeroline = True,
                           gridwidth = 2,
                           ticklen = 5,
                           titlefont = dict(size = 16,
                                            color = '#000000'),
                           tickfont = dict(size = 14,
                                           color = '#000000')),
              yaxis = dict(title = 'Attack',
                           tickangle = 0,
                           zeroline = True,
                           gridwidth = 2,
                           ticklen = 5,
                           titlefont = dict(size = 16,
                                            color = '#000000'),
                           tickfont = dict(size = 14,
                                           color = '#000000')),
              legend = dict(x = 0.92,
                            y = 1.0,
                            bgcolor = '#ffffff',
                            bordercolor = '#ffffff'),
              hovermode = 'closest')

# Create the plot
fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=6>Boxplot</a>

# In[ ]:


title = 'Boxplot: Pokemon Total Values'

data = go.Box(x = df.Total, # x for rotated graph
              name = 'Pokemon',
              marker = dict(color = '#00cccc',
                            size = 5),
              text = df.Name,
              boxmean = 'sd',
              boxpoints = 'all', # 'suspectedoutliers', 'outliers'
              whiskerwidth=0.2,
              fillcolor='#00ffff',
              line = dict(color = '#00cccc'),
              jitter = 0.3,
              pointpos = -1.8)

layout = go.Layout(title = title,
                   width = None,
                   height = 600,
                   xaxis = dict(title = 'Total',
                                autorange = True,
                                showgrid = True,
                                zeroline = True,
                                dtick = 20,
                                gridcolor = '#ffffff',
                                gridwidth = 1,
                                zerolinecolor = '#ffffff',
                                zerolinewidth = 2),
                   yaxis = dict(tickangle = -90),
                   margin=dict(l = 40,
                               r = 30,
                               b = 80,
                               t = 100),
                   paper_bgcolor = '#ffffff',
                   plot_bgcolor = '#ffffff',
                   showlegend = False)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=7>Histogram</a>

# In[ ]:


title = 'Histogram: Pokemon Special Attack'

data = go.Histogram(x = df['Sp. Atk'], # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Sp. Atk',
                    xbins = dict(start = 0.0,
                                 end = 200.0,
                                 size = 5.0),
                    marker = dict(color = '#FFAA22'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))

layout = go.Layout(title = title,
                   xaxis = dict(title = 'Special Attack'),
                   yaxis = dict(title = 'Count'),
                   #barmode = 'stack',
                   bargap = 0.2,
                   bargroupgap = 0.1)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=8>Distplots</a>

# In[ ]:


title = 'Distplot: Pokemon HP And Speed'

hist_data = [df['HP'], df['Speed']]
group_labels = ['HP', 'Speed']
colors = ['#FF00FF', '#FFFF00']
rug_text = [df.Name, df.Name]

fig = ff.create_distplot(hist_data, 
                         group_labels,
                         histnorm = 'count', # 'probability'
                         bin_size = [5, 5], 
                         colors = colors, 
                         rug_text = rug_text,
                         curve_type = 'kde', # 'normal'
                         show_hist = True,
                         show_curve = True,
                         show_rug = True)

fig['layout'].update(title=title)
iplot(fig)


# # <a id=9>Split Violin</a>

# In[ ]:


title = 'Split Violinplot: Pokemon Attack Grouped By Legendary'

fig = {'data': [{'type' : 'violin',
                 'x' : df['Legendary'].astype(str),
                 'y' : df['Attack'],
                 'legendgroup' : 'Attack',
                 'scalegroup' : 'Attack',
                 'name' : 'Attack',
                 'side' : 'negative',
                 'box' : {'visible' : True},
                 'points' : 'all',
                 'pointpos' : -1.15,
                 'jitter' : 0.1,
                 'scalemode' : 'probability', # 'count'
                 'meanline' : {'visible' : True},
                 'line' : {'color' : 'blue'},
                 'marker' : {'line' : {'width': 0,
                                       'color' : '#000000'}},
                 'span' : [0],
                 'text' : df['Name']},
                
                {'type' : 'violin',
                 'x' : df['Legendary'].astype(str),
                 'y' : df['Defense'],
                 'legendgroup' : 'Defense',
                 'scalegroup' : 'Defense',
                 'name' : 'Defense',
                 'side' : 'positive',
                 'box' : {'visible' : True},
                 'points' : 'all',
                 'pointpos' : 1.15,
                 'jitter' : 0.1,
                 'scalemode' : 'probability', #'count'
                 'meanline' : {'visible': True},
                 'line' : {'color' : 'green'},
                 'marker' : {'line' : {'width' : 0,
                                       'color' : '#000000'}},
                 'span' : [1],
                 'text' : df['Name']}],
       'layout' : {'title' : title,
                   'xaxis' : {'title' : 'Legendary'},
                   'yaxis' : {'zeroline' : False,
                              'title' : 'Attack / Defense'},
                   'violingap' : 0,
                   'violinmode' : 'overlay'}}

iplot(fig, validate=False)


# # <a id=10>Donut</a>

# In[ ]:


title = 'Donut: Pokemon Count By Generation'

plot_df = df.groupby('Generation').Name.count()

fig = {'data' : [{'values' : plot_df,
                  'labels' : plot_df.index,
                  'text' : plot_df,
                  'name' : 'Pokemon',
                  'hoverinfo' :'label+percent+name',
                  'hole' : .4,
                  'type' : 'pie'}],
       'layout': {'title' : title,
                  'annotations' : [{'font' : {'size' : 20},
                                    'showarrow' : False,
                                    'text' : 'Pokemon<br>Generationen',
                                    'x' : 0.5, 
                                    'y' : 0.5}]}}

iplot(fig)


# # <a id=11>Tree Map</a>

# In[ ]:


title = 'Tree Map: Pokemon Type'

x = 0
y = 0
width = 1000
height = 1000

plot_df = df[['Type 1', 'Type 2']].stack().rename('Type').reset_index(drop=True).to_frame().groupby('Type').Type.count()

normed = squarify.normalize_sizes(plot_df, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = ['rgb(166,206,227)','rgb(51,160,44)','rgb(251,154,153)','rgb(227,26,28)']
shapes = []
annotations = []
counter = 0

for i, r in enumerate(rects):
    shapes.append(dict(type = 'rect', 
                       x0 = r['x'], 
                       y0 = r['y'], 
                       x1 = r['x']+r['dx'], 
                       y1 = r['y']+r['dy'],
                       line = dict( width = 2 ),
                       fillcolor = color_brewer[counter]))
    annotations.append(dict(x = r['x']+(r['dx']/2),
                            y = r['y']+(r['dy']/2),
                            text = plot_df.index[i],
                            showarrow = False))
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

data = go.Scatter(x = [ r['x']+(r['dx']/2) for r in rects ], 
                    y = [ r['y']+(r['dy']/2) for r in rects ],
                    text = plot_df,
                    mode = 'text')
        
layout = dict(title = title,
              width = 600,
              height = 600,
              shapes = shapes,
              annotations = annotations,
              xaxis=dict(autorange=True,
                         showgrid=False,
                         zeroline=False,
                         showline=False,
                         autotick=True,
                         ticks='',
                         showticklabels=False),
              yaxis=dict(autorange=True,
                         showgrid=False,
                         zeroline=False,
                         showline=False,
                         autotick=True,
                         ticks='',
                         showticklabels=False),
              hovermode = 'closest')

fig = dict(data=[data], layout=layout)
iplot(fig)


# # <a id=12>Sankey Diagram</a>

# In[ ]:


title = 'Sankey Diagram: Pokemon With Their First And Second Type'

# Create labels
label = ['1. {}'.format(typ) for typ in df['Type 1'].astype(str).value_counts().index] 
label += ['2. {}'.format(typ) for typ in df['Type 2'].astype(str).value_counts().index]
label_dict = {l:n for n, l in enumerate(label)}

# Convert types to source and target for the diagram
plot_df = df.astype(str).groupby(['Type 1', 'Type 2']).Name.count().reset_index()
plot_df['Type 1'] = plot_df['Type 1'].map(lambda x: label_dict['1. '+x])
plot_df['Type 2'] = plot_df['Type 2'].map(lambda x: label_dict['2. '+x])


# Create the diagram
data = dict(type = 'sankey',
            customdata = '00ff00',
            node = dict(pad = 5,
                        thickness = 10,
                        line = dict(color = '#000000',
                                    width = 0.0),
                        label = label,
                        color = '#8b0000'),
            link = dict(source = plot_df['Type 1'],
                        target = plot_df['Type 2'],
                        value = plot_df['Name'],
                        color = '#333333'),
            orientation = 'h',
            valueformat = '.0f',
            valuesuffix = ' Pokemon')

layout =  dict(title = title,
               width = None,
               height = 600,
               font = dict(size = 12))

fig = dict(data=[data], layout=layout)
iplot(fig)


# # <a id=13>2D Histogram</a>

# In[ ]:


title = '2D Histogram: Pokemon Attack And Defense'

data = go.Histogram2d(x = df.Attack,
                      y = df.Defense,
                      histnorm = 'count', # 'probability'
                      autobinx = False,
                      xbins = dict(start = 0,
                                   end = 200,
                                   size = 10),
                      autobiny = False,
                      ybins = dict(start = 0, 
                                   end = 250,
                                   size = 10),
                      colorscale = 'Jet',
                      colorbar = dict(title = 'Count'))

layout = go.Layout(title = title,
                   xaxis = dict(title = 'Attack',
                                ticks = '', 
                                showgrid = False, 
                                zeroline = False, 
                                nticks = 20 ),
                   yaxis = dict(title = 'Defense',
                                ticks = '', 
                                showgrid = False, 
                                zeroline = False, 
                                nticks = 20),
                   autosize = False,
                   height = 550,
                   width = 550,
                   hovermode = 'closest')

fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=14>2D Contour</a>

# In[ ]:


title = '2D Contour: Pokemon Sp. Atk And Sp. Def'

data = go.Histogram2dContour(x = df['Sp. Atk'],
                             y = df['Sp. Def'],
                             colorscale = 'Jet',
                             contours = dict(showlabels = True,
                                             labelfont = dict(family = 'Raleway',
                                                              color = 'white')),
                             hoverlabel = dict(bgcolor = 'white',
                                               bordercolor = 'black',
                                               font = dict(family = 'Raleway',
                                                           color = 'black')))

layout = go.Layout(title = title)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=15>Ternary</a>

# In[ ]:


title = 'Ternary: Pokemon HP-Attack-Defense Triangle'

rawData = df.drop(['HP', 'Attack', 'Defense'], axis=1).join(df[['HP', 'Attack', 'Defense']].div((df.HP + df.Attack + df.Defense), axis=0))[['Name', 'HP', 'Attack', 'Defense']].T.to_dict().values()

def makeAxis(title, tickangle): 
    return {'title': title,
            'titlefont': {'size': 20},
            'tickangle': tickangle,
            'tickfont': {'size': 15},
            'tickcolor': '#ffffff',
            'ticklen': 5,
            'showline': True,
            'showgrid': True}

data = [{'type': 'scatterternary',
         'mode': 'markers',
         'a': [i for i in map(lambda x: x['HP'], rawData)],
         'b': [i for i in map(lambda x: x['Attack'], rawData)],
         'c': [i for i in map(lambda x: x['Defense'], rawData)],
         'text': [i for i in map(lambda x: x['Name'], rawData)],
         'marker': {'color': '#000000',
                    'size': 10,
                    'line': { 'width': 1 }}}]

layout = {'title' : title,
          'ternary': {'sum': 100,
                      'aaxis': makeAxis('HP', 0),
                      'baxis': makeAxis('<br>Attack', 45),
                      'caxis': makeAxis('<br>Defense', -45)}}

fig = {'data': data, 'layout': layout}
iplot(fig)


# # <a id=16>Radar Chart</a>

# In[ ]:


title = 'Radar Chart: Pokemon Features'

plot_df = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']].groupby('Legendary').median()

data_1 = go.Scatterpolar(r = plot_df.loc[True],
                        theta = plot_df.columns,
                        fill = 'toself',
                        name = 'Legendary')

data_2 = go.Scatterpolar(r = plot_df.loc[False],
                        theta = plot_df.columns,
                        fill = 'toself',
                        name = 'Common')


layout = go.Layout(title = title,
                   polar = dict(radialaxis = dict(visible = True,
                                                  range = [0, 125])),
                   showlegend = True)

fig = go.Figure(data=[data_1, data_2], layout=layout)
iplot(fig)


# # <a id=17>Parallel Coordinates</a>

# In[ ]:


title = 'Parallel Coordinates: Pokemon Features'

map_dict = { typ:i for i, typ in enumerate(df[['Type 1', 'Type 2']].astype(str).stack().unique())}

data = go.Parcoords(line = dict(color = '#cccccc',
                                showscale = False,
                    reversescale = True),
                    dimensions = list([dict(label = 'Type 1', values = df['Type 1'].astype(str).map(lambda x: map_dict[x])),
                                       dict(label = 'Type 2', values = df['Type 2'].astype(str).map(lambda x: map_dict[x])),
                                       dict(label = 'Total', values = df['Total']),
                                       dict(label = 'HP', values = df['HP']),
                                       dict(label = 'Attack', values = df['Attack']),
                                       dict(label = 'Sp. Atk', values = df['Sp. Atk']),
                                       dict(label = 'Sp. Def', values = df['Sp. Def']),
                                       dict(label = 'Speed', values = df['Speed']),
                                       dict(label = 'Generation', values = df['Generation'])]))

layout = go.Layout(title = title)

fig = go.Figure(data=[data], layout=layout)
iplot(fig)


# # <a id=18>Subplots</a>

# In[ ]:


title = 'Subplots: Pokemon Features'

data_1 = go.Violin(y = df.HP,
                   text = df.Name,
                   name='HP')
data_2 = go.Violin(y = df.Attack,
                   text = df.Name,
                   name='Attack',
                   xaxis='x2',
                   yaxis='y2')
data_3 = go.Violin(y = df.Defense,
                   text = df.Name,
                   name='Defense',
                   xaxis='x3',
                   yaxis='y3')
data_4 = go.Violin(y = df.Speed,
                   text = df.Name,
                   name='Speed',
                   xaxis='x4',
                   yaxis='y4')

layout = go.Layout(title = title,
                   xaxis=dict(domain=[0, 0.45]),
                   yaxis=dict(domain=[0, 0.45]),
                   xaxis2=dict(domain=[0.55, 1]),
                   yaxis2=dict(domain=[0, 0.45],
                               anchor='x2'),
                   xaxis3=dict(domain=[0, 0.45],
                               anchor='y3'),
                   yaxis3=dict(domain=[0.55, 1]),
                   xaxis4=dict(domain=[0.55, 1],
                               anchor='y4'),
                   yaxis4=dict(domain=[0.55, 1],
                               anchor='x4'))

fig = go.Figure(data=[data_1, data_2, data_3, data_4], layout=layout)
iplot(fig)


# I hope you had fun experimenting with the graphs.<br>
# Have a good day!

# In[ ]:




