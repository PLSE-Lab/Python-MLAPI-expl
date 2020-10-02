#!/usr/bin/env python
# coding: utf-8

# # Plotly
# Plotly is one of my favorite data visualization packages for Python. The wide variety of plots and the level of customization available give the users a high amount of control on how the chart looks. As I learn more about how to work with Plotly, I want to experiment with different chart types through kernels as a way for me to practice and also for the Kaggle community to know how to use them.
# 
# ## Notes about Plotly
# Plotly charts have two major components: data and layout.
# 
# Data - this represents the data that we are trying to plot. This informs the Plotly's plotting function of the type of plots that need to be drawn. It is basically a list of plots that should be part of the chart. Each plot within the chart is referred to as a 'trace'.
# 
# Layout - this represents everything in the chart that is not data. This means the background, grids, axes, titles, fonts, etc. We can even add shapes on top of the chart and annotations to highlight certain points to the user.
# 
# The data and layout are then passed to a "figure" object, which is in turn passed to the plot function in Plotly.
# 
# - Figure
#     - Data
#         - Traces
#     - Layout
#         - Layout options

# In my previous experiments with Plotly, I tinkered with [scatterplots](https://www.kaggle.com/meetnaren/plotly-experiments-scatterplots), [bar (or column) plots](https://www.kaggle.com/meetnaren/plotly-experiments-bar-column-plots) and [line plots](https://www.kaggle.com/meetnaren/plotly-experiments-line-plots). Plotly has a very powerful library of plots beyond just these basic charts. I will try exploring a wide variety of plots in this notebook.

# ## The dataset
# The first dataset I am using is the Mobile App Store dataset. Let us import packages, read the dataset and take a first look at the data.

# In[ ]:


import pandas as pd
import numpy as np

appstore=pd.read_csv('../input/app-store-apple-data-set-10k-apps/AppleStore.csv')
appstore.head()


# In[ ]:


from pandas_summary import DataFrameSummary

dfs=DataFrameSummary(appstore)

dfs.summary().T


# We can see that there is a healthy mix of categorical (Genre, Content Rating) and continuous variables (Rating, Rating count, price) in the dataset. Let us create another variable to indicate whether an app is paid or free and one to indicate the size of the app in MB.

# In[ ]:


appstore['paid_free']=np.where(appstore.price>0, 'paid', 'free')
appstore['size_MB']=appstore.size_bytes/(1024*1024)


# Alright, time to Plotly! Let us start with one of the most useful statistical plots.
# 
# ## Box plots
# A box plot is a very useful method to depict the range and dispersion of a numerical variable, usually across different groups of a categorical variable. One can quickly see in what range most of the data lies, and can also easily see outliers (or the lack thereof). Let us investigate the price of apps across different genres.

# In[ ]:


import plotly.offline as ply
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.tools import make_subplots

ply.init_notebook_mode(connected=True)

import colorlover as cl
from IPython.display import HTML

colors=cl.scales['12']['qual']

chosen_colors=[j for i in colors for j in colors[i]]

print('The color palette chosen for this notebook is:')
HTML(cl.to_html(chosen_colors))


# Each genre needs to be a different trace in the plot, so let's cyle through the different genres and add them to the list of traces.

# In[ ]:


paid_apps=appstore[appstore.paid_free=='paid']

genres=list(paid_apps.groupby(['prime_genre']).price.quantile(.75, interpolation='linear').sort_values().reset_index().prime_genre)

data=[]
for i in genres:
    data.append(
        go.Box(
            y=paid_apps[appstore.prime_genre==i].price,
            name=i,
            marker=dict(
                color=chosen_colors[genres.index(i)]
            )
        )
    )

layout=go.Layout(
    title='<b>Price comparison across app genres</b>',
    xaxis=dict(
        title='App genre',
        showgrid=False
    ),
    yaxis=dict(
        title='Price'
    ),
    showlegend=False,
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# I think this is better represented in a horizontal box plot, instead of a vertical one.

# In[ ]:


paid_apps=appstore[appstore.paid_free=='paid']
genres=list(appstore.prime_genre.unique())
data=[]
for i in genres:
    data.append(
        go.Box(
            x=paid_apps[appstore.prime_genre==i].price,
            name=i,
            marker=dict(
                color=chosen_colors[genres.index(i)]
            )
        )
    )

layout=go.Layout(
    title='<b>Price comparison across app genres</b>',
    yaxis=dict(
        title='App genre',
        showgrid=False
    ),
    xaxis=dict(
        title='Price'
    ),
    showlegend=False,
    margin=dict(
        l=150
    ),
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# There are some outliers in the 'Education' genre (I wonder which apps cost USD250 and USD300!!!), because of which we are not able to view the box plots properly. Let us restrict our plot data to apps < $50.

# In[ ]:


paid_apps_lt_50=appstore[(appstore.paid_free=='paid') & (appstore.price<=50)]

genres=list(paid_apps_lt_50.groupby(['prime_genre']).price.quantile(.75, interpolation='linear').sort_values().reset_index().prime_genre)

data=[]
for i in genres:
    data.append(
        go.Box(
            x=paid_apps_lt_50[appstore.prime_genre==i].price,
            name=' '*15+i,
            marker=dict(
                color=chosen_colors[genres.index(i)]
            ),
        )
    )

layout['height']=800

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# We can see that among the paid apps, medical apps seem to have a higher range of prices, whereas weather apps seem to be on the lower end of the spectrum. Let us verify that by looking at the data.

# In[ ]:


paid_apps_lt_50[(paid_apps_lt_50.prime_genre == 'Medical') | (paid_apps_lt_50.prime_genre == 'Weather')][['prime_genre','price']].sort_values(by=['prime_genre', 'price'])


# We can see that there are quite a few paid apps in the Medical genre that cost more than USD9.99, whereas most of the paid Weather apps cost USD3 or lesser.

# ## Violin plots
# Violin plots are similar to box plots in that they are used to examine the distribution of a numerical variable across different groups of a categorical variable. But, violin plots also depict the density function of the numerical variable for each group, thus giving the user an idea of how common a certain value of the variable is.
# 
# To illustrate violin plots, let us examine the size of the app based on the genre. Does app size vary by genre?
# 
# Since there are a lot of genres, let us examine just the  genres that have at least 100 apps.

# In[ ]:


genres_count=appstore.groupby(['prime_genre']).id.count().reset_index().sort_values(by=['id'], ascending=False)

top_genres=list(genres_count[genres_count.id>100].prime_genre)

top_apps=appstore[appstore.prime_genre.isin(top_genres)].copy()

top_apps['text']=top_apps.track_name+': '+top_apps.size_MB.astype(str)+' MB'


# In[ ]:


data=[]

for g in top_genres:
    data.append(
        go.Violin(
            y=top_apps[top_apps.prime_genre==g].size_MB,
            x=top_apps[top_apps.prime_genre==g].prime_genre,
            name=g,
            text=top_apps[top_apps.prime_genre==g].text,
            marker=dict(
                color=chosen_colors[list(top_genres).index(g)]
            ),
            box=dict(
                visible=True
            ),
            jitter=1,
            #points=False
        )
    )

layout=go.Layout(
    title='<b>App size comparison across genres</b>',
    xaxis=dict(
        title='App genre',
        showgrid=False
    ),
    yaxis=dict(
        title='Size (MB)'
    ),
    showlegend=False,
    hovermode='closest',
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# This chart shows that Games, Book, Music, Education and Entertainment genres have some outlier apps that are really large in size, at more than 500MB.
# 
# We can also see that the outlier count in the 'Games' genre is far higher than any other genre. We have to remember that these are outliers, though. Let us restrict the data to apps less than 500MB to see the violin plots more clearly.

# In[ ]:


top_apps_lt_500MB = top_apps[top_apps.size_MB<500].copy()


# In[ ]:


data=[]

for g in top_genres:
    data.append(
        go.Violin(
            y=top_apps_lt_500MB[top_apps_lt_500MB.prime_genre==g].size_MB,
            x=top_apps_lt_500MB[top_apps_lt_500MB.prime_genre==g].prime_genre,
            name=g,
            text=top_apps_lt_500MB[top_apps_lt_500MB.prime_genre==g].text,
            marker=dict(
                color=chosen_colors[list(top_genres).index(g)]
            ),
            box=dict(
                visible=True
            ),
            jitter=.75,
            #points=False
        )
    )

layout=go.Layout(
    title='<b>App size comparison across genres</b>',
    xaxis=dict(
        title='<b>App genre</b>',
        showgrid=False
    ),
    yaxis=dict(
        title='<b>Size (MB)</b>'
    ),
    showlegend=False,
    hovermode='closest',
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# This plot helps us see the median for each category clearly, and we can see that Games and Education apps have a higer median app size in the 100 MB range, which is higher than the other genres.

# ## Bubble charts
# One of the most versatile, and the  most informative charts is the bubble chart. It helps us plot multiple dimensions into a single chart that is instantly interpretable without much difficulty. The multiple dimensions can be represented through the X-axis, Y-axis, size of the bubble, color of the bubble and also the saturation of the color. Bubble charts are just a variation of scatter plots, with the size of markers adding an extra dimension. Let us see an illustration using the medical cost dataset.

# In[ ]:


insurance=pd.read_csv('../input/insurance/insurance.csv')
insurance.head()


# Let us see what contributes to high insurance charges. Does it vary by gender, region or smoking behaviour?

# In[ ]:


def bubble(trace_col):
    data=[]
    trace_vals=list(insurance[trace_col].unique())
    for i in range(len(trace_vals)):
        data.append(
            go.Scatter(
                x=insurance[insurance[trace_col]==trace_vals[i]].age,
                y=insurance[insurance[trace_col]==trace_vals[i]].bmi,
                mode='markers',
                marker=dict(
                    color=chosen_colors[i*4+1],
                    opacity=0.5,
                    size=insurance[insurance[trace_col]==trace_vals[i]].charges/3000,
                    line=dict(
                        width=0.0
                    )
                ),
                text='Age:'+insurance[insurance[trace_col]==trace_vals[i]].age.astype(str)+'<br>'+'BMI:'+insurance[insurance[trace_col]==trace_vals[i]].bmi.astype(str)+'<br>'+'Charges:'+insurance[insurance[trace_col]==trace_vals[i]].charges.astype(str)+'<br>'+trace_col+':'+trace_vals[i],
                hoverinfo='text',
                name=trace_vals[i]
            )
        )

    layout=go.Layout(
        title='<b>Insurance cost comparison for different ages / BMIs</b>', 
        hovermode='closest',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(
            family='Segoe UI',
            color='#ffffff'
        ),
        xaxis=dict(
            title='<b>Age<b>'
        ),
        yaxis=dict(
            title='<b>BMI<b>'
        ),
        legend=dict(
            orientation='h',
            x=0,
            y=1.1
        ),
        shapes=[
            dict(
                type='rect',
                xref='x',
                x0=min(insurance.age)-1,
                x1=max(insurance.age)+1,
                yref='y',
                y0=18.5,
                y1=24.9,
                line=dict(
                    width=0.0
                ),
                fillcolor='rgba(255,255,255,0.2)'
            )
        ],
        annotations=[
            dict(
                xref='x',
                x=45,
                yref='y',
                y=18.5,
                text='Healthy BMI zone',
                ay=35
            )
        ]
    )
    
    figure = go.Figure(data=data, layout=layout)
    
    ply.iplot(figure)


# In[ ]:


bubble('sex')


# It's hard to discern any clear distinguishing pattern based on gender. But we can see that almost all the bigger bubbles are outside the healthy BMI zone.
# 
# Let us check if the geographical region influences the medical insurance cost. 

# In[ ]:


bubble('region')


# It is surprising to see that so many of the instances where the insurance charges are high are in the southeast region. It also seems that they are all in the high BMI range.
# 
# Let us see how smoking behavior influences insurance charges.

# In[ ]:


bubble('smoker')


# Wow. No surprises there. Almost all of the bigger bubbles we see on this chart are those of smokers.
# 
# I hope these charts gave you an idea of the usefulness of bubble charts. We were able to tell a story using four dimensions in each of these charts.
# 
# ## Heatmaps
# Heatmaps are a useful way to visualize a numerical variable in a matrix of two categorical variables. The magnitude of the numerical variable is represented through a colorscale. Let us examine the average insurance charges for different ages and regions through a heatmap.
# 
# I will create a new dataframe with ages as the indices and regions as the columns.

# In[ ]:


ages=insurance.age.unique().tolist()
regions=insurance.region.unique().tolist()
charge_matrix=pd.DataFrame(data=0, index=ages, columns=regions).sort_index()
charge_count=pd.DataFrame(data=0, index=ages, columns=regions).sort_index()


# In[ ]:


def create_charge_matrix(row):
    a=row['age']
    r=row['region']
    c=row['charges']
    charge_matrix.loc[a, r]+=c
    charge_count.loc[a, r]+=1    


# In[ ]:


insurance.apply(lambda row: create_charge_matrix(row), axis=1)

#Calculating average charges
charge_matrix /= charge_count


# In[ ]:


z=[]
for i in range(len(charge_matrix)):
    z.append(charge_matrix.iloc[i].tolist())


# In[ ]:


trace1=go.Heatmap(
    x=charge_matrix.columns.tolist(),
    y=charge_matrix.index.tolist(),
    z=z,
    colorscale='Electric'
)

data=[trace1]

layout=go.Layout(
    title='<b>Average insurance charges by age and region<b>',
    height=800,
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    ),
    xaxis=dict(
        title='<b>Region<b>'
    ),
    yaxis=dict(
        title='<b>Age<b>'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# As we saw earlier in one of our bubble charts, the southeast region seems to have higher insurance costs than any other region.
# 
# I hope you found this notebook useful to learn some more plotting techniques with Plotly. Let us explore geographical charts / plots in a subsequent notebook. Thanks for reading!
