#!/usr/bin/env python
# coding: utf-8

# # Plotly
# Plotly is one of my favorite data visualization packages for Python. The wide variety of plots and the level of customization available give the users a high amount of control on how the chart looks. As I learn more about how to work with Plotly, I want to experiment with different chart types through kernels as a way for me to practice and also for the Kaggle community to know how to use them.

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

# ## Scatterplots
# In Plotly, the Scatter function is used for scatterplots, line plots and bubble charts. We will just explore the scatterplots here. Scatterplots are a good way to examine the relationship between two variables, usually both of them continuous. It can show us if there is a clear correlation between the two variables or not.
# 
# For this notebook, I am using the King County House Sales dataset. Intuitively, house prices do depend on how big the house is, how many bathrooms there are, how old the house is, etc. Let us examine these relationships through a series of scatterplots.

# Plotly charts can be published "offline", which means you do not need an API key and can be used on your local machine or in Jupyter notebook environments. Your chart will be available for download as a PNG file.
# 
# Plotly charts can also be published "online", which means you have to have a Plotly account, an API key and your charts will be added to your account and can be published for wide distribution and consumption.
# 
# We will be using the offline mode for our purposes here. Let us import the required packages now.

# In[ ]:


import plotly.offline as ply
import plotly.graph_objs as go
ply.init_notebook_mode(connected=True)


# Let us read in the dataset and look at the data.

# In[ ]:


import pandas as pd
import numpy as np

df=pd.read_csv('../input/kc_house_data.csv')

df.head()


# The dataset has a nice mix of continuous and categorical independent variables, and a continuous dependent variable (price). Let us try plotting the price against the living room area (sqft_living15). Note that I am using scattergl instead of scatter as it has better performance. For smaller datasets with fewer no. of data points, scatter should work fine.

# In[ ]:


trace1 = go.Scattergl(
    x=df.sqft_living15,
    y=df.price,
    mode='markers',
    marker=dict(
        opacity=0.5
    ),
    #showlegend=True
)
data=[trace1]

layout = go.Layout(
    title='Price vs. Living Room Area',
    xaxis=dict(
        title='Living room area (sq. ft.)'
    ),
    yaxis=dict(
        title='Price ($)'
    ),
    hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# That looks like a nice plot showing some relation between the living room area and the price. I think the relation would be better demonstrated if we plot the log(price) instead.

# In[ ]:


dataPoints = go.Scattergl(
    x=df.sqft_living15,
    y=np.log(df.price),
    mode='markers',
    marker=dict(
        opacity=0.25,
        line=dict(
            color='white'
        )
    ),
    name='Data points'
)

data=[dataPoints]

layout.update(
    yaxis=dict(
        title='Log(Price)'
    )
)

figure.update(
    data=data,
    layout=layout
)
ply.iplot(figure)


# Usually, scatterplots showing a linear relationship are accompanied by the "line of best fit". If you are familiar with the Seaborn visualization package, you are probably aware that it gives an easy way to plot a line of best fit, as shown below:

# In[ ]:


import seaborn as sns

sns.regplot(df.sqft_living15, np.log(df.price))


# Let us see how we can add a line of best fit to our Plotly scatterplot. This line will be an additional trace in the data component.

# In[ ]:


m,b = np.polyfit(df.sqft_living15, np.log(df.price), 1)
bestfit_y = (df.sqft_living15 * m + b)


# In[ ]:


lineOfBestFit=go.Scattergl(
    x=df.sqft_living15,
    y=bestfit_y,
    name='Line of best fit',
    line=dict(
        color='red',
    )
)

data=[dataPoints, lineOfBestFit]
figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Let us now look at a variation in the scatterplot. How can we show categories in the scatterplot through color? For example, does the data differ based on the no. of floors, bedrooms and bathrooms in the house? We can examine that by passing the color parameter to the marker in the Scatter function, as shown below:

# In[ ]:


dataPoints = go.Scattergl(
    x=df.sqft_living15,
    y=np.log(df.price),
    mode='markers',
    text=[f'Living Room Area:{df.at[i, "sqft_living15"]} sq.ft.<br>Grade:{df.at[i, "grade"]}<br>Price:${df.at[i, "price"]}' for i in range(len(df))],
    marker=dict(
        opacity=0.75,
        color=df.grade,
        showscale=True,
        colorscale='Jet',
        colorbar=dict(
            title='Grade'
        ),
    ),
    name='Data points'
)

data=[dataPoints]

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)


# Note that this gives a "colorscale" instead of a legend with different "grades" indicated in separate colors. Plotly assigns separate colors when there are multiple traces. We can accomplish it this way:

# In[ ]:


grades=sorted(df.grade.unique())


# In[ ]:



data=[]
for g in grades:
    df_grade=df[df.grade==g]
    data.append(
        go.Scattergl(
            x=df_grade.sqft_living15,
            y=np.log(df_grade.price),
            mode='markers',
            text=[f'Living Room Area:{df_grade.at[i, "sqft_living15"]} sq.ft.<br>Grade:{df_grade.at[i, "grade"]}<br>Price:${df_grade.at[i, "price"]}' for i in df_grade.index],
            marker=dict(
                opacity=0.75,
            ),
            name='Grade:'+str(g)
        )
    )

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)    


# We can clearly see that in addition to the living room area, the grade of the house (assigned by King County) also affects the price of the house. But, what if we want to see the impact of grade, condition and other variables in this chart together at the same time? The answer: [subplots](https://plot.ly/python/subplots/).
# 
# Let us draw a stacked subplot of the chart shown above, but with no. of bedrooms, no. of bathrooms, condition, grade and waterfront as parameters.
# 
# In Plotly, we make subplots by adding the traces to the figure and specifying their position in the plot. Let's see how.

# The subplots functionality is available under plotly.tools

# In[ ]:


subplots=['No. of bedrooms', 'No. of bathrooms', 'Condition', 'Grade', 'Waterfront']
subplot_cols=['bedrooms', 'bathrooms', 'condition', 'grade', 'waterfront']

from plotly.tools import make_subplots
figure=make_subplots(rows=5, cols=1, subplot_titles=['Breakup by '+col for col in subplots])

for i in range(len(subplots)):
    col_name=subplots[i]
    col=subplot_cols[i]
    col_values=sorted(df[col].unique())
    for value in col_values:
        df_subset=df[df[col]==value]
        trace=go.Scattergl(
            x=df_subset.sqft_living15,
            y=np.log(df_subset.price),
            mode='markers',
            text=[f'Living Room Area:{df_subset.at[i, "sqft_living15"]} sq.ft.<br>{col_name}:{df_subset.at[i, col]}<br>Price:${df_subset.at[i, "price"]}' for i in df_subset.index],
            marker=dict(
                opacity=0.75,
            ),
            name=col_name+':'+str(value),
            showlegend=False
        )
        figure.append_trace(trace, i+1, 1)

figure['layout'].update(
    height=2000, 
    title='Price vs. Living room area - subplots', 
    hovermode='closest',
    xaxis=dict(title='Living room area (sq. ft.)'),
    xaxis2=dict(title='Living room area (sq. ft.)'),
    xaxis3=dict(title='Living room area (sq. ft.)'),
    xaxis4=dict(title='Living room area (sq. ft.)'),
    xaxis5=dict(title='Living room area (sq. ft.)'),
    yaxis=dict(title='Log(Price)'),
    yaxis2=dict(title='Log(Price)'),
    yaxis3=dict(title='Log(Price)'),
    yaxis4=dict(title='Log(Price)'),
    yaxis5=dict(title='Log(Price)'),
)

ply.iplot(figure)


# Grade looks like a very clear distinguishing factor to determine price along with the living room area. The other variables seem to have some impact, but we may have to conduct a regression analysis to examine that.
# 
# I hope this gave you some insight on how to use scatterplots in plotly. I will practice column / bar charts in a subsequent notebook.
