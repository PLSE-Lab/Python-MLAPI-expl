#!/usr/bin/env python
# coding: utf-8

# # In this topic I would like to show you a different methods of visualisation and it's types.
# 
# Not so long ago I had a conversation with a "very good" data scientist about visualizations and its usage in the projects. And he argued a lot that he can do anything without it and visualization process is a lack of time and does not deserve his attention. Hmmmm....
# 
# **Is it really so? And do a data scientists need visualization at all?**
# 
# I like to answer to this question with an example that was invented by the **talented mathematician Francis Encombe in 1973 year.** With this example, he wanted to show the importance of visualization for data analysis and the effect of data on statistical indicators. 
# 
# The point is that he came up with several random samples that have the same statistical indicators.
# 
# ![random_samples.jpg](attachment:random_samples.jpg)

# So as you can see above the dataFrame has 4 random samples. And if we calculate a statistics for each individual sample we will get the same result for each of them:
# 
# * Sample Mean for ***x*** = 9
# * Sample Variance of ***x*** = 11
# * Sample Mean for ***y*** =11.5
# * Sample Variance of ***y*** = 4.125
# * Correlation between ***x*** and ***y*** = 0.816
# 
# If you don't trust me try your own! 
# 
# **And so is it really all 4 random samples are the same?**
# 

# # Nope!
# 
# ![1.jpg](attachment:1.jpg)
# 
# And I found many of such examples where you can find a lot of funny examples https://www.autodeskresearch.com/publications/samestats
# 
# So from this point I will immerse you into the world of data visualization and show you with examples and what you should use (and what should not, but you can if you want). So let's go!

# # Python's libriaries for data visualization
# 
# ![libs.jpg](attachment:libs.jpg)
# 
# 
# I will point out a list of libriaries that I use the most time. Here are they:
#    * **[matplotlib](http://matplotlib.org/3.1.1/contents.html)** - the most classical libriary (hello from 90's :)
#             1. This is the first data visualization libriary in Python.
#             2. Very Very flexible and powerfull. It is relatively simple to use for complete beginners.
#             3. Styles from 90's.
#             4. Exist wrappers - pandas and seaborn.
#    * **[seaborn](http://seaborn.pydata.org/)** 
#             1. Based on matplotlib libriary.
#             2. Complex visualizations for couple of lines of code.
#             3. Very attractive styles (by default and tuned).
#             4. And if you want to add or change something you need to know matplotlib.
#    * **[plotly (+dash)](http://dash.plot.ly/)** - my favourite 
#             1. Very interactive data visualizations.
#             2. Relatively simple API and you can tune it for your own needs.
#             3. Beatiful default styles for graphs.
#             4. You can use dash for building and encoding your visualizations into different web applications.
#    * **[ggpolot](http://ggplot2.tidyverse.org/reference/)**
#             1. Based on ggplot2 from R's language libriary.
#             2. The Grammar of graphics Zen: complex component layers.
#             3. It is easy tu use than matplotlib but it is less flexible.
#    * **[bokeh](http://bokeh.pydata.org/en/latest/)**
#             1. Same The Grammar of graphics Zen like in ggplot.
#             2. Very interactive visualizations.
#             3. Very complex API (not flexible).
#             4. Complex (very hard) libriary to use. But if you will know it you can do anything.
#    * **[pygal](http://www.pygal.org/en/stable/documentation/)**
#             1. Interactivity in visualizations.
#             2. Graphs in SVG formats (it is not a good solution for the big data frames).
#             3. Simple API that you can use. 

# # Visualization examples and codes
# ## *Matplotlib review*

# In[ ]:


# if you dont have the libriaries show above you can upload it by using 
# pip install seaborn
# pip install plotly
# pip install ggplot
# pip install matplotlib


# In[ ]:


# pip install future
from __future__ import (absolute_import, division, print_function, 
                        unicode_literals)
# turn off warnings
import warnings
warnings.simplefilter('ignore')

# inline visualizations 
get_ipython().run_line_magic('pylab', 'inline')
# turn the visualization into SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# and lets resize default size of graphs
from pylab import rcParams
rcParams['figure.figsize'] = 6,5

# and last point lets import libriaries for working with data and data manipulation
import pandas as pd
import numpy as np
import seaborn as sns


# Lets upload dataframe with which we will be working. I choose a dataset with videogame information (ranks, sales, regions and etc.) from one of the **[Kaggle's datasets](http://www.kaggle.com/rush4ratio/video-game-sales-with-ratings).** There is some empty data that will be deleted from the dataset. So lets go!

# In[ ]:


# I forgot the path on Kagle so lets find out where we are now:
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/video_games_sales.csv')
print(df.shape)


# In[ ]:


df.info()


# In[ ]:


# As you see there are not all data for "Critic_Score", "Critic_count", "Developer" and "Rating" 
# Usually I delete it (but not always)
df = df.dropna()
print(df.shape)
df.info()


# In[ ]:


# As mentiod from df.info() not all features have an appropriate data-type. Change it to the another data-type
df['User_Score'] = df.User_Score.astype('float64')
df['Year_of_Release'] = df.Year_of_Release.astype('int64')
df['User_Count'] = df.User_Count.astype('int64')
df['Critic_Count'] = df.Critic_Count.astype('int64')


# In[ ]:


df.head()


# In[ ]:


# I think I will analyze not all features. In this dataframe we have 6825 rows and 16 columns(features)
# Lets leave features that are the most meaningfull
meaningfull_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Global_Sales',
                    'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating']

df[meaningfull_cols].head(5)


# Let's start with the simplest and often convenient way to visualize data from <code>**Pandas DataFrame**</code> the usage of the <code>**plot**</code> function. 
# For example let's create a **Sales** graph in different **Countries** by the **Year_of_release**. First of all let's filter out only the columns we need, then 
# calculate the total sales by year and then call the <code>**plot**</code> function without any arguments for the final dataframe.
# 
# And I want to point out that <code>**Pandas Libriary**</code> already have wrapper for the <code>**Matplotlib Libriary**</code>

# In[ ]:


[x for x in df.columns if 'Sales' in x]


# In[ ]:


# And I want to mention that NA_Sales stands for North_American_Sales and not for NA as you could think 
df1 = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']].groupby('Year_of_Release').sum()
df1.head()


# In[ ]:


df1.plot();


# In this case we focused on displaying sales trends in different regions. 
# Using the <code>**kind**</code> argument parameter you can change the type of the chart to any other (bar chart, pie chart and etc.)
# <code>**Matplotlib libriary**</code> allows you to make any kind of chart customization. On the chart you can change almost anything, 
# but you need to go through documentation for the purpose you need and find necessary parameters. 
# 
# For example the <code>**rot**</code> parameter is responsible for the slope of the labels on the x-axis and the <code>**figsize = (x, y)**</code>
# you can resize you chart to any size.

# In[ ]:


df1.plot(kind = 'bar', rot = 45, figsize = (10, 5));


# To show both the dynamics of sales and their breakdown by market you should use a **Stacked Bar Chart**.

# In[ ]:


df1[list(filter(lambda x: x != 'Global_Sales', df1.columns))].plot(kind = 'bar', rot = 45, stacked = True, figsize = (10, 5));


# In[ ]:


# stacked parameter is for visibility
df1[list(filter(lambda x: x != 'Global_Sales', df1.columns))].plot(kind = 'area', rot = 45, stacked = False, figsize = (10, 5));


# ### Histograms in matplotlib
# Histograms are very well suited for visualizing **various kinds of distributions.** On below examples I will show distribution of critic scores in the dataframe.
# Histograms in matplotlib are relatively simple to plot. You need just select a feature you want to plot (for example **Critic_Score**) and add a method <code>**.hist()**</code>

# In[ ]:


df.Critic_Score.hist(figsize = (10, 5));


# The more beautiful way:

# In[ ]:


ax = df.Critic_Score.hist(figsize = (10, 5));
ax.set_title('Critic Score distribution');
ax.set_xlabel('Critic Score');
ax.set_ylabel('Games');


# In[ ]:


# You can choose the number of bins for the distribution by calling it bins = #
ax = df.Critic_Score.hist(figsize = (10, 5), bins = 25);
ax.set_title('Critic Score distribution');
ax.set_xlabel('Critic Score');
ax.set_ylabel('Games');


# # *Seaborn libriary review*
# 
# Seaborn libriary is just an API of high level based on matplotlib libriary. Seaborn includes more attractive default styles and its customization for the charts. 
# Seaborn libriary also have different complex styles and types of visualizations than you can do with a several rows of code (in mathplotlib it will be messy).
# 
# The first complex chart is <code>**pair plot (scatter plot matrix)**</code>. This type of chart helps us to see the different types of correlation between values and 
# its distributions. So let's see it.

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
sns_plot = sns.pairplot(df[['Global_Sales', 'User_Score', 'Critic_Score']]);
# you can save the chart by
# sns_plot.savefig('#name_of_chart.png')


# In[ ]:


# Also seaborn can visualize distribution of quantitative value in different ways
# Joint_plot - is a hybrid of scatter_plot and histogram. 
# Lets see how it works on two values Critic_Score and User_Score
sns.jointplot(x = 'Critic_Score', y = 'User_Score', data = df, kind = 'scatter');


# In[ ]:


sns.jointplot( x = 'Critic_Score', y = 'User_Score', data = df, kind = 'reg');


# Also what I really like in seaborn's libriary is **heatmap** diagrams. It is really powerfull and from this feature we can gain a lot of insights.
# Using the heatmap we can see how for example Rates are varies from platform to platform. 

# In[ ]:


platform_gender_sales = df.pivot_table(index = 'Platform', columns = 'Genre', values = 'Global_Sales', aggfunc = sum).fillna(0).applymap(float)
platform_gender_sales.head()


# In[ ]:


sns.heatmap(platform_gender_sales, annot = True, fmt = '.0f', linewidths = 0.7);


# So as you can see a seaborn is very powerfull libriary. Using this you can create any type of charts (box plots, histograms, line charts, distribution or correlation chart, scatter plot and etc.). If you want more examples you can read a documentation of every libriary. Links to the documentations I have placed in the beggining of this tutorial. 
# 
# And lastly I want to show you how to work in my favourite libriary called <code>**Plotly**</code>. It is a really powerfull libriary that I have ever seen among Python libriaries for visualization purposes. Let's see how it works and what kind of things you can possibly do using **Plotly**

# # *Plotly review* 
# ![](http://)Plotly is an open-source libriariy which allows to create different **interactive** charts. The advantage of interactive charts is that you can see any quantitative value by pointing a mouse cursor on the chart and increase or decrease a scale of the chart. Let's see how it works on practice!
# 
# ## I dont know why but Kaggle notebook do not appropriately show the Plotly output. If you want to see it simply download this notebook. Sorry.

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected = True)


# In[ ]:


global_sales_df = df.groupby('Year_of_Release')[['Global_Sales']].sum()
global_sales_df.head(5)


# In[ ]:


released_years_df = df.groupby('Year_of_Release')[['Name']].count()
released_years_df.head(5)


# In[ ]:


years_df = global_sales_df.join(released_years_df)
years_df.columns = ['Global_Sales', 'Number_of_Games']
years_df.head()


# In <code>**Plotly**</code> the visualization is constructed by the Figure object, which consists from data (an array of lines, called **traces**) and from design / style for which the layout object is responsible. In simple cases you can call the <code>**iplot**</code> function and just plot the data from its **traces**. Let see it on the example:

# In[ ]:


# declare a trace which is an array and specify the design
# go.Scatter - specifies type of chart
# trace0 simply just declaring a firts line on the graph
trace0 = go.Scatter(
    # what should be on x-axis
    x = years_df.index,
    # what should be on y-axis
    y = years_df.Global_Sales,
    # Title for the Chart
    name = 'Global Sales'
)
# declare a second Scatter (second line on the graph)
trace1 = go.Scatter(
    # Specify x-axis
    x = years_df.index,
    # and y-axis
    y = years_df.Number_of_Games,
    # Title of the chart
    name = 'Number of games released'
)

# collect all the traces (arrays) in separate dataframe
data = [trace0, trace1]
# choosing a single title
layout = {'title': 'Statistics of video games'}
# Plotting the figure
fig = go.Figure(data = data, layout = layout)

iplot(fig, show_link = False)


# In[ ]:


# if you want to save your graph just use the next code
# plotly.offline.plot(fig, filename = 'years_stats_sales.#specify_format_after_dot', show_link = False);


# Also Plotly is best suitable solution for visualizing such data as market share, prices, segments, distributions and etc. Let's see on market share of game platforms, calculated by quantity of games released and its sales for the all time. The best suited graphs are bar chart or pie chart. I will plot a bar chart beacuse it is easily to read.

# In[ ]:


platform_sales_global_df = df.groupby('Platform')[['Global_Sales']].sum()
released_df = df.groupby('Platform')[['Name']].count()
platforms_df = platform_sales_global_df.join(released_df)


# In[ ]:


platforms_df.columns = ['Global_Sales', 'Number_of_Games']
platforms_df.sort_values('Global_Sales', inplace = True)
platforms_df = platforms_df.apply(lambda x: 100 * x / platforms_df.sum(), axis = 1)
platforms_df.head()


# In[ ]:


# Finally lets plot the data on chart

# again create a traces where specify the chart type and its axis
trace0 = go.Bar(
    x = platforms_df.index,
    y = platforms_df.Global_Sales,
    name = 'Global Sales',
    orientation = 'v'
)

trace1 = go.Bar(
    x = platforms_df.index,
    y = platforms_df.Number_of_Games,
    name = 'Number of games released',
    orientation = 'v'
)

data = [trace0, trace1]
layout = {'title': 'Platforms share'}

fig = go.Figure(data = data, layout = layout)

iplot(fig, show_link = False)


# In[ ]:


# We can interactively represent the dependency between mean User_Score and Critic_Score and its influence on Global_Sales
# To do it we need to join two tables with scores and sales
scores_genres = df.groupby('Genre')[['Critic_Score', 'User_Score']].mean()
sales_genres = df.groupby('Genre')[['Global_Sales']].sum()
genres_sales = scores_genres.join(sales_genres)

genres_sales.head()


# In[ ]:


# So finally plot the data on char. I choose a scatter plot because it will show dependencies
trace0 = go.Scatter(
            x = genres_sales.Critic_Score,
            y = genres_sales.User_Score,
            mode = 'markers+text',
            text = genres_sales.index)

data = [trace0]
layout = {'title': 'Influence of User and Critic Scores on Sales'}

fig = go.Figure(data = data, layout = layout)
iplot(fig, show_link = False)


# In[ ]:


# From this scatter plot we can modify it and create a bubble chart which will show the amount of sales that was calculated before
genres_sales.index


# In[ ]:


trace0 = go.Scatter(
    x = genres_sales.Critic_Score,
    y=genres_sales.User_Score,
    mode = 'markers+text',
    text = genres_sales.index,
    marker = dict(
        size = 1/10*genres_sales.Global_Sales,
        color = [
            'aqua', 'azure', 'beige', 'lightgreen',
            'lavender', 'lightblue', 'pink', 'salmon',
            'wheat', 'ivory', 'silver'
        ]
    )
)

data = [trace0]
layout = {
    'title': 'Influence of User and Critic Scores on Sales',
    'xaxis': {'title': 'Critic Score'},
    'yaxis': {'title': 'User Score'}
}

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)


# So this is it. I wanted to show you how you can use different libriaries on visualization process and gain a powerfull insights from it. It is your choice which of you should use, but the best of them is Plotly libriary which I use on daily basis. Besides this if you want to be real expert there are d3.js libriary for JavaScript. It is also very powerfull and using it you can create a lot of different charts in different manner. 
# 
# Hope that this tutorial will help you in future. 
# Thank you!
# 
# P.S. If you found grammar mistakes in my English, sorry. I didn't have a practice for a long time.
#      Best to you all and thank you!
