#!/usr/bin/env python
# coding: utf-8

# # Easy Plotly
# 
# **Plotly** is one of the most widely used Data Visualization Library available in both **R and Python**, used for creating interactive plots. The best thing about Plotly is that it is free and graphs rendered using Plotly are interactive which can be helpful in drilling down on categories. However the downside is it's lengthy code. Plotly is very verbose i.e we need to write lengthy codes for creating even the simplest graph.
# 
# However while working on a recent project, I came across a library which made my life easier. Using this library, I rendered Plotly graphs using code similar to **Matplotlib**. The library I used is **CUFFLINKS**. Cufflinks is a library which binds Plotly graphs directly onto **Pandas** dataframe. The code is very similar to that of Matplotlib, thereby making it easier to interpret. This library is probably available for **Python only**. If I find R equivalent for this, I will surely post it.
# 
# This notebook is just an introduction to Cufflinks, so that more and more people can use it. Hope this notebook is useful. **Do Upvote if this notebook proves to be useful**.

# In[ ]:


import pandas as pd
import numpy as np
import cufflinks as cf
import plotly
plotly.offline.init_notebook_mode()
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')


# # Most Popular Games of All Time

# In[ ]:


fig = data.sort_values('Global_Sales', ascending=False)[:10]
fig = fig.pivot_table(index=['Name'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'], aggfunc=np.sum)
fig = fig.sort_values('Global_Sales', ascending=True)
fig = fig.drop('Global_Sales', axis=1)
fig = fig.iplot(kind='barh', barmode='stack' , asFigure=True) #For plotting with Plotly
fig.layout.margin.l = 350 #left margin distance
fig.layout.xaxis.title='Sales in Million Units'# For setting x label
fig.layout.yaxis.title='Title' # For setting Y label
fig.layout.title = "Top 10 global game sales" # For setting the graph title
plotly.offline.iplot(fig) # Show the graph


# Doesn't the code look simpler and compact as compared to the generic Plotly code we used to write??

# ## Critic Score Distribution

# In[ ]:


data['Critic_Score'].iplot(kind='histogram',opacity=.75,title='Critic Score Distribution')


# 
# ## Yearly Sales By Region
# 
# Lets see a one liner code to render a plot using Plotly.
# 

# In[ ]:


fig = data.pivot_table(index=['Year_of_Release'], values=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'], 
                       aggfunc=np.sum, dropna=False,).iplot( asFigure=True,xTitle='Year',yTitle='Sales in Million',title='Yearly Sales By region')
plotly.offline.iplot(fig)


# Thanks to Cufflinks, we now have a one liner code for Plotly. Just imagine the number of lines of code we would require to draw such a simple graph using native Plotly code. Lets now plot some complex graphs.
# 
# ## Sales per Year by Genre

# In[ ]:


fig = (data.pivot_table(index=['Year_of_Release'], values=['Global_Sales'], columns=['Genre'], aggfunc=np.sum, dropna=False,)['Global_Sales']
        .iplot(subplots=True, subplot_titles=True, asFigure=True, fill=True,title='Sales per year by Genre'))
fig.layout.height= 800
fig.layout.showlegend=False 
plotly.offline.iplot(fig)


# The above graph shows the trend in sales by Genre. But if u see carefully, not all the graphs have the same y-axis range, i.e some graphs have values higher than 100, whereas some have just below 20. So just looking at the graph can lead to faulty decisions. Lets caliberate the y-axis onto a common axis

# In[ ]:


fig = (data.pivot_table(index=['Year_of_Release'], values=['Global_Sales'], columns=['Genre'], aggfunc=np.sum, dropna=False,)['Global_Sales']
        .iplot(subplots=True, subplot_titles=True, asFigure=True, fill=True,title='Sales per year by Genre'))
fig.layout.height= 800
fig.layout.showlegend=False 
for key in fig.layout.keys():
    if key.startswith('yaxis'):
        fig.layout[key].range = [0, 145]
plotly.offline.iplot(fig)


# ## Publisher Releases By Years

# In[ ]:


top_publish=data['Publisher'].value_counts().sort_values(ascending=False)[:10]
top_publish=data[data['Publisher'].isin(top_publish.index)].groupby(['Publisher','Year_of_Release'])['Name'].count().reset_index()
top_publish=top_publish.pivot('Year_of_Release','Publisher','Name')
top_publish[top_publish.columns[:-1]].iplot(kind='heatmap',colorscale='RdYlGn',title='Publisher Games Releases By Years')


# I hope this notebook was useful. Following is a link for more graphs [More Graphs](https://github.com/santosjorge/cufflinks). 
# 
# If u liked this notebook, do have a look on my latest work and share some feedback. I am sure you will like this one too.  
# 
# ### For R users, Please follow this great Kernel by Nayan Solanki: [Notebook](https://www.kaggle.com/nayansolanki2411/insights-of-donors-choose)
# 
# **DO UPVOTE** if this notebook was useful.
# 
