#!/usr/bin/env python
# coding: utf-8

# # Exploring COVID19 related publications in a structured way
# 
# Following and in parallel to the recently released dataset [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) of scholarly articles, we provide the literature graph [LG-covid19-HOTP (10.5281/zenodo.3728215)](https://zenodo.org/record/3728216#.XoRsfKdfiV4) composed of not only articles (graph nodes) that are relevant to the study of coronavirus, but also citation links (graph edges) for facilitating navigation and search among the articles. The article records are related and connected, not isolated, in the same spirit of other existing literature graphs, and focused around the particular theme of covid-19 study.
# 
# The graph nodes include more than 800 hot-off-the-press (HOTP) articles since January 2020. The graph contains about one hundred thousand articles and nearly one million links. In addition to the dataset, we provide basic meta-data analysis and interactive visualization in terms of publication growth over time, ranking by citation, similarity in co-citation, and similarity in co-reference.

# ## So let's get started and explore the dataset

# First of all import all related libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# and list the dataset files

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The dataset comes with 2 files:
#  * **lg-covid19-hotp-article-info.csv** which contains the 100.000+ article as rows and 6 columns including *DOI, title, publication year, inCitCrossref, outCitCrossref* and *generation*
#  * **lg-covid19-hotp-literature-graph.mtx** which is a 103.861x103.861 sparse matrix with a non-zero entry at cell (i, j) indicating that article i was cited by article j(or article j referenced article i)

# ---

# ## First, lets take a look at the csv file using **pandas** library

# In[ ]:


df = pd.read_csv('/kaggle/input/lgcovid19hotp/lg-covid19-hotp-article-info.csv', encoding = "ISO-8859-1")
df.head()


# In[ ]:


df.describe()


# ---

# ## After viewing the basic metrics and statistics, we will procceed to some plotting to describe the data better

# We will start with a **histogram** of the **publication year** of the articles. For that we will need to extract the years column from the data

# In[ ]:


years = df['year']


# The y axis will be in log10 scale for more clarity and we will use articles from 1950 to 2020 with 70 bins.

# In[ ]:


fig = plt.figure(figsize=(10,8))
plt.xlabel('year')
plt.ylabel('#articles (logscale)')
years.hist(range=(1950, 2020), bins=70, log=True)
plt.show()


# ---

# ## Next, we are going to plot the **number of citations** for the top-50 ranked articles
# To do this we will need to import our second file, **lg-covid19-hotp-literature-graph.mtx**. Since it is in the portable matrix format **.mtx** we will use the **mmread** function from **scipy** package. This will load the matrix into **COO format**, but conversion to other sparse matrix formats is very easy(details [here](https://docs.scipy.org/doc/scipy/reference/sparse.html) )

# In[ ]:


from scipy.io import mmread
A = mmread('/kaggle/input/lgcovid19hotp/lg-covid19-hotp-literature-graph.mtx')


# Now, calculating the number of citations for each article is as easy as summing the columns!

# In[ ]:


inCitations = np.asarray( A.sum(axis=1) ).squeeze()


# Variable *inCitations* is a 103.861x1 vector containing the number of citations for each article. 
# 
# Following the same logic we could also calculate the number of referencecs of each article by simply summing the rows.
# 
# Let's sort the vector and extract the top 50 along with the corresponding table rows from the csv.

# In[ ]:


sortedIndices = np.argsort(inCitations)
sortedIndices = sortedIndices[::-1]   # Default sorting is ascending so convert it to descenting
top = df.iloc[sortedIndices[0:50]]
titles = top['title'].tolist()
topInCitations = inCitations[sortedIndices[0:50]].squeeze()


# Next we need to calculate the rank of each article by taking into account that many articles could have the same number of citations and consequently the same rank. The following code will do the job.

# In[ ]:


# Get unique elements and unique counts
u, uniqueCounts = np.unique(topInCitations, return_counts=True)
uniqueCounts = uniqueCounts[::-1]
u = u[::-1]

# 50x1 vector to hold each rank
ranks = np.zeros((len(topInCitations), 1)).squeeze()

i = 0  # Loop iterator
last = 0  # Index to last position of ranks array
rank = 1  # Current rank
while i<len(u):
    width = uniqueCounts[i]  # Number of articles with the same rank at each repetition
    ranks[last:last+width] = rank
    rank += 1
    i += 1
    last += width


# Now that we have the necessary data, let's do the plotting

# In[ ]:


# Bokeh Libraries
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource

# Output the visualization directly in the notebook
output_notebook()

# Create a figure with no toolbar
fig = figure(y_axis_label='#citations (logscale)', y_axis_type='log',
             x_axis_label='rank',
             plot_height=600, plot_width=800,
             x_range=(0, 50), y_range=(250, 1000),
             toolbar_location=None)

data={
    'x': ranks,
    'y': topInCitations,
    'title': titles
}
source = ColumnDataSource(data=data)

# Draw the coordinates as circles
fig.circle(x='x', y='y', source=source,
           color='blue', size=10, alpha=0.5)

# Format the tooltip
tooltips = [
            ('x', '@x'),
            ('y', '@y'),
            ('title', '@title')
           ]


# Add the HoverTool to the figure
fig.add_tools(HoverTool(tooltips=tooltips))

# Show plot
show(fig)


# Following the same procedure for the data from generation 1 yields yields the graph below

# In[ ]:


# Keep only generation 1
gen1 = df[df['generation'] == 1]
gen1Indices = gen1.index.tolist()
gen1Citations = inCitations[gen1Indices].squeeze()
gen1Citations.sort()
gen1Citations = gen1Citations[::-1]

gen1titles = df.iloc[gen1Indices]
gen1titles = gen1titles['title'].tolist()

# Get unique elements and unique counts
u, uniqueCounts = np.unique(gen1Citations, return_counts=True)
uniqueCounts = uniqueCounts[::-1]
u = u[::-1]

# vector to hold each rank
ranks = np.zeros((len(gen1Citations), 1)).squeeze()

i = 0  # Loop iterator
last = 0  # Index to last position of ranks array
rank = 1  # Current rank
while i<len(u):
    width = uniqueCounts[i]  # Number of articles with the same rank at each repetition
    ranks[last:last+width] = rank
    rank += 1
    i += 1
    last += width


# In[ ]:


# Output the visualization directly in the notebook
output_notebook()

# Create a figure with no toolbar
fig = figure(y_axis_type='log',
             y_axis_label='#citations (log scale)', x_axis_label='rank',
             plot_height=800, plot_width=1000,
             x_range=(0, 150), y_range=(0, 1000),
             toolbar_location=None)

data={
    'x': ranks,
    'y': gen1Citations,
    'title': gen1titles
}
source = ColumnDataSource(data=data)

# Draw the coordinates as circles
fig.circle(x='x', y='y', source=source,
           color='blue', size=8, alpha=0.5)

custom_hover = HoverTool()

custom_hover.tooltips = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>

    <b>X: </b> @x <br>
    <b>Y: </b> @y <br>
    <b>Title: </b> @title 
"""

# Add the HoverTool to the figure
fig.add_tools(custom_hover)

# Show plot
show(fig)


# In[ ]:




