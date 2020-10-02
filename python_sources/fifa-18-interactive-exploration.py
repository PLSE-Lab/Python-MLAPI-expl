#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from IPython.display import Image, HTML
import re
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Summary
# This module is focused on exploratory data analysis on FIFA 2018 dataset. The CompleteDataset is imported in a pandas dataframe.
# 
# We will look at some statistics for each country in an interactive map using Plotly library.
# 
# 

# ## **1. Read Data**

# In[ ]:


data = pd.read_csv('../input/CompleteDataset.csv', low_memory=False)


# In[ ]:


data.head()


# The first column will be remove, as it is just an index column.

# In[ ]:


data = data.drop(data.columns[0], axis = 1)
data.info()


# **1.1 Check for Missing Values**
# 
# Check for missing values by calling isnull() method on the dataframe followed by any() method.

# In[ ]:


data.isnull().any()


# Looks like there are some columns with missing values. Let's find them.

# In[ ]:


data.isnull().any()[data.isnull().any()==True]


# **1.3 Check Data Types**
# 
# It appears that the Wage and Value columns are of type Object. Let's extract the numeric values and store them in new columns. Wage is in Thousand Euros (TEUR) and Value is in Millin Euros (MEUR).

# In[ ]:


# Extract numeric vales of the wage and value
data['Wage(TEUR)'] = data['Wage'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64')
data['Value(MEUR)'] = data['Value'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64')


# **1.4 Re-index the Columns**
# 
# Let's move some of the personal attributes to the front of the dataframe in order too facilitate slicing and selecting different sections of the data. To do that, a list of column names in the preferred order is needed.

# In[ ]:


reordered_cols = []
personal_cols = []
personal_cols = ['ID', 'Name', 'Photo', 'Club', 'Club Logo', 'Preferred Positions', 'Wage', 'Value',
                 'Nationality', 'Flag']
reordered_cols = personal_cols + [col for col in data if (col not in personal_cols)]


# In[ ]:


data = data[reordered_cols]
data.head()


# ## **2. Explore Various Attributes Based on Nationality**
# 
# **2.1 Slice Dataframe**
# 
# Let's slice the dataframe to by removing the first seven columns.

# In[ ]:


country_data = data.iloc[:, 8:]


# **2.2 Convert to Numeric**
# 
# Some of the numeric columns are stored as Object type, as seen earlier. Those need to be converted to numeric. We use the apply() method of the dataframe on the selected columns and pass 'coerce' for the errors argument to replace inappropriate values with NaN.

# In[ ]:


country_data.iloc[:, 2:] = country_data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
country_data.info()


# **2.3 Aggregation**
# 
# We first group the dataframe by the column 'Nationality'. Then pass an aggregation dictionary to the agg() method to specify the aggregation functions on each feature.
# 
#    * Since the Flag column is non-numeric, we will use 'min' (or 'max', but not 'mean')
#     

# In[ ]:


agg_dict = {}
agg_dict = {'Flag': ['min']}
for col in country_data.columns[2:]:
    agg_dict[col] = ['mean', 'max', 'min', 'size']
group_by_country = country_data.groupby(['Nationality'])
country_stats = group_by_country.agg(agg_dict)
country_stats.head()


# **2.4 Displaying Flag Images**
# 
# First, the Flag values need to be replaced by an HTML image object. Then the to_html() method of the data frame is used to generated the HTML code for the dataframe. Finally the HTML() method is used to display the table in Jupyter notebook.

# In[ ]:


country_stats[('Flag', 'min')] = '<img src="' + country_stats[('Flag', 'min')] + '">'


# In[ ]:


HTML(country_stats.head(10).to_html(escape=False))


# **2.4 Visualize Data in an Interactive Map**
# 
# A function is written to create a graph object for each (feature, basic statistic) pair.

# In[ ]:


def create_trace(feature, basic_stat, is_visible):
    trace = go.Choropleth(
        locations= country_stats.index,
        locationmode= 'country names',
        z= country_stats[(feature, basic_stat)],
        text= country_stats[(feature, 'size')],
        visible = is_visible,
        colorscale = [
            [0.0,"rgb(25, 100, 255)"],[0.2,"rgb(25, 175, 255)"],[0.4,"rgb(25, 255, 255)"],
            [0.6,"rgb(25, 255, 175)"],[0.8,"rgb(25, 255, 100)"],[1.0,"rgb(25, 255, 25)"]
            ],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            #autotick = True,
            tickprefix = '',
            outlinecolor = "rgba(68, 68, 68, 0)",
            #dtick = 2,
            title = feature + '<br>' + basic_stat)
    )
    return trace


# We will then plot the data. Let's add a drop dow to be able select a pair of feature and statistic for visualization.

# In[ ]:


trace_data = []
buttons= []
is_visible = True
features = country_data.columns[2:]
stats = ['mean', 'max', 'min']
n = len(features) * len(stats)
nth_feature = 0
nth_stat = 0

for stat in stats:
    for feature in features:
        trace_data.append(create_trace(feature, stat, is_visible))
        is_visible = False
        
        pre_false = [False]*nth_feature
        post_false = [False]*(n-nth_feature-1)
        button = dict(
            label= feature + ' - ' + stat,
            method= 'update',
            args=[
                {'visible': pre_false + [True] + post_false},
                {'title': 'FIFA 2018 Statistics (' + feature + ' - ' + stat + ')'}
            ]
        )
        buttons.append(button)
        nth_feature += 1
        
updatemenus = list([
    dict(
        #type="buttons",
        direction='down',
        active=-1,
        xanchor='left',
        x=0,
        yanchor = 'top',
        y=1.065,
        showactive = True,
        buttons=buttons
    )
])

annotations = list([
    dict(text='(Feature - Stat):',
         x=0,
         y=1.12, 
         yref='paper',
         align='left',
         showarrow=False)
])

layout = dict(
    title = 'FIFA 2018 Statistics (' + features[0] + ' - ' + stats[0] + ')' ,
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    ),
    showlegend=False,
    updatemenus=updatemenus,
    annotations=annotations
)

fig = dict( data=trace_data, layout=layout )
py.iplot( fig, validate=False, filename='fifa-2018-world-map' )

