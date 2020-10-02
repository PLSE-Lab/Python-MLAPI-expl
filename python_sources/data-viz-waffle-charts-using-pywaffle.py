#!/usr/bin/env python
# coding: utf-8

# Waffle chart is an interesting plot mainly used to display progress towards the goal. Github uses it to display daily efforts by its users. It has cells structure and can be used to display proportions as well ( as in the case of this visualization). Kindly upvote it if you like and comment your feedback.<br><br>
# Pywaffle documentation link: https://pywaffle.readthedocs.io/en/latest/index.html

# In[ ]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt


# To use pywaffle, we first need to install it using pip. <br>
# Note: If you want to fork this notebook, you should go to left pane and in settings option enable Internet.

# In[ ]:


pip install pywaffle


# In[ ]:


#After installing pywaffle, import it
from pywaffle import Waffle


# In[ ]:


#reading data
df= pd.read_csv("../input/indian-women-in-defense/WomenInDefense.csv")
df


# In[ ]:


# Rename column name 
df.rename({"Army Medical Corps, Dental Corps & Military Nursing Service (Common for three forces)":"Others"}, axis=1, inplace=True)
df


# In[ ]:


# Add a column with total sum in all categories in respective years
df['Total']= df[['Army','Navy','Air Force','Others']].sum(axis=1)
df


# In[ ]:


# To check if column names are string type or not, and convert them to string type if not already
# This code is specifically if df was transposed to make year as column
print(all(isinstance(columns, str) for columns in df.columns))
df.columns= list(map(str, df.columns))


# In[ ]:


# Plotting single waffle chart, for All Indian women commissioned in Navy over the years
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5,
    values=df.Navy
)


# In[ ]:


# Tweaking the appearance of chart: Adding title, labels, figsize
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5,             #change number of rows here
    values=df.Navy,
    labels= list(df.Year),
    
    title={
        'label': 'Number of Indian women in Navy over the years',
        'loc': 'left'
    },
    figsize= (10,8)
)


# In[ ]:


# Tweaking the appearance of chart: More options
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5,             
    values=df.Navy,
    labels= list(df.Year),
    #legend={'loc': 'upper left'},
    legend={
        
        'loc': 'lower left',
        'bbox_to_anchor': (0, -0.4), #check matplotlib legend doc for axis position info
        'ncol': len(df)
    },
    title={
        'label': 'Number of Indian women in Navy over the years',
        'loc': 'left'
    },
    figsize=(10,8)
)


# In[ ]:


df['Total']/20


# In[ ]:


# Using subplots to create multiple charts
fig2 = plt.figure(
    FigureClass=Waffle,
    plots={
        '411': {                              #refer matplotlib subplot grids, '411' means 4 x 1 grid, first subplot
            'values': df['Army'],
            'labels': list(df.Year),
            'legend': {
                'loc': 'upper left',
                'bbox_to_anchor': (1.05, 1)
            },
            'title': {
                'label': 'Army',
                'loc': 'left'
            }
        },
        '412': {
            'values': df['Navy'],
            'labels': list(df.Year),
            'legend': {
                'loc': 'upper left',
                'bbox_to_anchor': (1.05, 1)
            },
            'title': {
                'label': 'Navy',
                'loc': 'left'
            }
        },
        '413': {
            'values': df['Air Force'],
            'labels': list(df.Year),
            'legend': {
                'loc': 'upper left',
                'bbox_to_anchor': (1.05, 1)
            },
            'title': {
                'label': 'Air force',
                'loc': 'left'
            }
        },
        '414': {
            'values': df['Total']/10,   #scaling purpose to fit 
            'labels': list(df.Year),
            'legend': {
                'loc': 'upper left',
                'bbox_to_anchor': (1.05, 1)
            },
            'title': {
                'label': 'Total',
                'loc': 'left'
            }
        },
        
    },
    rows=5,
    figsize=(11, 9)
)

