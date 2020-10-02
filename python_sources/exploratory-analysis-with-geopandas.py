#!/usr/bin/env python
# coding: utf-8

# # Exploratory analysis with Geopandas

# To use this notebook make sure to fork it off and run the code yourself.
# 
# The data for this project can be found at [NebraskaMap.gov](https://nebraskamap.gov/dataset/block-block-group-tract-tiger-2010). 
# 
# This notebook will not work with all browsers. While making it I had considerable trouble using Chrome, it is probably best to switch to Firefox or Edge. Additionally browser extentions may interfer with things, so you may need to disable uBlock or something.
# 
# As with almost anything in Python, we'll need to import some libraries.
# 

# In[ ]:


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp
import pyproj
import folium
from ipywidgets import widgets
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, Layout


# First we'll need to get our data, which is super easy with Geopandas. In one line of code we can import a Shapefile, GeoJSON and several others. For more information please vist the [Geopandas](http://geopandas.org) main site. 

# In[ ]:


Lincoln_blk_groups = gp.GeoDataFrame.from_file("../input/lincoln-block-groups/Lincoln_block_groups.geojson")
Lincoln_blk_groups.head()


# It's important to know what your fields mean, so we'll need to upload some metadata I created.

# In[ ]:


labels=pd.read_csv("../input/metadata/defition.csv", sep=',')
print((labels.iloc[:,1]).head())


# Next we're going to do some linear regression to see if we can find any interesting patterns. I've left choosing the fields totally up to you, but **be warned** not everything will run or make sense!
# 
# And as mentioned above, this may not work with Chrome. If you see the dropdown boxes being covered up by the plot, go ahead and try switching browsers. 

# In[ ]:


from sklearn.linear_model import LinearRegression
from ipywidgets import Button, Layout
b = Button(description='(50% width, 80px height) button',
           layout=Layout(width='50%', height='80px', margin="100px"))
def f(x='Median Age-Total Population', y='Median Value (Dollars)'):
    try:
        x_abbr = labels.loc[labels['label'] == x , 'abbreviated'].iloc[0]
        y_abbr = labels.loc[labels['label'] == y , 'abbreviated'].iloc[0] 
        x_values = ((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== x_abbr]).values)
        y_values = ((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== y_abbr]).values)
        regressor = LinearRegression()
        regressor.fit(x_values, y_values)
        fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
        plt.scatter(x_values, y_values, color = 'red')
        plt.plot(x_values, regressor.predict(x_values), color = 'blue')
        plt.title(x + " versus " + y +" in Lincoln, NE", fontsize=25 )
        plt.xlabel(x , fontsize=15)
        plt.ylabel( y , fontsize=15)
        plt.show()
    except:  
        print("Opps, something went wrong! Perhaps you should try diffent fields")
        plt.close()
interact(f,x=labels.iloc[:,1], y=labels.iloc[:,1])


# And lastly we'll create a choropleth map. And as above, choose your fields wisely. 

# In[ ]:


def f(x='Total Population', y='Total Population-Male'):
    try: 
        f, ax = plt.subplots(1, figsize=(15, 10))
        ax.set_title(x +" divided by " + y + ' in Lincoln, NE')
        denom = labels.loc[labels['label'] == x , 'abbreviated'].iloc[0]
        numer = labels.loc[labels['label'] == y , 'abbreviated'].iloc[0] 
        Lincoln_blk_groups[denom+" over "+numer] = ((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== denom]).values)/((Lincoln_blk_groups.iloc[:,Lincoln_blk_groups.columns== numer ]).values)
        Lincoln_blk_groups.plot(denom+" over "+numer, scheme='fisher_jenks', k=5, cmap=plt.cm.Blues, legend=True, ax=ax)
        ax.set_axis_off()
        plt.axis('equal');
        plt.show()
        del Lincoln_blk_groups[denom+" over "+numer] 
    except: 
        print("Opps, something went wrong! Perhaps you should try diffent fields")
        plt.close()
interact(f,x=labels.iloc[:,1], y=labels.iloc[:,1])


# In[ ]:




