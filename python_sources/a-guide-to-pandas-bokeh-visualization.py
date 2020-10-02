#!/usr/bin/env python
# coding: utf-8

# # Introduction :

# ### **In this kernel notebook I will be focusing on initially covering the new Pandas-Bokeh Data visualisation on Top 50 Spotify Songs Dataset.**

# ![data-visualization-using-python-libraries-like-bokeh-matplotlib-seaborn.jpg](attachment:data-visualization-using-python-libraries-like-bokeh-matplotlib-seaborn.jpg)

# ***Pandas-Bokeh* provides a Bokeh plotting backend for Pandas, GeoPandas and Pyspark DataFrames, similar to the already existing Visualization feature of Pandas. Importing the library adds a complementary plotting method plot_bokeh() on DataFrames and Series.**
# 
# **With *Pandas-Bokeh*, creating stunning, interactive, HTML-based visualization is as easy as calling:**
# 
# **df.plot_bokeh()**
# 
# ***Pandas-Bokeh* also provides native support as a Pandas Plotting backend for Pandas >= 0.25. When Pandas-Bokeh is installed, switchting the default Pandas plotting backend to Bokeh can be done via:**
# 
# **pd.set_option('plotting.backend', 'pandas_bokeh')**

# ### **Now its time to first install Pandas_Bokeh using PIP command.**

# In[ ]:


get_ipython().system('pip install pandas-bokeh')


# ### **Import Libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_bokeh
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend','pandas_bokeh')
# create a bokeh table using data frame
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnarDataSource


# ### **Import Spotify Songs Dataset**

# In[ ]:


df = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')


# Lineplot
# This simple lineplot in Pandas-Bokeh already contains various interactive elements:
# 
# 1. A pannable and zoomable (zoom in plotarea and zoom on axis) plot
# 2. By clicking on the legend elements, one can hide and show the individual lines
# 3. A Hovertool for the plotted lines
# 

# In[ ]:


df.head(20)


# ## **Line Plot**

# In[ ]:


df.plot_bokeh(kind="line",title='Length and Beats per minute Comparasion',
              figsize=(1000,800),# Figure Size 
              xlabel="Beats.Per.Minute", # X -axis Label
              ylabel="Length.") # Y-axis label


# ### **In the above data visualisation which is completely intereactive. You can click on any index regions and check the data .Is it an interesting data visualisation ???????**

# ## **Bar Plot**

# In[ ]:


df.plot_bokeh(kind='bar',title='Energy Vs Popularity',
              figsize=(1000,800),
              xlabel="Popularity",
              ylabel="Energy")


# ## **Point Plot**

# In[ ]:


df.plot_bokeh(kind="point",title="Dancebility Vs Liveness", figsize=(1000,800),
              xlabel="Liveness",
              ylabel="Dancebility")


# ## **Histogram**

# In[ ]:


df.plot_bokeh(kind="hist",title="Dancebility Vs Liveness", figsize=(1000,800),
              xlabel="Liveness",
              ylabel="Dancebility")


# ## **Line Plot Using Range Tool**

# In[ ]:


df.plot_bokeh(kind="line",title='Length and Beats per minute Comparasion',
              figsize=(1000,800),
              xlabel="Beats.Per.Minute",
              ylabel="Length.",rangetool=True)


# ## **Point Plot**

# In[ ]:


df.plot_bokeh.point(x=df.Energy,xticks=range(0,1), size=5,
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title = "Point Plot - Spotify Songs",fontsize_title=20,
    marker="x",figsize =(1000,800))


# ## **Step Plot**

# In[ ]:


df.plot_bokeh.step(
    x=df.Energy,
    xticks=range(-1, 1),
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title="Step Plot - Spotify Songs",
    figsize=(1000,800),
    fontsize_title=20,
    fontsize_label=20,
    fontsize_ticks=20,
    fontsize_legend=8,
    )


# ## If you like this kernel Greatly Appreciate to **UPVOTE** .Thank you
