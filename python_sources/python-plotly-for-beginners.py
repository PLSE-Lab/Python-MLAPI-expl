#!/usr/bin/env python
# coding: utf-8

# ## Python - Plotly [Interactive Plotting] for beginners
# 
# 
# Priyaranjan Mohanty

# ### Objective of this Kernel :
# 
# This kernel endeavors to explore & explain the Interactive Visualization tool called Plotly.
# We will be learning by working with examples .
# 
# #### Target Audience :
# This for any one who is new to Plotly Package in Python.

# ### Plot.ly
# 
# Plot.ly is a JSON-based plot tool for interactive visualization. Every graph can be defined by a JSON object with two keys named data and layout. It also offers tools for interaction, and they are much more straightforward to use and customize. Most plots come with hover labels and legends for groups, both interactive by default, which is awesome.

# Lets delve into 'Learning by Building / Creating '

# Before , we can use Plotly , we need to install Plotly ......

# In[ ]:


# Install plotly ...as Plotly is not installed as part of Python base package

get_ipython().system('pip install plotly')


# Now that Plotly has been installed ( or confirmed that it is already installed ) ...  lets Import the Plotly Package and check the version of Plotly being used .

# In[ ]:


# (*) Import plotly package
import plotly

# Check plotly package version
plotly.__version__ 


# ### Bar Plot
# 

# In[ ]:


import plotly.graph_objects as go

Bar_Plot = go.Figure(
                    data=[go.Bar(y=[2, 1, 3])],
                    layout_title_text="Bar Plot using Plotly"
                    )

Bar_Plot.show()


# In[ ]:


import plotly.graph_objects as go

Scatter_Plot = go.Figure(
                    data=[go.Scatter(x=[0, 1, 2] , y=[2, 1, 3])],
                    layout_title_text="Scatter Plot using Plotly"
                    )

Scatter_Plot.show()

