#!/usr/bin/env python
# coding: utf-8

# <img src="https://s3.trustandjustice.org/images/partners/cpe.png">
# 
# ## Hunting for Insights / Exploration - Policing Equity
# The Center for Policing Equity (CPE) is research scientists, race and equity experts, data virtuosos, and community trainers working together to build more fair and just systems. Police departments across the United States have joined our National Justice Database, the first and largest collection of standardized police behavioral data.
# 
# ## Data Science for Good : Problem Statement
# How do you measure justice? And how do you solve the problem of racism in policing? We look for factors that drive racial disparities in policing by analyzing census and police department deployment data. The ultimate goal is to inform police agencies where they can make improvements by identifying deployment areas where racial disparities exist and are not explainable by crime rates and poverty levels. The biggest challenge is automating the combination of police data, census-level data, and other socioeconomic factors.

# In[ ]:


# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import os
print(os.listdir("../input"))


# In[ ]:


# Displaying available data files and directories
def dirlist(parent,child,count=0):
    if os.path.isdir(parent+child+"/"):
        p_list = os.listdir(parent+child+"/")
        p_text = parent+child+"/"
        if len(p_list)>0:
                count = count + 1
                for val,child in enumerate(p_list):
                    print("{}-{}".format(" "*(count*4),child))
                    dirlist(p_text,child,count)
        else:
            pass
    
    
for child in os.listdir("../input/cpe-data/"):
    print("{}".format(child))
    dirlist("../input/cpe-data/",child)
        


# In[ ]:


#Let's look at the available department files
[f for f in os.listdir("../input/cpe-data/") if f.startswith("Dept")]


# In[ ]:


#Let's start with Dept_23-00089
os.listdir("../input/cpe-data/Dept_23-00089/23-00089_ACS_data/")


# In[ ]:


#First, we will look at race,sex and age data
one_df = pd.read_csv("../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv")
one_df.info()
one_df.describe()


# ### Distribution of population according to gender

# In[ ]:


data = [go.Histogram(x=one_df["HC01_VC04"],
                     name='Male',
                     opacity=0.7,
                     marker=dict(
                        color='rgb(158,202,225)',
                        line=dict(
                            color='rgb(8,48,107)',
                            width=1.5,
                        )
                    )),
        go.Histogram(x=one_df["HC01_VC05"],
                     name='Female',
                     opacity=0.7,
                     marker=dict(
                        color='rgb(255,254,115)',
                        line=dict(
                            color='rgb(255,233,93)',
                            width=1.5,
                        )
                    )),
        go.Histogram(x=one_df["HC01_VC03"],
                     name='Total',
                     opacity=0.7,
                     marker=dict(
                        color='rgb(0,255,174)',
                        line=dict(
                            color='rgb(51,158,53)',
                            width=1.5,
                        )
                    ))]

updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'Male',
                 method = 'update',
                 args = [{'visible': [True, False, False]},
                         {'title': 'Male'}]),
            dict(label = 'Female',
                 method = 'update',
                 args = [{'visible': [False, True, False]},
                         {'title': 'Female'}]),
            dict(label = 'Total',
                 method = 'update',
                 args = [{'visible': [False,False,True]},
                         {'title': 'Total'}]),
            dict(label = 'Male & Female',
                 method = 'update',
                 args = [{'visible': [True, True, False]},
                         {'title': 'Male & Female'}]),
             dict(label = 'Male & Total',
                 method = 'update',
                 args = [{'visible': [True, False, True]},
                         {'title': 'Male & Total'}]),
             dict(label = 'Female & Total',
                 method = 'update',
                 args = [{'visible': [False, True, True]},
                         {'title': 'Female & Total'}]),
             dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, True]},
                         {'title': 'All'}])
        ]),
    )
])
layout = go.Layout(barmode='overlay')
layout['title'] = 'Population Distribution'
layout['showlegend'] = True
layout['updatemenus'] = updatemenus

fig = dict(data=data, layout=layout)
iplot(fig, filename='update_button')


# **This will be a work-in-progress kernel till I don't remove this last statement. It is in a very early stage of development. So, please upvote, comment and stay tuned for future versions where I promise to bring out the best out of this data.**
