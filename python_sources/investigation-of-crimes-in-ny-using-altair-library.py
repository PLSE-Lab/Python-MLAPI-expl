#!/usr/bin/env python
# coding: utf-8

# **Information about Dataset:**
# 
# **Context:**
# Dataset hosted by the State of New York o Kaggle.com. The dataset is updated anually (most recent udate in 2017). It contains information regarding the total number of crimes that were reported in the State of New York since 1970. The data was obtained from Law enforcement agenceis Criminal history database. More informatin can be foud in the "Overview section"
# 
# **Content:**
# Dataset contains 3055 obeservations and 13 variables as follows"
# 1. County : Name of the county where crime was reported/recorded.
# 2. Year: Year crime occured.
# 3. Total: Total number of adult felony and misdemeanor arrests.
# 4-8 : Felonies : Drug, Violent, DWI, Other
# 9-13: Misdemeanor: Total, Drug, DWI,Propert, Other
# 
# **Importing and reading Dataset**

# In[ ]:


import numpy as np
import pandas as pd
import altair as alt
alt.renderers.enable('notebook')
df=pd.read_csv("../input/adult-arrests-by-county-beginning-1970.csv")
df.head(5) # Reading first few observations


# **Setting up Altair Library for Visualization.** [Source](https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey)

# In[ ]:


# Setup
import csv
import json
import re
import numpy as np
import pandas as pd
import altair as alt

from collections import Counter, OrderedDict
from IPython.display import HTML
from  altair.vega import v3

# The below is great for working but if you publish it, no charts show up.
# The workaround in the next cell deals with this.
#alt.renderers.enable('notebook')

HTML("This code block contains import statements and setup.")


# In[ ]:


import altair as alt
from  altair.vega import v3
import json

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
    "This code block sets up embedded rendering in HTML output and<br/>",
    "provides the function `render(chart, id='vega-chart')` for use below."
)))


# **Finding if dataset contains missing values**

# In[ ]:


df.isna().sum()


# **Analysis:**
# Dataset doesn't contians missing values.
# 
# **What is Altair Library ?**
# 
# "Altair is a declarative statistical visualization library for Python, based on Vega and Vega-Lite". More information can be found here: http://https://altair-viz.github.io
# 
# Let's explore the datsset using Altair library....
# 
# **Plot 1:**
# Let's plot total number of Adult felony in each county in Y-axis and Year on X-axis

# In[ ]:


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

chart=alt.Chart(df).mark_line().encode(
alt.Y('Total', axis=alt.Axis(title='Total number of Adult Felony'))
,x='Year:N'
,color=alt.Color('County', legend=None),
tooltip=['County','Total']).properties(width=650)


# In[ ]:


render(chart, id='vega-chart')


# Great! Now we have plot Total number of Adult felony on Y-axis and Year on X-axis. But how can we see name of the county ?
# 
# **Answer:** Move/Hover mouse on the line and see the magic of Altair library. In the above code, I have used "tooltip" function available in Altair library.
# 
# **Data Analysis:**
# From above plot, one can see that New York, Kings, Bronx and Queens recorded highest number of Adult felonies especially between 1980's- 1990's. Let's see if we can find similar trend with other variables such as Drug Felonies and Misdemeanor's .
# 
# **Plot 2:**
# Let's zoom in on specific year on the plot using " brush" function of Altair.

# In[ ]:


brush = alt.selection(type='interval', encodings=['x'])

upper = alt.Chart().mark_line().encode(
    alt.X('Year:N', scale={'domain': brush.ref()}),
    y='Drug Felony:Q'
,color=alt.Color('County', legend=None)
,tooltip=['County','Drug Felony']
).properties(
    width=650,
    height=300
)
lower = upper.properties(
    height=150
).add_selection(
    brush
)
render(alt.vconcat(upper, lower, data=df))


# Let's try to zoom in specific year. To zoom in simply move/hover mouse over bottom chart left click and select specific year and One can see that the top  chart get's zoom in on the year. Magical righ !!! :)
# 
# **Data Analysis:**
# New York, Kings, Queens and Bronx recorded highest Drug felonies in 1980's-1990's.  Same trend continues as we saw with Total adult felonies. Let's explore one more variable i.e total misdemeanor.
# 
# **Plot 3:** 
# Let's highlight lines and try to see the trend.

# In[ ]:


highlight = alt.selection(type='single', on='mouseover',
                          fields=['County'], nearest=True)
base = alt.Chart(df).encode(
    x='Year:N',
    y='Drug Misd:Q',
  color=alt.Color('County:N',legend=None),
 tooltip=['County','Drug Misd'])
points = base.mark_circle().encode(
    opacity=alt.value(0)
).add_selection(
    highlight
).properties(
    width=650
)
lines = base.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3))
)

render(points + lines)


# To see line highlighted simply move/hover mouse on specific line and analyze the result.
# 
# **Data Analysis:***
# New York, Kings, Queens and Bronx also recorded highest misdemeanor. Though drug misdemeanor was lower till the year 1980.It spiked up rapidly in New York till the year 1990. Then there was decline in drug misdemeanor for apprximately 3 year and for some reason it spiked up rapidly after the year 1993. After the year 2004, there appears to be decline in the crime rate.
# 
# 
# **Overall  Summary:**
# 1. New York, Kings, Queens and Bronx recored highest crimes between 1980-1990's, then crime rates start to decline.
# 2. We saw the power of Altair library.
# 
# ***What's Next ?***
# I will try to explore more power of Altair  and will also try to answer the question***Why was crime rate so high in New York between 1980's-1990's ?*** [ To my understanding some reasons can be high unemployment, high immigration of people in NY during that period ]- I will try to find the dataset related to that. In the mean time, if you find any dataset that can be helpful in my analysis, please feel free to share in the comment section below. 
# 
# **Note:**
# *This Kernel is influenced by the example mentioned on Altair's official page. It is possible that some of the code used in this Kernel might appear on offical page. *
# 
# Part 2 can be found [here](https://www.kaggle.com/apnanaam08/crimes-in-ny-using-altair-library-p2).
# 
# Love . Share . Care . Peace
# 
# **References:**
# Altair Developers (n.d.). Altair: Declarative Visualization in Python. Retrieved August 15, 2018, from https://altair-viz.github.io/
