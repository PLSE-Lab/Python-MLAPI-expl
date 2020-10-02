#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
output_notebook()

from bokeh.io import output_file,show,output_notebook,save
from bokeh.plotting import figure


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/icu-patients/ICU.csv")
df


# In[ ]:


fig = figure(x_axis_label = "Age",y_axis_label = "SysBP")
fig.circle(df["Age"],df["SysBP"])
output_notebook()
show(fig)


# In[ ]:


fig.x(df["Age"],df["SysBP"],line_color="red",size=10)
show(fig)


# In[ ]:


fig.line(df["Age"],df["SysBP"],line_color="red")
show(fig)

