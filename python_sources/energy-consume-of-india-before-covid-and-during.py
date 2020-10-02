#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyforest')


# In[ ]:





# In[ ]:


from pyforest import *


# In[ ]:


df= pd.read_csv("../input/state-wise-power-consumption-in-india/long_data_.csv")
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df["Dates"]= pd.to_datetime(df.Dates)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.Dates.dt.month_name()


# In[ ]:


g=df.groupby(by="States")


# In[ ]:


g


# In[ ]:


for state, state_df in g:
  print(state)
  print(state_df)


# In[ ]:


punjab=g.get_group("Punjab")


# In[ ]:


punjab.head()


# In[ ]:


punjab.tail()


# #A line plot to depict the power consumption of Punjab

# In[ ]:


ax=sns.relplot(x="Dates",y="Usage", data=punjab, kind="line", markers=True)
for axes in ax.axes.flat:
  axes.set_xticklabels(axes.get_xticklabels(), rotation=65, horizontalalignment='right')


# #Scatter plot

# In[ ]:


from bokeh.models import HoverTool
import bokeh

from bokeh.io import output_notebook, reset_output, show

from bokeh.plotting import figure

import numpy as np
import pandas as pd

output_notebook()

from bokeh.models import ColumnDataSource


# In[ ]:


line_plot= figure(plot_width=700, plot_height=300, title="line plot", x_axis_label="Dates", y_axis_label= "Usage", toolbar_location="below")

line_plot.line(df.Dates,df.Usage, legend_label="line")

line_plot.add_tools(HoverTool())

show(line_plot)


# Above we can see the power consumption of all states for the 2019 and 2020. Hovering on the scatter points will give the value of the power for a particular date

# In[ ]:


punjab=punjab.reset_index()


# In[ ]:


punjab.head()


# #Bar plot of all state power usage

# In[ ]:


ax=sns.barplot(x="States", y="Usage", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right')


# Above we can see the total power usage of the states in Million Units.
# 
# Also it can be seen that Maharashtra is the state with the highest power consumption both for 2019 and 2020

# In[ ]:





# 

# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df2= pd.read_csv("../input/state-wise-power-consumption-in-india/dataset_tk.csv")

df2.head()


# In[ ]:


df2.columns


# In[ ]:


df2["Dates"]= df2["Unnamed: 0"]


# In[ ]:


df2.head()


# In[ ]:


df2= df2.drop("Unnamed: 0", axis=1)


# In[ ]:


df2.tail()


# In[ ]:


df2= df2.set_index("Dates")


# In[ ]:


df2.head()


# In[ ]:





# #Dividing the regions into NR,SR,ER,WR,NER

# In[ ]:


df2['NR'] = df2['Punjab']+ df2['Haryana']+ df2['Rajasthan']+ df2['Delhi']+df2['UP']+df2['Uttarakhand']+df2['HP']+df2['J&K']+df2['Chandigarh']
df2['WR'] = df2['Chhattisgarh']+df2['Gujarat']+df2['MP']+df2['Maharashtra']+df2['Goa']+df2['DNH']
df2['SR'] = df2['Andhra Pradesh']+df2['Telangana']+df2['Karnataka']+df2['Kerala']+df2['Tamil Nadu']+df2['Pondy']
df2['ER'] = df2['Bihar']+df2['Jharkhand']+ df2['Odisha']+df2['West Bengal']+df2['Sikkim']
df2['NER'] =df2['Arunachal Pradesh']+df2['Assam']+df2['Manipur']+df2['Meghalaya']+df2['Mizoram']+df2['Nagaland']+df2['Tripura']


# In[ ]:


df2.head()


# In[ ]:


df2["NR"].values


# In[ ]:





# #Another demonstration of Punjab power statistics

# In[ ]:


import plotly.graph_objects as go
fig = go.Figure( go.Scatter(x=df2.index, y=df2["Punjab"]))
fig.show()


# In[ ]:


df_new = pd.DataFrame({"Northern Region": df2["NR"].values,
                        "Southern Region": df2["SR"].values,
                        "Eastern Region": df2["ER"].values,
                        "Western Region": df2["WR"].values,
                        "North Eastern Region": df2["NER"].values},index=df2.index)


# In[ ]:


df_new.head()


# #Power consumption in the Northern Region for 2019 and 2020

# In[ ]:


fig2 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Northern Region"],fillcolor=None))
fig2.show()


# In[ ]:





# #Power consumption in the Southern Region

# In[ ]:


fig3 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Southern Region"],fillcolor=None))
fig3.show()


# #Power consumption in the Eastern Region

# In[ ]:


fig4 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Eastern Region"],fillcolor=None))
fig4.show()


# #Power consumption in the Western Region

# In[ ]:


fig5 = go.Figure( go.Scatter(x=df_new.index, y=df_new["Western Region"],fillcolor=None))
fig5.show()


# #Power consumption in North Eastern Region
# 

# In[ ]:


fig6 = go.Figure( go.Scatter(x=df_new.index, y=df_new["North Eastern Region"],fillcolor=None))
fig6.show()


# #Comparision of Northern and Southern Region

# In[ ]:


sns.distplot(df_new["Northern Region"], bins=10)
sns.distplot(df_new["Southern Region"], bins=10)


# #Comparision of Eastern and Western Region

# In[ ]:


sns.distplot(df_new["Eastern Region"], bins=10)
sns.distplot(df_new["Western Region"], bins=10)


# #Bar graph to depict North Eastern Region probability distribution

# In[ ]:


sns.distplot(df_new["North Eastern Region"], bins=10)


# #A pairplot to show the usage according to the latitude and longitude of the region

# In[ ]:


sns.pairplot(df)


# #Correlation matrix to show the relation between different regions

# In[ ]:


sns.heatmap(df_new.corr(),annot=True)


# #Descriptive  Statistics through Pandas Profiling

# In[ ]:


get_ipython().system('pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip')


# In[ ]:


from pandas_profiling import ProfileReport


# In[ ]:


profile = ProfileReport(df, title='Power Stats', html={'style':{'full_width':False}})


# In[ ]:


profile.to_widgets()


# In[ ]:


profile


# In[ ]:


#x=df.loc[df["Dates"]>"2020-02-01"]

