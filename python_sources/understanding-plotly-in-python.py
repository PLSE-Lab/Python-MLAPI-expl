#!/usr/bin/env python
# coding: utf-8

# # Understanding plotly at high level in Python
# 
# Dataset being used - People Analytics | Atrrition Data 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/MFG10YearTerminationData.csv")


# ### View of the data set

# In[ ]:


df.head()


# ### Potential uses of the data
# 1. What are the factors that directly affect attrition?
# 2. How does attrition do amongst various categorical features?
# 3. What are the various groups of employees that exist in the data? How does attrition do with respect to these groups?
# 
# ### Exploratory Data Analysis and Visualization
# 
# Before that. Describe the data.
# 

# In[ ]:


print("Basic statistics of numerical features in the data are \n", df.describe())
print("\n Dimension of the data are", df.shape)


# * From plotly's website
# 
# # How to enable plotly to plot in ipython notebook?
# * Following two commands can be used to plot the plotly graphs in ipython notebook. Without these commands, we won't be able to
# * **plotly.offline** package has functions which enable us to save the plotply plots offline.
# 
# 
# 
# 
# 

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# # What is plotly?
# 
# ### Plotly at its core is a data visualization toolbox. Under every plotly graph is a JSON object, which is a dictionary like data structure. Simply by changing the values of some keywords in this object, we can get vastly different and ever more detailed plots.
# 
# * Plotly's graph_objs module contains various object definitions for different plots. This is similar to various geoms in ggplot2.
# * Ex: plotly.graph_objs.Scatter() is similar to geom_point()
# * All graph objects take arguements similar to the aesthetics we provide in ggplot2. Ex: x,y, etc..

# In[ ]:


# Importing the graph_objs module which contains plotting objects
import plotly.graph_objs as go
# Trace1 can be viewed like a geom_point() layer with various arguements
trace1 = go.Scatter(x=df.age, y=df.length_of_service, marker=dict(size=5,
                line=dict(width=1),
                color="yellow"
               ), 
                    mode="markers")


# * So now trace1 becomes our first layer.
# * **plotly.graph_objs.Data()** function is used to combine multiple layers. Ex: If, for a certain plot, we have to use two graph objects stored in trace1 and trace2, then we will combine them using Data function
# * **plotly.graph_objs.Layout()** function creates a layer for the layout of the plot. This layer has options similar to (but not exactly like) THEME() layer in ggplot2.
# * Finally **plotly.graph_objs.Figure()** function is used to combine all the elements together so that it can be supplied to the **iplot()** function

# In[ ]:


data1 = go.Data([trace1])
layout1=go.Layout(title="Age vs Length of service", xaxis={'title':'Age'}, yaxis={'title':'Length of Service'})
figure1=go.Figure(data=data1,layout=layout1)
iplot(figure1)


# ## Histograms
# * Our data has AGE of employee. Lets see the distribution of the ages of employees
# * We will follow a similar format by replacing **go.Histogram()** for **go.Scatter()**

# In[ ]:


trace2 = go.Histogram(x=df.age)
data2 = go.Data([trace2])
layout2=go.Layout(title="Distribution of Age", xaxis={'title':'AGE'}, yaxis={'title':'Number of employees in data'})
figure2=go.Figure(data=data2,layout=layout2)
iplot(figure2)


# ### Normalized histogram of age
# 
# * Obtained by simply adding **histnorm='probability'** in **go.Histogram()** function

# In[ ]:


trace3 = go.Histogram(x=df.age, histnorm='probability')
data3 = go.Data([trace3])
layout3=go.Layout(title="Distribution of Age", xaxis={'title':'AGE'}, yaxis={'title':'Number of employees in data'})
figure3=go.Figure(data=data3,layout=layout3)
iplot(figure3)


# ## Overlaid histogram

# In[ ]:


terminated = df[df['STATUS'] == 'TERMINATED']
active = df[df['STATUS'] == 'ACTIVE'] 
print("Shape of df for terminated records is \n", terminated.shape)


# In[ ]:


trace4 = go.Histogram(
    x=terminated.age,
    opacity=1
)
trace5 = go.Histogram(
    x=active.age,
    opacity=0.3
)

data45 = go.Data([trace4, trace5])
layout45 = go.Layout(barmode='overlay')
figure45 = go.Figure(data=data45, layout=layout45)

iplot(figure45, filename='overlaid histogram')


# ## Stacked histogram

# In[ ]:


trace4 = go.Histogram(
    x=terminated.age,
    opacity=0.8
)
trace5 = go.Histogram(
    x=active.age,
    opacity=0.8
)

data45 = go.Data([trace4, trace5])
layout45 = go.Layout(barmode='stack')
figure45 = go.Figure(data=data45, layout=layout45)

iplot(figure45, filename='stacked histogram')


# ## Bar charts
# 
# * plotly's bar chart layer needs us to supply the categories and frequencies as lists i.e. x = ['Male', 'Female'], y = [3444,2412]
# * It does not automatically do the group by counts by itself.
# * So I defined a function **bar_chart_counts** which
# 1. Takes dataframe and categorical feature name as arguements
# 2. Throws out a bar chart of counts and various categories

# In[ ]:


def create_counts_df(df, categorical_feature):
    new_df = df.groupby([categorical_feature]).count()['store_name'].reset_index()
    new_df.columns = [categorical_feature, 'Total records']
    return new_df

def bar_chart_counts(df, categorical_feature):
    
    df_new = create_counts_df(df, categorical_feature)
    
    data = [go.Bar(
            x=df_new.iloc[:,0],
            y=df_new.iloc[:,1]
    )]

    iplot(data, filename='basic-bar')
    
bar_chart_counts(df, "gender_full")


# * Applying the same function to several other categorical features in the data

# In[ ]:


bar_chart_counts(df, "city_name")


# * Supplying different DF to the function.
# * Supplying **terminated** and **active**  dataframes which contains records of employees who were terminated/active.

# In[ ]:


bar_chart_counts(terminated, "city_name")


# In[ ]:


bar_chart_counts(active, "city_name")


# * Since this is a SYNTHETICALLY generated data, there seems to be no city wise variation in ACTIVE/TERMINATED employees
# 
# # Grouped bar charts
# 
# * For grouped bar charts, the idea is to create two BAR objects and then place them side by side.
# * Again, the group by and counts (or any other metric) of the underlying data is not going to be automatically done by plotly.
# * So first using the function **create_counts_df()** we created above, we will create the group bys for 1 categorical feature ("CITY_NAME") for two separate groups of the data ("ACTIVE/TERMINATED") employees which are stored in **active** and **terminated** data frames.

# In[ ]:


active_gb_gender = create_counts_df(active, "gender_full")
terminated_gb_gender = create_counts_df(terminated, "gender_full")

trace1 = go.Bar(
    x= active_gb_gender.iloc[:,0],
    y= active_gb_gender.iloc[:,1],
    name='Active'
)
trace2 = go.Bar(
    x=terminated_gb_gender.iloc[:,0],
    y=terminated_gb_gender.iloc[:,1],
    name='Terminated'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# ## Stacked bar with just a change of 1 arguement

# In[ ]:


data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


#   
