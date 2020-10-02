#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Data Scientists in 2018 - Kaggle Survey</font></center></h1>
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png"></img>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
#  - <a href='#21'>Load packages</a>  
#  - <a href='#21'>Load the data</a>  
# - <a href='#3'>Data exploration</a>   
# - <a href='#4'>Combine the features</a>   
# - <a href='#5'>Final note</a>   

# # <a id='1'>Introduction</a>  
# 
# We will analyze the dataset `2018 Kaggle ML & DS Survey Challenge` with answers provided by the respondents to the survey of Kaggle users in 2018.
# 

# # <a id='2'>Prepare the data analysis</a>   
# 
# 
# Before starting the analysis, we need to make few preparation: load the packages, load and inspect the data.
# 
# 

# # <a id='21'>Load packages</a>
# 
# We load the packages used for the analysis.

# In[ ]:


import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# # <a id='22'>Load the data</a>  
# 
# Let's see first what data files do we have in the root directory.

# In[ ]:


os.listdir("../input")


# There are three dataset files. Let's load all the files.

# In[ ]:


multiple_df = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)
free_df = pd.read_csv('../input/freeFormResponses.csv', low_memory=False)
schema_df = pd.read_csv('../input/SurveySchema.csv', low_memory=False)


# In[ ]:


print("Multiple choice response - rows: {} columns: {}".format(multiple_df.shape[0], multiple_df.shape[1]))
print("Free form response - rows: {} columns: {}".format(free_df.shape[0], free_df.shape[1]))
print("Survey schema - rows: {} columns: {}".format(schema_df.shape[0], schema_df.shape[1]))


# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# # <a id='3'>Data exploration</a>  
# 
# 
# Let's start by exploring the multiple choice response dataset.
# 
# We will also glimpse the free format response dataset.
# 
# ## Glimpse the data

# In[ ]:


multiple_df.head(3)


# In[ ]:


free_df.head(3)


# Because the first row contains a description of the column, we will read only from 2nd row the categorical values per each column.
# 
# ## Missing data
# 
# Let's represent the distribution of available data for all the columns, using a boxplot.

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[ ]:


df = missing_data(multiple_df)


# In[ ]:


def plot_percent_of_available_data(title):
    trace = go.Box(
        x = df['Percent'],
        name="Percent",
         marker=dict(
                    color='rgba(238,23,11,0.5)',
                    line=dict(
                        color='tomato',
                        width=0.9),
                ),
         orientation='h')
    data = [trace]
    layout = dict(title = 'Percent of available data  - all columns ({})'.format(title),
              xaxis = dict(title = 'Percent', showticklabels=True), 
              yaxis = dict(title = 'All columns'),
              hovermode = 'closest',
             )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='percent')


# In[ ]:


plot_percent_of_available_data('multiple_df')


# In[ ]:


plot_percent_of_available_data('free_df')


# 
# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## Columns to visualize
# 
# 
# Some of the following columns are grouped, to capture the multiple choice answers where the order of the answer gives the order of preferences. 
# Let's check which columns groups have only one item in the group (columns with multiple items in the group will be called like `Q11_Part1`, `Q11_Part2`[...]. For this we will compose filters like `Q1`, `Q2`, ..., `Q11`, `Q12` etc. and filter the columns containing these values and count the items. We separate only the group of columns with one item in the group. These will be the columns we will further represent.
# 
# 

# In[ ]:


tmp = pd.DataFrame(multiple_df.columns.values)
columns = []
for i in range(1,50):
    var = "Q{}".format(i)
    l = len(list(tmp[tmp[0].str.contains(var)][0]))
    if(l == 1):
        columns.append(var)

print("The columns with only one item in the column group are:\n",columns)


# We will make sure to include these columns in the following, besides the obvious options `Q1`, `Q2` ... `Q7`.

# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## Gender
# 
# Let's show the gender, as declared by respondents. 
# We will create here also a function to count categories for categorical data and a function to draw barplots using Plotly.

# In[ ]:


def get_categories(data, val):
    tmp = data[1::][val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


df = get_categories(multiple_df, 'Q1')


# In[ ]:


def draw_trace_bar(data, title, xlab, ylab,color='Blue'):
    trace = go.Bar(
            x = data['index'],
            y = data['Number'],
            marker=dict(color=color),
            text=data['index']
        )
    data = [trace]

    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=15,
                          tickfont=dict(
                            size=9,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')


# In[ ]:


draw_trace_bar(df, 'Number of people', 'Gender', 'Number of people' )


# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# 
# ## Age group
# 
# Let's explore the age groups.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q2'), "Number of people in each age range", "Age range", "Number of people", "Green")


# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# 
# ## Country
# 
# Let's plot the number of responses per country..

# In[ ]:


df = get_categories(multiple_df, 'Q3')
df.head()


# In[ ]:


trace = go.Choropleth(
            locations = df['index'],
            locationmode='country names',
            z = df['Number'],
            text = df['index'],
            autocolorscale =False,
            reversescale = True,
            colorscale = 'rainbow',
            marker = dict(
                line = dict(
                    color = 'rgb(0,0,0)',
                    width = 0.5)
            ),
            colorbar = dict(
                title = 'Respondents',
                tickprefix = '')
        )

data = [trace]
layout = go.Layout(
    title = 'Number of respondents per country',
    geo = dict(
        showframe = True,
        showlakes = False,
        showcoastlines = True,
        projection = dict(
            type = 'natural earth'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)


# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## Highest level of formal education

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q4'), "Highest level of formal education", "Education", "Number of people", "Magenta")


# ## Undergraduate major
# 
# Next, let's see what is the answer distribution to the question `Which best describes your undergraduate major?`.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q5'), "Undergraduate major (best descrition)", "Undergraduate major", "Number of respondents", "Orange")


# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## Current title
# 
# The next question is about the description of the title.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q6'), "Current title", "Current title", "Number of respondents", "Red")


# ## Current employer industry
# 
# The next question is about the description of the industry of the current employer.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q7'), "Current employer industry", "Current employer industry", "Number of respondents", "Tomato")


# ## Years of experience in current industry
# 
# The following question is about the number of years of experience in the current position of the respondents.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q8'), "Years of experience in the current employer industry", "Years of experience", "Number of respondents", "Lightblue")


# 
# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## Current yearly compensation
# 
# The next question is about the current yearly compensatiomn.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q9'), "Current yearly compensation", "Current yearly compensation", "Number of respondents", "Gold")


# ## Usage of machine learning by current employer
# 
# The next question is about usage of machine learning by current employer business.

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q10'), "Does the current employer uses machine learning", "Use of machine learning", "Number of respondents", "Brown")


# ## Approximately what percent of your time at work or school is spent actively coding?
# 
# 

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q23'), multiple_df['Q23'][0], "Option", "Number of respondents", "Yellow")


# ## How long have you been writing code to analyze data?

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q24'), multiple_df['Q24'][0], "Option", "Number of respondents", "Lightgreen")


# ## For how many years have you used machine learning methods (at work or in school)?

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q25'), multiple_df['Q25'][0], "Option", "Number of respondents", "Orange")


# ## Do you consider yourself to be a data scientist?

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q26'), multiple_df['Q26'][0], "Option", "Number of respondents", "Lightblue")


# ## Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q40'), multiple_df['Q40'][0], "Option", "Number of respondents", "Magenta")


# ## Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q43'), multiple_df['Q43'][0], "Option", "Number of respondents", "Tomato")


# ## Approximately what percent of your data projects involve exploring model insights?

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q46'), multiple_df['Q46'][0], "Option", "Number of respondents", "Green")


# ## Are ML models *black boxes*?
# 
# Let's see the answers to the question `Do you consider ML models to be "black boxes" [...]`

# In[ ]:


draw_trace_bar(get_categories(multiple_df,'Q48'), "Are ML models `black boxes`?", "Are ML models `black boxes`?", "Number of respondents", "Black")


# 
# # <a id='4'>Combine the features</a>
# 
# 
# Let's visualize some of the dimmensions presented previously in combination. For example, let's see the combined distribution of sex and age to see how these two are distributed.
# 
# ## Number of respondents by Sex and Age

# In[ ]:


def get_categories_group(data, val_group, val):
    tmp = data[1::].groupby(val_group)[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


def draw_trace_group_bar(data_df, val_group, val, title, xlab, ylab,color='Blue'):
    data = list()
    groups = (data_df.groupby([val_group])[val_group].nunique()).index
    for group in groups:
        data_group_df = data_df[data_df[val_group]==group]
        trace = go.Bar(
                x = data_group_df[val],
                y = data_group_df['Number'],
                name = group,
                #marker=dict(color=color),
                text=data_group_df[val]
            )
        data.append(trace)

    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=15,
                          tickfont=dict(
                            size=9,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')


# In[ ]:


df = get_categories_group(multiple_df, 'Q1', 'Q2')
draw_trace_group_bar(df, 'Q1', 'Q2', 'Number of respondents by Sex and age', 'Age', 'Number of respondents')


# ## Number of respondents by Age and Highest level of formal education

# In[ ]:


df = get_categories_group(multiple_df, 'Q2', 'Q4')
draw_trace_group_bar(df, 'Q2', 'Q4', 'Number of respondents by Age and Highest level of formal education', 'Highest level of formal education', 'Number of respondents')


# ## Age and current yearly compensation

# In[ ]:


df = get_categories_group(multiple_df, 'Q2', 'Q9')
draw_trace_group_bar(df, 'Q2', 'Q9', 'Number of respondents by Age and Current yearly compensation', 'Current yearly compensation', 'Number of respondents')


# ## Highest level of formal education and current yearly compensation

# In[ ]:


df = get_categories_group(multiple_df, 'Q4', 'Q9')
draw_trace_group_bar(df, 'Q4', 'Q9', 'Number of respondents by Highest level of formal education and Current yearly compensation', 'Current yearly compensation', 'Number of respondents')


# ## Years of experience and current yearly compensation

# In[ ]:


df = get_categories_group(multiple_df, 'Q8', 'Q9')
draw_trace_group_bar(df, 'Q8', 'Q9', 'Number of respondents by Years of experience and Current yearly compensation', 'Current yearly compensation', 'Number of respondents')


# ## Current title and Highest level of education

# In[ ]:


df = get_categories_group(multiple_df, 'Q6', 'Q4')
draw_trace_group_bar(df, 'Q6', 'Q4', 'Number of respondents by Current title and Highest level of education', 'Current title', 'Number of respondents')


# 
# # <a id='5'>Final note</a>  
# 
# This Kernel is still under construction. Stay tuned, we will update it frequently in the following days.
# 
# <a href="#0"><font size="1">Go to top</font></a>  
