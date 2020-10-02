#!/usr/bin/env python
# coding: utf-8

# # How to Create Award Winning Data Visualizations
# 
# ![chart intro](https://i.imgur.com/iRqHgrC.png)
# 
# 
# -----------------
# 
# Earlier this week I won the third prize (U$ 6.000) on Kaggle's [Machine Learning and Data Science Survey Competition](https://www.kaggle.com/c/kaggle-survey-2019) with my notebook [Is there any job out there? Kaggle vs Glassdoor](https://www.kaggle.com/andresionek/is-there-any-job-out-there-kaggle-vs-glassdoor). 
# 
# While the competition evaluated several aspects of my submission, I think that the way that I displayed information was a central piece in getting awarded. **Here I want to share a little bit of the thought process behind building the data visualizations. With step by step instructions.**
# 
# -----

# # 1. Loading Data and preparing it for plotting
# 
# Boilerplate to load and prepare data... unhide if you want to see it.

# In[ ]:


## Basic Cleaning - Kaggle Survey
import numpy as np
import pandas as pd
import os
import re

# Loading the multiple choices dataset from Kaggle Survey, we will not look to the free form data on this study
kaggle_multiple_choice = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)
kaggle_multiple_choice.head()


# In[ ]:


# This DataFrame stores all answers
kaggle = kaggle_multiple_choice.iloc[1:,:]
kaggle.head()


# In[ ]:


# removing everyone that took less than 3 minutes or more than 600 minutes to answer the survey
answers_before = kaggle.shape[0]
print(f'Initial dataset length is {answers_before} answers.')

# Creating a mask to identify those who took less than 3 min
less_3_minutes = kaggle[round(kaggle.iloc[:,0].astype(int) / 60) <= 3].index
# Dropping those rows
kaggle = kaggle.drop(less_3_minutes, axis=0)

# Creating a mask to identify those who took more than 600 min
more_600_minutes = kaggle[round(kaggle.iloc[:,0].astype(int) / 60) >= 600].index
kaggle = kaggle.drop(more_600_minutes, axis=0)

answers_after = kaggle.shape[0]
print('After removing respondents that took less than 3 minutes or more than 600 minutes'       f'to answer the survey we were left with {answers_after} answers.')


# In[ ]:


# removing respondents who are not employed or project/product managers
answers_before = kaggle.shape[0]

# Creating a mask to identify Job titles that are not interesting for this study
students_and_others = kaggle[(kaggle.Q5 == 'Student') |                              (kaggle.Q5 == 'Other') |                              (kaggle.Q5 == 'Not employed') |                              (kaggle.Q5 == 'Product/Project Manager')
                            ].index
# Dropping rows
kaggle = kaggle.drop(list(students_and_others), axis=0)
answers_after = kaggle.shape[0]
print(f'After removing respondents who are not employed or project/product managers we were left with {answers_after} answers.')


# In[ ]:


# Removing those who didn't disclose compensation (Q10 is NaN)
answers_before = kaggle.shape[0]
kaggle.dropna(subset=['Q10'], inplace=True)
answers_after = kaggle.shape[0]
print(f'After removing respondents who did not disclose compensation there were left {answers_after} answers.')


# In[ ]:


# Now lets group some data
kaggle.Q5.value_counts()


# In[ ]:


# Groupping DBA + Data Engineer
kaggle.Q5 = kaggle.Q5.replace('DBA/Database Engineer', 'Data Engineer/DBA')
kaggle.Q5 = kaggle.Q5.replace('Data Engineer', 'Data Engineer/DBA')
kaggle.Q5.value_counts()


# In[ ]:


# Groupping Statistician + Research Scientist
kaggle.Q5 = kaggle.Q5.replace('Statistician', 'Statistician/Research Scientist')
kaggle.Q5 = kaggle.Q5.replace('Research Scientist', 'Statistician/Research Scientist')
kaggle.Q5.value_counts()


# In[ ]:


# Simplifying country names
kaggle.Q3 = kaggle.Q3.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
kaggle.Q3 = kaggle.Q3.replace('United States of America', 'United States')


# In[ ]:


# Now lets rename some columns to have more meaningfull names
kaggle.columns = kaggle.columns.str.replace('Q5', 'JobTitle')
kaggle.columns = kaggle.columns.str.replace('Q3', 'Country')


# In[ ]:


# Add count column to make groupby easier
kaggle['Count'] = 1


# In[ ]:


# Filtering only the columns we will need
kaggle = kaggle[['Country', 'JobTitle', 'Count']]
# Finally our Dataframe looks like this
kaggle.head(10)


# # 2. Data to Plot
# My idea was to measure how many people are working in different positions (job titles) comparing different countries.

# In[ ]:


# Grouping it by job title
plot_data = kaggle.groupby(['JobTitle'], as_index=False).Count.sum()
plot_data


# In[ ]:


# Grouping it by job title and country
plot_data = kaggle.groupby(['JobTitle', 'Country'], as_index=False).Count.sum()
plot_data


# # 3. First Plot
# I choose top use [Plotly](https://plot.ly/python/) mainly because I'm already used to it and because I enjoy the interactivity it provides out of the box. However the concepts that I applied, and that I'm going to demonstrate can be used with any other graphing library or software, language agnostic.

# In[ ]:


import plotly.express as px

plot_data = kaggle.groupby(['JobTitle'], as_index=False).Count.sum()

fig = px.bar(plot_data, x='JobTitle', y='Count')
fig.show()


# In[ ]:


import plotly.express as px

# Grouping it by job title and country
plot_data = kaggle.groupby(['JobTitle', 'Country'], as_index=False).Count.sum()

fig = px.bar(plot_data, x='JobTitle', y='Count', color='Country')
fig.show()


# # 4. Choosing a better plot type
# I was not satisfied with a bar plot for this visualization. I wanted that could display the same information and allow comparing different countries. Adding multiple bar charts for multiple countries would not look good at all. Then I tryed a line chart, since adding multiple lines works better than multiple columns.
# 

# In[ ]:


import plotly.express as px

plot_data = kaggle.groupby(['JobTitle'], as_index=False).Count.sum()

fig = px.line(plot_data, x='JobTitle', y='Count')
fig.show()


# Much better... now lets try to add countries, instead of the aggregated of all of them. 

# In[ ]:


import plotly.express as px

# Grouping it by job title and country
plot_data = kaggle.groupby(['JobTitle', 'Country'], as_index=False).Count.sum()

fig = px.line(plot_data, x='JobTitle', y='Count', color='Country')
fig.show()


# Looking better than the bars... But still not good. My next idea was to transform the x axis into a radial.

# In[ ]:


import plotly.express as px

# Grouping it by job title and country
plot_data = kaggle.groupby(['JobTitle', 'Country'], as_index=False).Count.sum()

fig = px.line_polar(plot_data, theta='JobTitle', r='Count', color='Country')
fig.show()


# This looks interesting! I think I can go forward with this chart type.

# # 5. Changing the data to be plotted from absolute to relative
# Some countries have much more absolute rows than others. This creates a distortion in the chart, so I will change the metric and plot the relative percentage for each job title in each country instead. This should allow us to compare them! 

# In[ ]:


# Here plotly express wasn't usefull anymore, so I changed to the standard Plotly API.
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode()

figure = go.Figure()

# List of all countries
countries = list(set(kaggle.Country.tolist()))

for country in countries:
    # iterates in the list of countries
    data_filtered = kaggle[kaggle.Country == country] # filters data for a single country
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum() # group the data by jobtitle
    axis = plot_data.JobTitle
    plot_data = plot_data.Count.tolist() # Get count and convert it to a list 
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist() # Transform absolute values into percentages
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, # Data points 
            theta=axis, # Axis (Software Engineer, Data Scientist, etc)
            mode='lines'
        )
    )
figure.show()


# Indeed! Now all countries are at the same scale. Comparing them proportionaly is much better than comparing absolute values.

# # 6. Closing the lines
# This is a little bit trickier... To "close" the radar chart on Plotly, you need to add the first data point again to the end.[[](http://)](http://)

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

for country in countries:
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    
    axis.append(axis[0]) # appendind the first element on axis
    
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    
    plot_data.append(plot_data[0]) # appendind the first element on plot data
        
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            mode='lines'
        )
    )
figure.show()


# # 7. Creating a hierarchy in the axis titles
# If you look at the axis they don't currently have an hierarchy or any specific order. I wan't to create one by keeping close togheter job titles that are similar, from the least technichal to the most technical. For me it makes sense to organize it in the following order:
# 
# > Business Analyst **<** Data Analyst **<** Data Scientist **<** Data Engineer/DBA **<** Software Engineer **<** Statistician/Research Scientist

# In[ ]:


# And the same for JobTitle column. Transform it into category
job_titles = ['Business Analyst', 'Data Analyst', 'Data Scientist', 
              'Data Engineer/DBA', 'Software Engineer', 'Statistician/Research Scientist']
cat_dtype = pd.api.types.CategoricalDtype(categories=job_titles, ordered=True)
kaggle.JobTitle = kaggle.JobTitle.astype(cat_dtype)


# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

for country in countries:
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    
    axis.append(axis[0]) # appendind the first element on axis
    
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    
    plot_data.append(plot_data[0]) # appendind the first element on plot data
        
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            mode='lines'
        )
    )
figure.show()


# # 8. Declutering 
# This is the most important part of any good data visualization. Remove any noise that is not directly contributing to how people are reading and interpreting data.
# 
# ## 8.1 Turning background to white
# Now that the background is white, we also need to change some other elements, such as grids, ticks and lines. Here my choices were:
# * Turn all text and chart elements to a light grey
# * Hide any chart grid and axis lines that are not essential. I only kept radial axis grid (the circles), because they are important for readers to understand the magnitude of the values displayed.

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

for country in countries:
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            mode='lines'
        )
    )
    
figure.update_layout(
    polar_bgcolor='white', # White background is always better
    polar_radialaxis_visible=True, # we want to show the axis
    polar_radialaxis_showticklabels=True, # we want to show the axis titles
    polar_radialaxis_tickfont_color='darkgrey', # grey to the axis label (Software Engineer, Data Scientist, etc)
    polar_angularaxis_color='grey', # Always grey for all elements that are not important
    polar_angularaxis_showline=False, # hide lines that are not necessary
    polar_radialaxis_showline=False, # hide lines that are not necessary
    polar_radialaxis_layer='below traces', # show the axis bellow all traces
    polar_radialaxis_gridcolor='#F2F2F2', # grey to not draw attention
)

figure.show()


# It looks much better now, doesn't it?
# 
# ## 8.2 Smoothing all lines, and making they grey and thinner
# Now we will transform all lines to grey, reduce their width and also make them less sharp by applying some smoothing. 

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

for country in countries:
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            mode='lines',
            
            line_color='slategrey', # line color
            opacity= 0.25, # Oppacity to 0.25 to reduce clutter.
            line_shape='spline', # This will allow smoothing the lines
            line_smoothing=0.8, # How much the lines will smooth
            line_width=0.6 # thinner lines 
        
        )
    )
    
figure.update_layout(
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='grey',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
)

figure.show()


# This is SO MUCH BETTER! Because now we can actually "see" each line! But as they say, the devil is in the details. Let's keep improving it!
# 
# ## 8.3 Removing legends, adjusting grid and range
# * Now I thought that there where too much lines for grids (0, 10, 20, ..., 50) cluttering the chart. I don't like it, instead I will create only three lines (25% and 50%). 
# * I also don't like that many values for legend, they dont help with anything other than making our visualization worse. Let's hide it!
# * There is one line that is almost over **Data Scientist** axis title, I will increase the range a little bit to avoid that.

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

for country in countries:
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            
            showlegend=False, # Hide legend
            
            mode='lines',
            line_color='slategrey',
            opacity= 0.25,
            line_shape='spline',
            line_smoothing=0.8,
            line_width=0.6        
        )
    )
    
figure.update_layout(
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='grey',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
    
    polar_radialaxis_range=(0,72), # increases the range to avoid line over axis title.
    polar_radialaxis_tickvals=[25, 50], # show ticks at 25, 50
    polar_radialaxis_ticktext=['25%', '50%'], # Label ticks accordingly
    polar_radialaxis_tickmode='array', # boilerplate for this to work
)

figure.show()


# # 9. Adjusting hover information
# When we hover the mouse over the datapoints we are seeing **r** and **theta** values. This is not informative for anyone that is viewing the chart. Let's adjust a few things there.
# 
# * Add the country name (so it's easy to relate the line to the country without needing to have multiple colors or legends)
# * Add the data point (so it's easy to relate the value of each data point to each line)
# * Format the data point as percentage without decimal points (no need to have any decimal points, because the analysis is more qualitative) 
# 

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

for country in countries:
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            showlegend=False,
            
            name=country, # name to be exibited on legend and on hover
            hoverinfo='name+r', # what to show on hover (name + data point)
            hovertemplate='%{r:0.0f}%', # Format of data point
            
            mode='lines',
            line_color='slategrey',
            opacity= 0.25,
            line_shape='spline',
            line_smoothing=0.8,
            line_width=0.6        
        )
    )
    
figure.update_layout(
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='grey',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
    polar_radialaxis_range=(0,72),
    polar_radialaxis_tickvals=[25, 50],
    polar_radialaxis_ticktext=['25%', '50%'],
    polar_radialaxis_tickmode='array',
)

figure.show()


# # 10. Telling a story
# Now we will highlight some specific countries that we want to tell a story about. For this chart I'm choosing United States vs China. What we will do:
# 
# * Add color to highlighted countries lines
#     * Choose color from flags colors (red for China, blue for United States)
# * Increase width of highlighted countries lines
# * Increase opacity of highlighted countries lines
# * Show legend only for highlighted countries

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

country_color = {
    'United States': '#002366', 
    'China': '#ED2124', 
} # creates dict of colors for highlights

for country in countries:
    
    color = country_color.get(country, 'lightslategrey') # Get color of each line from country_color dict
    highlight = color != 'lightslategrey' # We only want to highlight a few traces, this will decide if a trace is highlighted or not
    
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            
            showlegend=highlight, # Show legend only if highlight
            
            name=country, 
            hoverinfo='name+r',
            hovertemplate='%{r:0.0f}%',
            mode='lines',
            
            line_color=color, # Line color
            opacity=0.8 if highlight else 0.25, # Different oppacity for highlighted lines
            
            line_shape='spline',
            line_smoothing=0.8,
            
            line_width=1.6 if highlight else 0.6 # Different width for highlighted lines     
        )
    )
    
figure.update_layout(
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='grey',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
    polar_radialaxis_range=(0,72),
    polar_radialaxis_tickvals=[25, 50],
    polar_radialaxis_ticktext=['25%', '50%'],
    polar_radialaxis_tickmode='array',
)

figure.show()


# Now we are telling some story with this data! We are able to see the differences between China and United States! We can easily draw some conclusions from this chart!
# 
# # 11. Adding a meaningful title
# Now we will add a meaninful title and the data source to the chart.
# 
# * Bigger font size for title
# * Smaller font size for source
# * Both grey to not draw much attention

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

country_color = {
    'United States': '#002366', 
    'China': '#ED2124', 
} 

for country in countries:
    color = country_color.get(country, 'lightslategrey')
    highlight = color != 'lightslategrey' 
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            showlegend=highlight, 
            name=country, 
            hoverinfo='name+r',
            hovertemplate='%{r:0.0f}%',
            mode='lines',
            line_color=color,
            opacity=0.8 if highlight else 0.25,
            line_shape='spline',
            line_smoothing=0.8,
            line_width=1.6 if highlight else 0.6 
        )
    )
    
title = 'Proportionally USA has more Data Scientists, while China has more Software Engineers.'         '<br><span style="font-size:10px"><i>Kaggle Survey Q5: Select the title most similar to '         'your current role (sums to 100% for each country)</span></i>'

figure.update_layout(
    
    title_text = title, # add title to chart
    title_font_color = '#333333', # Grey is always better to not draw much attention
    title_font_size = 14,
    
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='grey',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
    polar_radialaxis_range=(0,72),
    polar_radialaxis_tickvals=[25, 50],
    polar_radialaxis_ticktext=['25%', '50%'],
    polar_radialaxis_tickmode='array',
)

figure.show()


# # 12. Bringing all elements more close together
# Using the standard autosize of Plotly is all elements too much separated from each other, let's bring them closer by defining the height and width size of the chart.
# 
# Also some minor adjustments to legend as well:
# * Changing color to grey
# * Change behaviour when click and doubleclick

# In[ ]:


pyo.init_notebook_mode()

figure = go.Figure()

countries = list(set(kaggle.Country.tolist()))

country_color = {
    'United States': '#002366', 
    'China': '#ED2124', 
} 

for country in countries:
    color = country_color.get(country, 'lightslategrey')
    highlight = color != 'lightslategrey' 
    data_filtered = kaggle[kaggle.Country == country]
    plot_data = data_filtered.groupby(['JobTitle'], as_index=False).Count.sum()
    axis = plot_data.JobTitle.tolist()
    axis.append(axis[0])
    plot_data = plot_data.Count.tolist()
    plot_data = (np.array(plot_data) / sum(plot_data) * 100).tolist()
    plot_data.append(plot_data[0])     
    figure.add_trace(
        go.Scatterpolar(
            r=plot_data, 
            theta=axis,
            showlegend=highlight, 
            name=country, 
            hoverinfo='name+r',
            hovertemplate='%{r:0.0f}%',
            mode='lines',
            line_color=color,
            opacity=0.8 if highlight else 0.25,
            line_shape='spline',
            line_smoothing=0.8,
            line_width=1.6 if highlight else 0.6 
        )
    )
    
title = 'Proportionally USA has more Data Scientists, while China has more Software Engineers.'         '<br><span style="font-size:10px"><i>Kaggle Survey Q5: Select the title most similar to '         'your current role (sums to 100% for each country)</span></i>'

figure.update_layout(
    title_text = title,
    title_font_color = '#333333',
    title_font_size = 14,    
    polar_bgcolor='white',
    polar_radialaxis_visible=True,
    polar_radialaxis_showticklabels=True,
    polar_radialaxis_tickfont_color='darkgrey',
    polar_angularaxis_color='grey',
    polar_angularaxis_showline=False,
    polar_radialaxis_showline=False,
    polar_radialaxis_layer='below traces',
    polar_radialaxis_gridcolor='#F2F2F2',
    polar_radialaxis_range=(0,72),
    polar_radialaxis_tickvals=[25, 50],
    polar_radialaxis_ticktext=['25%', '50%'],
    polar_radialaxis_tickmode='array',
    
    legend_font_color = 'grey', # We don't want to draw attention to the legend 
    legend_itemclick = 'toggleothers', # Change the default behaviour, when click select only that trace
    legend_itemdoubleclick = 'toggle', # Change the default behaviour, when double click ommit that trace
    width = 800, # chart size 
    height = 500 # chart size
)

figure.show()


# # 13. More tips on how to do a great data storytelling
# 
# ## 13.1 Keep all your charts similar
# When showing multiple charts, stick to a few different chart types. The purpose of this is to make your readers lifes easier, as they will not need to learn how to read every new chart. After learning how to interpret one chart, reading all the other charts will be automatic.
# 
# ## 13.2 Stick to a few colors in every chart
# If your highlighted data colors don't hold any meaning (unlike this example, where they encode the color of country flag) then you shouldn't use a lot of different colors. Keep them to a minimum.
# 
# ## 13.4 Tell just one story per chart
# It might be tempting to highlight multiple things in a single chart. Instead create multiple charts to highlith different things, one single story per chart. 

# In[ ]:




