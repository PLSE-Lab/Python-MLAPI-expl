#!/usr/bin/env python
# coding: utf-8

# # **CORONA VIRUS**
# 
# * Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans.
# 
# * Coronaviruses are zoonotic, meaning they are transmitted between animals and people.  Detailed investigations found that SARS-CoV was transmitted from civet cats to humans and MERS-CoV from dromedary camels to humans. Several known coronaviruses are circulating in animals that have not yet infected humans. 
# 
# * Common signs of infection include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death. 
# 
# * Standard recommendations to prevent infection spread include regular hand washing, covering mouth and nose when coughing and sneezing, thoroughly cooking meat and eggs. Avoid close contact with anyone showing symptoms of respiratory illness such as coughing and sneezing.
# 
# [World Health Organization - 2020](https://www.who.int/health-topics/coronavirus)

# # **CASE STUDY**
# 
# * In this study we will create a visualization using the updated data of corona virus in the world

# # **LIBRARIES**
# 
# In this case we are using:
# 
# * pandas to manipulate the dataset;
# * datetime to working with dates;
# * matplotlib to plot charts;
# * HTML from IPython to visualize the final results.

# In[ ]:


# Importing Python Libraries

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

from IPython.display import HTML


# In[ ]:


import matplotlib
matplotlib.use("Agg")


# # **DATASET**
# 
# For this case we are using a Kaggle Dataset available in:
# 
# https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

# In[ ]:


# Importing dataset and seting "Last Update" and "ObservationDate" as date by parse_dates

dataset = (pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',
                       parse_dates = ['Last Update', 'ObservationDate'])
                       .sort_values(by='Last Update', ascending = False))
dataset.head(10)


# # **DATASET PRE-PROCESSING**
# 
# 1. After import we need to understand the dataset and prepare the data to work:

# In[ ]:


# Visualize empty values in dataset
dataset.isna().sum()


# In[ ]:


#Rename Mainland China to China
dataset['Country/Region'].replace('Mainland China', 'China', inplace = True)

#Filling empty provinces with "NA"
dataset['Province/State'].fillna('NA', inplace = True)


# 2. List of countries infected by Corona Virus:

# In[ ]:


list_infected_countries = pd.DataFrame(data = dataset['Country/Region'].sort_values().unique(), columns = {'Country/Region'})

num_infected_countries = len(list(dataset['Country/Region'].sort_values().unique()))

print("Actually there's %d countries infected by Corona Virus in the World \n \n" %
      len(list(dataset['Country/Region'].sort_values().unique())))

list_infected_countries


# 3. Creating a temporary dataframe to visualize the top 10 infected countries by day

# In[ ]:


# Last observation date in dataset
last_data_day = dataset['ObservationDate'].max()

# Filtering the dataset with the selected date
df = dataset[dataset['ObservationDate'].eq(last_data_day)]

# Creating a dataset grouped by countries and sortened by confirmed cases
df_group = pd.DataFrame(data = (df.groupby(['Country/Region'], as_index = False)
      .sum()
      .sort_values(by='Confirmed', ascending=False)
      .head(10)
      .reset_index(drop=True)))

# Removing 'SNo' column
df_group.drop(columns = ['SNo'], inplace = True)

df_group


# 4. Creating a chart bar with Countries by Confirmed cases

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
df_group = df_group[::-1]
ax.barh(df_group['Country/Region'], df_group['Confirmed'])


# 5. Define a function to draw the chart bar with a range of dates

# In[ ]:


def draw_barchart(day):
    
    #Creating Top 10 Confirmed Dataset
    
    df = dataset[dataset['ObservationDate'].eq(day)]
    
    df_group = (df.groupby(['Country/Region'], as_index = False)
          .sum()
          .sort_values(by='Confirmed', ascending=False)
          .head(10)
          .reset_index(drop=True))

    df_group.drop(columns = ['SNo'], inplace = True)
    
    #Creating Bar Chart
    ax.clear()
    df_group = df_group[::-1]
    ax.barh(df_group['Country/Region'], df_group['Confirmed'])
    
    dx = df_group['Confirmed'].max() / 1000
    
    #Format Bar Chart
    for i, (value, name) in enumerate(zip(df_group['Confirmed'], df_group['Country/Region'])):
        
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='center')
        #ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    
    ax.text(1, 0.4, day.strftime("%d/%m/%Y"), transform=ax.transAxes, color='#777777', size=30, ha='right', weight=600)
    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'Confirmed Corona Virus cases in the world',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by @joaocampista', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)


# 6. Plotting a chart bar using the created function for the last observation date in dataset

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
draw_barchart(dataset['ObservationDate'].max())


# 7. Automated chart generator using a date interval

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))

day_zero = dataset['ObservationDate'].min()
day_target = dataset['ObservationDate'].max()
dates = list(pd.date_range(day_zero, day_target))

animator = animation.FuncAnimation(fig, draw_barchart, frames=dates) 

HTML(animator.to_jshtml())


# # **CONFIRMED CORONA VIRUS CASES OUTSIDE CHINA**
# To create this vizualization we need:
# * Prepare the dataset removing china from the list of infected Countries;
# * Define the function to draw the chart;
# * Define the interval of dates to call the function;
# * Plot the chart.

# In[ ]:


dataset_w_c = dataset[~dataset['Country/Region'].eq('China')].sort_values(by='Confirmed', ascending = False)
dataset_w_c.head(10)


# In[ ]:


def draw_barchart_w_c(day):
    
    #Creating Top 10 Confirmed Dataset
    
    df = dataset_w_c[dataset_w_c['ObservationDate'].eq(day)]
    
    df = (df.groupby(['Country/Region'], as_index = False)
          .sum()
          .sort_values(by='Confirmed', ascending=False)
          .head(10)
          .reset_index(drop=True))

    df.drop(columns = ['SNo'], inplace = True)
    
    #Creating Bar Chart
    ax.clear()
    df = df[::-1]
    ax.barh(df['Country/Region'], df['Confirmed'])
    
    dx = df['Confirmed'].max() / 1000
    
    #Format Bar Chart
    for i, (value, name) in enumerate(zip(df['Confirmed'], df['Country/Region'])):
        
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='center')
        #ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    
    ax.text(1, 0.4, day.strftime("%d/%m/%Y"), transform=ax.transAxes, color='#777777', size=30, ha='right', weight=600)
    ax.text(0, 1.06, 'Confirmed Cases', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'Confirmed Corona Virus cases outside China',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by @joaocampista', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    

fig, ax = plt.subplots(figsize=(15, 8))
draw_barchart_w_c(dataset['ObservationDate'].max())


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))

day_zero = dataset_w_c['ObservationDate'].min()
day_target = dataset_w_c['ObservationDate'].max()
dates = list(pd.date_range(day_zero, day_target))

animator = animation.FuncAnimation(fig, draw_barchart_w_c, frames=dates)

HTML(animator.to_jshtml())

