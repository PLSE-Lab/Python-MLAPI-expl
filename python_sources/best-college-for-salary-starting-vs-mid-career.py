#!/usr/bin/env python
# coding: utf-8

# First we import required packages to be used in reading and processing the input files

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from re import sub
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')


# Reading the input files

# In[ ]:


regional_salaries = pd.read_csv("../input/salaries-by-region.csv")
type_salaries = pd.read_csv("../input/salaries-by-college-type.csv")
majors = pd.read_csv("../input/degrees-that-pay-back.csv")


# In[ ]:


type_salaries.head(n=5)


# In[ ]:


majors.head( n = 5)


# In[ ]:


regional_salaries = regional_salaries.drop(regional_salaries.columns[4:], axis=1)


# We remove the $ sign in the numbers and transform them to float to be able to process the them

# In[ ]:


for col in range(2,4):
    for i in range(len(regional_salaries)):
        regional_salaries.iloc[i][col] = sub(r'[^\d.]', '', regional_salaries.iloc[i][col]) 
regional_salaries_float= regional_salaries.apply(pd.to_numeric, errors='ignore')


# In[ ]:


regional_salaries_float.boxplot(column='Starting Median Salary',by='Region')
plt.suptitle('')


# **Insights from above figure (Starting Median Salary vs region):**
# * Highest median starting salary is in California region
# * However, if you are good, or in have a major with reasonable market demand, you may command higher salaries in Northeastern region
# * On the other hand, if you are outlier, i.e. exceptionally good or with a major with very good demand, you are better off in California.

# In[ ]:


regional_salaries_float.boxplot(column='Mid-Career Median Salary',by='Region')
plt.suptitle('')


# **Insights from above figure (Mid-Career Median Salary vs region):**
# * Highest median Mid-career salary is in California region
# * However, if you are good, or in have a major with reasonable market demand, you may command higher salaries in Northeastern region

# <h1>Analysing the relation between school type, region and starting and mid-career salary</h1>
# 
# First we augment the "school type" table with the region of each school to have the better picture of salaries vs school and region of school.

# In[ ]:


regional_salaries.sort_values(['School Name'],ascending = [True])
university_region_dict = {}
for jj in range(len(regional_salaries)):
    university_region_dict[regional_salaries.loc[jj,'School Name']] = regional_salaries.loc[jj,'Region']


# In[ ]:


for kk in range(len(type_salaries)):
    if type_salaries.loc[kk,'School Name'] in university_region_dict.keys():
        type_salaries.loc[kk,'Region'] = university_region_dict[type_salaries.loc[kk,'School Name']]
    else:
        type_salaries.loc[kk,'Region'] = 'Southern'


# In[ ]:


type_salaries


# In[ ]:


for col in range(2,4):
    for i in range(len(type_salaries)):
        type_salaries.iloc[i,col] = sub(r'[^\d.]', '', type_salaries.iloc[i,col]) 
type_salaries_float= type_salaries.apply(pd.to_numeric, errors='ignore')


# In[ ]:


type_salaries_float['Salary Increase'] = type_salaries_float['Mid-Career Median Salary'] - type_salaries_float['Starting Median Salary']


# In[ ]:


type_salaries_float.boxplot(column='Starting Median Salary',by='School Type')
plt.suptitle('')


# Good news for Ivy league and engineering here on starting salary in the above figure. However ...

# In[ ]:


type_salaries_float.boxplot(column='Mid-Career Median Salary',by='School Type')
plt.suptitle('')


# ... When it gets to mid-career salary, being a Ivy leager clearly has its own advamntages compare to engineering school.

# <h1>Salary Increase, from starting to mid-career,  based on region and school type</h1>

# In[ ]:


type_salaries_float.boxplot(column='Salary Increase',by='School Type')
plt.suptitle('')


# In[ ]:


type_salaries_float.boxplot(column='Salary Increase',by='Region')
plt.suptitle('')


# In[ ]:


byType = type_salaries_float.groupby("School Type")
byType.groups.keys()


# In[ ]:


for name, group in byType:
    print(name)
    print(group["Mid-Career Median Salary"].describe())


# <h1>One scatter plot to visualize, starting vs mid-career salary for different majors</h1>
# 
# *Note: you can hover your mouse to see the information on each data point*

# In[ ]:


for col in [1,2,4,5,6,7]:
    for i in range(len(majors)):
        majors.iloc[i,col] = sub(r'[^\d.]', '', majors.iloc[i,col]) 
majors_float= majors.apply(pd.to_numeric, errors='ignore')
majors_float['maj'] = majors_float['Undergraduate Major']


# In[ ]:


from bokeh.plotting import figure,output_file, show,output_notebook
from bokeh.models import HoverTool,sources
from collections import OrderedDict
output_notebook()

x=majors_float['Starting Median Salary']
y=majors_float['Mid-Career Median Salary']
increase = 100 * (y-x)/x
label=majors_float['maj']
#from bokeh.plotting import *
source = sources.ColumnDataSource(
    data=dict(
        x=x,
        y=y,
        increase = increase,
        label=label
    )
)
TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
p = figure( x_axis_label = "Starting Salary",
            y_axis_label = "Mid-career Salary",title="Starting vs mid-career salary (Major)",plot_width = 800,plot_height = 800, tools=TOOLS)
p.circle('x', 'y', color="#2222aa", size=10, source=source)

hover =p.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([
    ("increase salary[%]","@increase"),
    ("label", "@label"),
])

show(p)


# **Top 3 mid-career salary for a  major**
# 
# * Chemical Engineering
# * Computer Engineering
# * Electrical Engineering
# 
# ** Top 3 startin salary for a  major**
# 
# * Physician Assistant
# * Chemical Engineering
# * Computer Engineering
# 
# ** Top 3 Majors with highest increase from starting to mid-career **
# 
# * Philosophy
# * Math
# * International relation

# <h1>One Scatter plot to rule them all!</h1>
# 
# In this section we visualize salary increase, from starting to mid-career, by school name and region.

# In[ ]:


colormap = {'California': 'red', 'Western': 'green', 'Northeastern': 'blue','Midwestern':'orange','Southern':'purple'}
colors = [colormap[x] for x in type_salaries_float['Region']]


# In[ ]:


x=type_salaries_float['Starting Median Salary']
y=type_salaries_float['Mid-Career Median Salary']
increase = 100 * (y-x)/x
label=type_salaries_float['School Name']
region = type_salaries_float['Region']
#from bokeh.plotting import *
source = sources.ColumnDataSource(
    data=dict(
        x=x,
        y=y,
        increase = increase,
        label=label,
        region = region
    )
)
TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
p1 = figure(x_axis_label = "Starting Salary",
            y_axis_label = "Mid-career Salary",title="Starting vs mid-career salary (Schools)",plot_width = 800,plot_height = 800, tools=TOOLS)
p1.circle('x', 'y', color=colors, size=10, source=source)

hover =p1.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([
    ("increase salary[%]","@increase"),
    ("label", "@label"),
    ("Region","@region")
])

show(p1)


# <h1>Insights from above figure </h1>
# 
# **Top 3 colleges for starting salary are**
# 
# * Caltech
# * MIT
# * Harvey Mudd
# 
# **Top 3 colleges for Mid-career salary are**
# 
# * Dartmouth
# * Princton
# * Yale
# 
# ** If you want to command median Mid-Career salary of 120K/year  you better study either in California or Northeastern region.**
# 
# ** In fact there are only handful of colleges outside California and Northeastern region provide the graduates with opportunity to earn median Mid-career salary of above 100K/year**
# 
# **Southern with Mid-career salary of  > 100K/year**
# 
# * Davidson College 
# * University of Virginia  
# * Washington and Lee university  
# * Georgia institue of Technology 
# 
# ** Mid-Western with Mid-career salary of  > 100K/year**
# 
# * Carlton college
# 
# ** Western with Mid-career salary of  > 100K/year**
# 
# * Colorado school of Mines
# 
# ** In total in the league of "Mid-career salary of > 100K" league there are:**
# 
# * 23 university and colleges from Northeaster region
# * 8 from California region
# * 4 from Southern region
# * 1 from Midwestern region
# * 1 from Western region
# 
# <h1>  Next Steps</h1>
# 
# 1. Including the tuition fee of each university can yield a better (fairer?) results toward public universities
# 2. Cost of living in each region also can be added to the analysis.
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




