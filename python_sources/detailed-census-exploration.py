#!/usr/bin/env python
# coding: utf-8

# **This script contains detailed data exploration of Census 2001 data.All the essential features have been covered**.*Any advises or suggestion regarding the improvement are most welcome*

# In[ ]:


# Importing essential packages

import os
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading csv file.
 
f=pd.read_csv("../input/all.csv")
data=DataFrame(f)
data.head()


# In[ ]:


data.shape


# In[ ]:


# Grouping by states

states=data.groupby('State').sum()
states=states.sort(['Persons'],ascending=[0])
states.reset_index()


# # Population

# In[ ]:


population=states[['Persons','Males','Females']]
population.head()


# In[ ]:


ax=plt.figure(figsize=(20,10))
ax=population.plot(kind='bar',stacked=True)
ax.set_axis_bgcolor('violet')


# # Sex Ratio

# In[ ]:


sex_ratio=states[['Sex.ratio..females.per.1000.males.','Sex.ratio..0.6.years.']].reset_index()
sorted_sex_ratio=(sex_ratio.sort(['Sex.ratio..females.per.1000.males.'],ascending=[0]))
g=sorted_sex_ratio.plot(kind='bar',cmap='Accent',stacked=True)
g.set_xticklabels(sex_ratio['State'])
g.set_axis_bgcolor('brown')


# # Education

# In[ ]:


# Here comes the pie chart
sum_df=data.sum()

labels = 'Graduate.and.Above','Below.Primary', 'Primary', 'Middle','Matric.Higher.Secondary.Diploma' 
sizes = [sum_df['Graduate.and.Above'],sum_df['Below.Primary'],sum_df['Primary'],sum_df['Middle'],sum_df['Matric.Higher.Secondary.Diploma']]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red']
explode = (0.1, 0, 0, 0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()


# # Water Facilities

# In[ ]:


water=states[['Drinking.water.facilities','Safe.Drinking.water']]
fig = plt.figure(figsize=(20,10))
plt.figure(figsize=(20,10))
fig=(water.sort(['Safe.Drinking.water'],ascending=[0])).plot(kind='bar',stacked=True)
fig.set_axis_bgcolor('red')


# # Schools

# In[ ]:


fig = plt.figure(figsize=(20,10))
school=DataFrame(states['Primary.school'])
sorted_School=school.sort(['Primary.school'],ascending=[0])
fig=sorted_School.plot(kind='bar',cmap='plasma')
fig.set_axis_bgcolor('green')


# In[ ]:




