#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mis
from matplotlib.ticker import PercentFormatter
import squarify as sq
from palettable.colorbrewer.qualitative import Pastel1_7

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing Data

# In[ ]:


df=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()


# Extracting the information about the dataset

# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# Checking if there are any missing values

# In[ ]:


df.isna().any()


# In[ ]:


mis.matrix(df, figsize=(15,8), fontsize=15, color=(1, 0.42, 0.5), sparkline=False)


# # Plotting nullity correlation
# 1: Implies that if the value of one variable is present, the value of the other variable is absent<br>
# 0: No correlation at all<br>
# 1: Implies that if the value of one variable is present the value of the other variable is definitely present<br>

# In[ ]:


mis.heatmap(df,cmap="PuBu", sort="ascending", figsize=(12,8), fontsize=12)


# Dendogram for the missing values correlation

# In[ ]:


fig = plt.figure(figsize=(15,7))

ax1 = fig.add_subplot(1,2,1)
mis.dendrogram(df, orientation="right", method="centroid", fontsize=12, ax=ax1)

ax2 = fig.add_subplot(1,2,2)
mis.dendrogram(df, orientation="top", method="ward", fontsize=12, ax=ax2)

plt.tight_layout()


# In[ ]:


df.dropna(axis=0,inplace=True)


# In[ ]:


df.isna().any()


# # Most popular locations

# In[ ]:


top_locations=df['Location'].value_counts().head(10)
top_locations


# In[ ]:


len(df['Location'])


# In[ ]:


top_location_names=df['Location'].value_counts().head(10).index
top_location_names


# Pareto Chart for the location

# In[ ]:


cumulative=(top_locations.cumsum()/len(df['Location']))*100
cumulative=cumulative.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(20,10))
ax.bar(top_location_names, top_locations, color="C3")
ax2 = ax.twinx()
ax2.plot(top_location_names, cumulative, color="C10", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax.tick_params(axis="y", colors="C3")
ax2.tick_params(axis="y", colors="C10")
plt.show()


# # Job Titles in Demand

# In[ ]:


job_title=df['Job Title'].value_counts().head(10)
job_title


# Square map

# In[ ]:


plt.figure(figsize=(14,8))
sq.plot(sizes=job_title, label=job_title.index, alpha=.8)
plt.axis('off')
plt.show()


# Bar Chart

# In[ ]:


plt.figure(figsize=(32,15))
plt.title('Top 10 Job Titles',fontsize=30)
plt.xticks(fontsize=10)
plt.yticks(fontsize=15)
plt.xlabel('Job Titles', fontsize=20)
sns.barplot(x=job_title.index,y=job_title,color="cyan", estimator=max)


# # Job Experience 

# Categories of experience available

# In[ ]:


years_reqd=df['Job Experience Required'].value_counts().head(10)


# Now we have to create categories of different experience years so that we can get a better idea

# In[ ]:


experience=pd.DataFrame(years_reqd)
experience.reset_index(inplace=True)
experience


# In[ ]:


experience.rename(columns={'index':'Job Experience','Job Experience Required':'Number of vacancies'},inplace=True)
experience


# In[ ]:


experience.loc[experience['Job Experience'].str.contains('2 - 5 yrs',case=False),'Experience Category']='Experienced'
experience.loc[experience['Job Experience'].str.contains('5 - 10 yrs',case=False),'Experience Category']='Highly Experienced'
experience.loc[experience['Job Experience'].str.contains('2 - 7 yrs',case=False),'Experience Category']='Experienced'
experience.loc[experience['Job Experience'].str.contains('3 - 8 yrs',case=False),'Experience Category']='Highly Experienced'
experience.loc[experience['Job Experience'].str.contains('1 - 3 yrs',case=False),'Experience Category']='Experienced'
experience.loc[experience['Job Experience'].str.contains('1 - 6 yrs',case=False),'Experience Category']='Experienced'
experience.loc[experience['Job Experience'].str.contains('3 - 5 yrs',case=False),'Experience Category']='Experienced'
experience.loc[experience['Job Experience'].str.contains('1 - 5 yrs',case=False),'Experience Category']='Experienced'
experience.loc[experience['Job Experience'].str.contains('0 - 5 yrs',case=False),'Experience Category']='Freshers'
experience.loc[experience['Job Experience'].str.contains('0 - 1 yrs',case=False),'Experience Category']='Freshers'


# In[ ]:


experience


# In[ ]:


category=experience.groupby('Experience Category').sum()
category


# In[ ]:


cat=category.index
number=category['Number of vacancies']
fig = plt.figure(figsize=(10,15))
plt.title("Share of different experience categories", fontsize=25)
# fig.patch.set_facecolor('black')
plt.rcParams['text.color'] = 'black'
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(number, labels=cat, colors=["Red","Blue","Black"],wedgeprops = { 'linewidth' : 9, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# # Emerging industries

# In[ ]:


industry=df['Industry'].value_counts().head(20)
industry


# In[ ]:


plt.figure(figsize=(32,15))
plt.title('Emerging industries',fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Industries', fontsize=20)
plt.xlabel('Count',fontsize=20)
sns.barplot(x=industry,y=industry.index,estimator=max)


# 

# In[ ]:


from math import pi

from bokeh.io import output_file, show
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum


# In[ ]:


data=pd.Series(industry).reset_index(name='count').rename(columns={'index':'industry'})
data['angle'] = data['count']/data['count'].sum() * 2*pi
data['color'] = Category20c[len(industry)]


# In[ ]:


p = figure(plot_height=350, title="Top 20 Emerging Industries", toolbar_location=None,
           tools="hover", tooltips="@industry: @count", x_range=(-0.6, 1.0))

p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_label='country', source=data)

p.axis.axis_label=None
p.axis.visible=False
p.grid.grid_line_color = None

show(p)

