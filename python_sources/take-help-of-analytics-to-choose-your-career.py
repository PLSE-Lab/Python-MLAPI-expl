#!/usr/bin/env python
# coding: utf-8

#                      #Lets explore investment in which colleges and courses will yield you high ROI                        

#                
# 
# ![](http://2mbetterfutures.org/wp-content/uploads/2016/07/Keep-strong-networking-after-college-for-good-career-opportunities.jpg)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Degree_df = pd.read_csv("../input//degrees-that-pay-back.csv")
College_df = pd.read_csv("../input//salaries-by-college-type.csv")
Region_df = pd.read_csv("../input//salaries-by-region.csv")


# In[ ]:


Degree_df.head()


# In[ ]:


#Degree_df['Mid-Career Median Salary']
#Degree_df.info()
Degree_df['Mid-Career Median Salary'] = Degree_df['Mid-Career Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
Degree_df['Mid-Career Median Salary'] = pd.to_numeric(Degree_df['Mid-Career Median Salary'], errors='coerce')


# From the below graph we can derive a coclusion that having an Engineering degree ensures you higher payscale. 
# But honestly I believe being the best in your chosen field will definately earn you satisfactory income.

# In[ ]:


Degree_df = Degree_df.sort_values("Mid-Career Median Salary", ascending=False).reset_index(drop=True)
#Degree_df['Mid-Career Median Salary']
f, ax = plt.subplots(figsize=(15, 15)) 
ax.set_yticklabels(Degree_df['Undergraduate Major'], rotation='horizontal', fontsize='large')
g = sns.barplot(y = Degree_df['Undergraduate Major'], x= Degree_df['Mid-Career Median Salary'])
plt.show()


# In[ ]:


Region_df.head()


# In[ ]:


Region_df['Mid-Career Median Salary'] = Region_df['Mid-Career Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
Region_df['Mid-Career Median Salary'] = pd.to_numeric(Region_df['Mid-Career Median Salary'], errors='coerce')


# Looks like having a degree from Colleges in California and Northeastern region will promise you better
# pay than the rest

# In[ ]:


sns.boxplot(x=Region_df['Region'],y=Region_df['Mid-Career Median Salary'])


# In[ ]:


#Region_df[Region_df['Mid-Career Median Salary'] == Region_df[Region_df['Region']=='Western']['Mid-Career Median Salary'].max()]
#Region_df[Region_df['Mid-Career Median Salary'] == Region_df[Region_df['Region']=='Midwestern Region']['Mid-Career Median Salary'].max()]
Region_df['Region'].unique()
#Region_df[Region_df['Mid-Career Median Salary'] == Region_df[Region_df['Region']=='Northeastern']['Mid-Career Median Salary'].max()]
#Region_df[Region_df['Mid-Career Median Salary'] == Region_df[Region_df['Region']=='California']['Mid-Career Median Salary'].max()]


# In[ ]:


x1 = Region_df[Region_df['Region']=='Southern']['Mid-Career Median Salary'].max()
x2 = Region_df[Region_df['Region']=='Western']['Mid-Career Median Salary'].max()
x3 = Region_df[Region_df['Region']=='Midwestern']['Mid-Career Median Salary'].max()
x4 = Region_df[Region_df['Region']=='Northeastern']['Mid-Career Median Salary'].max()
x5 = Region_df[Region_df['Region']=='California']['Mid-Career Median Salary'].max()


# In[ ]:


School_array = []
School_array.append(Region_df[(Region_df['Region']=='Southern') & (Region_df['Mid-Career Median Salary']==x1)]['School Name'].reset_index(drop=True)[0])
School_array.append(Region_df[(Region_df['Region']=='Western') & (Region_df['Mid-Career Median Salary']==x2)]['School Name'].reset_index(drop=True)[0])
School_array.append(Region_df[(Region_df['Region']=='Midwestern') & (Region_df['Mid-Career Median Salary']==x3)]['School Name'].reset_index(drop=True)[0])
School_array.append(Region_df[(Region_df['Region']=='Northeastern') & (Region_df['Mid-Career Median Salary']==x4)]['School Name'].reset_index(drop=True)[0])
School_array.append(Region_df[(Region_df['Region']=='California') & (Region_df['Mid-Career Median Salary']==x5)]['School Name'].reset_index(drop=True)[0])


# In[ ]:


School_array = str(School_array)
School_array


#                                    Top colleges from every region in word cloud

# In[ ]:


from wordcloud import WordCloud,STOPWORDS
from PIL import Image
from mpl_toolkits.basemap import Basemap
from os import path
text = College_df[pd.notnull(College_df["School Name"])]["School Name"]
# Generate the wordcloud
cloud = WordCloud(height=600, width=500, background_color="white", colormap="Blues", relative_scaling=0.2, 
        random_state=74364)
cloud.generate(" ".join(text))
# Plot wordcloud
plt.figure(figsize=(12,12))
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off");


# In[ ]:


College_df['Mid-Career Median Salary'] = College_df['Mid-Career Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Mid-Career Median Salary'] = pd.to_numeric(College_df['Mid-Career Median Salary'], errors='coerce')

College_df['Starting Median Salary'] = College_df['Starting Median Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Starting Median Salary'] = pd.to_numeric(College_df['Starting Median Salary'], errors='coerce')

College_df['Mid-Career 75th Percentile Salary'] = College_df['Mid-Career 75th Percentile Salary'].apply(lambda x: x.replace(",","").strip('$').split('.')[0])
College_df['Mid-Career 75th Percentile Salary'] = pd.to_numeric(College_df['Mid-Career 75th Percentile Salary'], errors='coerce')


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10), ncols=3, nrows=1)

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .8     # the amount of width reserved for blank space between subplots
hspace =  1.5    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)

# The amount of space above titles
y_title_margin = 1.0

ax[0].set_title("Starting Median Salary", y = y_title_margin)
ax[1].set_title("How much you earn throughout", y = y_title_margin)
ax[2].set_title("How much you will reach after reaching 3rd quarter of your career", y = y_title_margin)

ax[0].set_xticklabels(College_df['School Type'], rotation='vertical', fontsize='large')
ax[1].set_xticklabels(College_df['School Type'], rotation='vertical', fontsize='large')
ax[2].set_xticklabels(College_df['School Type'], rotation='vertical', fontsize='large')


sns.boxplot(x='School Type',y='Starting Median Salary', data=College_df,ax=ax[0])
sns.boxplot(x='School Type',y='Mid-Career Median Salary', data=College_df,ax=ax[1])
sns.boxplot(x='School Type',y='Mid-Career 75th Percentile Salary', data=College_df,ax=ax[2])

plt.tight_layout()

