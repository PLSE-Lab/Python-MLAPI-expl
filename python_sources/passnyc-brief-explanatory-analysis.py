#!/usr/bin/env python
# coding: utf-8

# # Hi there!
# This notebook is currently a work in progress. If this is anyway helpful, please upvote! More to come!  
# -Tristan
# ______
# 
# # PASSNYC: Brief Explanatory Analysis
# ![PASSNYC](http://static1.squarespace.com/static/5576f1c0e4b08f31a497b582/t/5576fcf6e4b0a6d0afa92d0f/1527185421758/)
# 
# **Outline:**
# 1. [About PASSNYC](## About PASSNYC)
# 2. [Objectives](## Objectives)
# 3. [Loading the Libraries](## Loading the Libraries)
# 4. [About the Data](## About the Data)
# 5. [Exploring the Data](## Exploring the Data)  
# 5.1.[Registrations and Testers](### Registrations and Testers)  
# 5.1.1. [School Distribution](#### School Distribution)  
# 5.1.2. [Year of SHST Distribution](#### Year of SHST Distribution)  
# 5.1.3. [Grade Level Distribution](#### Grade Level Distribution)  
# 5.1.4. [Enrollment on 10/31 Descriptive Statistics](#### Enrollment on 10/31 Descriptive Statistics)  
# 5.1.5. [Enrollment on 10/31 by School Name](#### Enrollment on 10/31 by School Name)  
# 5.1.6. [Number of students who registered for the SHSAT Descriptive Statistics](#### Number of students who registered for the SHSAT Descriptive Statistics)  
# 5.1.7. [Number of students who registered for the SHSAT by School Name](#### Number of students who registered for the SHSAT by School Name)    
# 5.1.8. [Difference between registered and test takers](#### Difference between registered and test takers)  
# 5.2. [2016 School Explorer Dataset](### 2016 School Explorer Dataset)  
# 5.2.1. ...  
# 5.2.2. ...  
# 5.2.3. ...  
# 
# 6. [Summary](## Summary)  
#  
# ** Key Findings:** (so far)
# * There is a steady increase of number of school on SHST per year starting from 33 in 2016 to 37 in 2016. 
# * 60.7% of schools on PASSNYC's records are on Grade 8 while 39.3% are on Grade 9. 
# * Frederick Douglas Academy gets the most number of enrollments on 10/31 with 1640 recorded enrollments. This is followed by Democracy Prep Charter School with 943 enrollees and Democracy Prep Harlem School with 896 enrollees. 
# * KIPP STAR College Prep Charter School had the most number of students registered for SHSAT.
# * Academy fo Social Action doesn't have any student to take the SHSAT.
# * Both KIPP affiliated schools are the top schools with the most number of registered for SHSAT.
# * However, not all students who registered took the SHSAT.
# * The average number of students no taking the exam is 9 students, with maximum of 92.

# ## About PASSNYC
# PASSNYC is a not-for-profit organization that facilitates a collective impact that is dedicated to broadening educational opportunities for New York City's talented and underserved students. 
# 
# **Problem:** The demographics of NYC's specialized highschools have becoming less and less diverse (perhaps, indicative of the fact that some populations do not get equal access to high quality education)
# 
# 

# ## Objectives
# **Primary Objective:** To improve school selection process with a more informed and granular approach for determining potential for outreach at any given school
# 
# **Secondary objective:** Assessment of student needs in taking the SHSAT so that minority and underserved students stand to gain most from services like after school programs, test preps, mentoring, or resources for parents

# ## Loading the Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
color = sns.color_palette()
from numpy import array
from matplotlib import cm
from scipy.misc import imread
import base64
from sklearn import preprocessing
#from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud, STOPWORDS
import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools


import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


df_d5 = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
df_exp = pd.read_csv('../input/2016 School Explorer.csv')


# ## About the Data

# The PASSNYC data contains two datasets with the following number of datapoints and features.

# In[ ]:


desc = pd.DataFrame({'Dataset':['Registrations and Testers','School Explorer'],
             'Datapoints':[df_d5.shape[0],df_exp.shape[0]],
             'Features':[df_d5.shape[1],df_exp.shape[1]]})
desc = desc[['Dataset','Datapoints','Features']]
desc


# In[ ]:


df_d5.head(3)


# In[ ]:


df_exp.head(3)


# ## Exploring the Data

# ### Registrations and Testers

# In[ ]:


df_d5.dtypes


# #### School Distribution
# We check how the schools via `School name` are distributed in PASSNYC's records

# In[ ]:


plt.figure(figsize=(14,10))
plt.barh(df_d5['School name'].value_counts().index[::-1], 
       df_d5['School name'].value_counts()[::-1],
       color=sns.color_palette('viridis'))
plt.xlabel('Counts')
plt.ylabel('Schools')
plt.title('School Distribution', size = 15)
plt.tight_layout()


# #### Year of SHST Distribution
# We see how `Year of SHST` is distributed. There is a steady increase of number of school on SHST per year starting from 33 in 2016 to 37 in 2016.

# In[ ]:


plt.figure(figsize=(10,6))

plt.bar(df_d5['Year of SHST'].value_counts().index, 
        df_d5['Year of SHST'].value_counts(),
        color=sns.color_palette('viridis'))
plt.xlabel('Year')
plt.xticks([2013,2014,2015,2016])
plt.ylabel('Counts')
plt.title('Year of SHST Distribution', size = 18)
plt.tight_layout()


# #### Grade Level Distribution
# We observe that 60.7% of schools on PASSNYC's records are on Grade 8 while 39.3% are on Grade 9. 

# In[ ]:


plt.figure(figsize=(4,4))
plt.pie(df_d5['Grade level'].value_counts(),radius=2,
       colors=sns.color_palette('viridis'),
       labeldistance = 1.1);

g8_pct = (df_d5['Grade level'].value_counts().values[0]/len(df_d5)).round(3)
g9_pct = (df_d5['Grade level'].value_counts().values[1]/len(df_d5)).round(3)
plt.text(-3.5,0, 'Grade Level', size = 18)
plt.text(-3.5,-0.2, 'Distribution', size = 18)
plt.text(-0.5,0.75,'Grade 8', color = 'white', size = 20)
plt.text(-0.5,0.5,str(g8_pct)+'%', color = 'white', size = 20)

plt.text(-0.2,-0.9,'Grade 9', color = 'white', size = 20)
plt.text(-0.2,-1.2,str(g9_pct)+'%', color = 'white', size = 20)
plt.text(-0.2,-0.9,'Grade 9', color = 'white', size = 20)
plt.tight_layout()


# #### Enrollment on 10/31 Descriptive Statistics
# We explore the stats of enrollment numbers specifically on 10/31.

# In[ ]:


pd.DataFrame(df_d5['Enrollment on 10/31'].describe())


# #### Enrollments on 10/31 by School Name
# We see which schools got the most number of enrollments on 10/31. Frederick Douglas Academy gets the most number of enrollments on 10/31 with 1640 recorded enrollments. This is followed by Democracy Prep Charter School with 943 enrollees and Democracy Prep Harlem School with 896 enrollees.

# In[ ]:


#df_d5.groupby('School name')['Enrollment on 10/31'].sum()

plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Enrollment on 10/31'].sum().index, 
       df_d5.groupby('School name')['Enrollment on 10/31'].sum(),
       color=sns.color_palette('viridis'))
plt.xlabel('Enrollments on 10/31')
plt.ylabel('School')
plt.title('Enrollments on 10/31 School Distribution', size = 15)
plt.tight_layout()


# #### Number of students who registered for the SHSAT Descriptive Statistics
# We explore the stats on the number of students registered for the SHSAT.

# In[ ]:


pd.DataFrame(df_d5['Number of students who registered for the SHSAT'].describe())


# #### Number of students who registered for the SHSAT by School
# We see which schools got the most number of registered students for the SHSAT.

# In[ ]:


plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Number of students who registered for the SHSAT'].sum().index, 
       df_d5.groupby('School name')['Number of students who registered for the SHSAT'].sum(),
       color=sns.color_palette('viridis'))
plt.xlabel('Number of students who registered for the SHSAT')
plt.ylabel('School')
plt.title('Number of students who registered for the SHSAT School Distribution', size = 15)
plt.tight_layout()


# #### Number of students who took the SHSAT
# We explore stats on the number of students who took the SHSAT.

# In[ ]:


pd.DataFrame(df_d5['Number of students who took the SHSAT'].describe())


# #### Number of students who took the SHSAT by School
# We see which school garnered the highest number of SHSAT takers.

# In[ ]:


plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Number of students who took the SHSAT'].sum().index, 
       df_d5.groupby('School name')['Number of students who took the SHSAT'].sum(),
       color=sns.color_palette('plasma'))
plt.xlabel('Number of students who took the SHSAT')
plt.ylabel('School')
plt.title('Number of students who took the SHSAT  School Distribution', size = 15)
plt.tight_layout()


# #### Difference between registered and takers Descriptive Statistics 
# We look into the difference between the number of registered students and the actual test takers.

# In[ ]:


df_d5['Diff between registered and takers'] = df_d5['Number of students who registered for the SHSAT'] - df_d5['Number of students who took the SHSAT']


# In[ ]:


pd.DataFrame(df_d5['Diff between registered and takers'].describe())


# #### Difference between registered and takers by School
# We look into the difference between the number of registered students and the actual test takers per school.

# In[ ]:


plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Diff between registered and takers'].sum().index, 
       df_d5.groupby('School name')['Diff between registered and takers'].sum(),
       color=sns.color_palette('plasma'))
plt.xlabel('Difference between registered and takers')
plt.ylabel('School')
plt.title('Difference between registered and takers', size = 15)
plt.tight_layout()


# ## Summary

# In[ ]:




