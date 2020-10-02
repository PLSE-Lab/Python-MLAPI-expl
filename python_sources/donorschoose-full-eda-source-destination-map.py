#!/usr/bin/env python
# coding: utf-8

# # Preface
# <p>I am totally new to this field. Let see how far I can go before May 30th! <br/> Please suggest me, correct me, walk next to me or upvote this kernel if you feel like. <br/>This will be an on going kernel, **stay tuned!** : ) </p>
# 

# # Credits & Inspirational Kernels
# Below links and kernels are helping and inspiring me understanding and learning:
# * https://www.kaggle.com/kanncaa1/rare-visualization-tools

# # Table of Contents
# 1. [Introduction](http://)
# 2. [Import Libraries](http://)
# 3. Directory List
# 4. Read Data
# 5. Overview & Insight of Data<br/>
#      5.1.a Project Data Overview<br/>
#      5.1.b Project Data Insight
#      

# # Introduction
# <p>DonorChoose.org is an online platform, started by a Bronx history teacher in 2000, where classrooms in America has the opportunity to raise money. DonorsChoose.org has raised $685 million for America's classrooms. To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.</p>

# # Import libraries 

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
import cufflinks as cf
cf.go_offline()
from sklearn import preprocessing
import missingno as msno # to view missing values
import os


# # Directory List

# In[ ]:


print(os.listdir("../input"))


# # Read Data

# In[3]:


teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv')
resources = pd.read_csv('../input/Resources.csv')


# # Overview & Insight of Data

# ### Projects Data Overview

# In[5]:


projects.head()


# ### Project Data Insight

# In[7]:


projects.info()


# ### Missing Values in Project Data

# In[9]:


print(projects.isnull().sum())
msno.matrix(projects)
plt.show()


# ### Missing Value Percentage in Project Data

# In[30]:


# how many total missing values do we have?
missing_values_count = projects.isnull().sum()
total_cells = np.product(projects.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('% of Missing Values in Projects Data:')
print((total_missing/total_cells) * 100, "%")


# So, in the Projects table, only around 2.14% data are missing.

# ##### Thoughts on Projects Data: 
# DonorsChoose.org wants to build targeted email campaigns recommending specific classroom requests to prior donors. In the above Projects table, we can find which cateogires and sub categories are getting the most funding.  And we will have to match them to the donors as well. 

# ### Donation Data

# In[6]:


donations.head()


# ##### Thoughts on Donations Data:
# We can find which projects are getting the most donations. 

# ### Donor Data

# In[10]:


donors.head()


# ### Teachers Data

# In[11]:


teachers.head()


# ### Schools Data

# In[12]:


schools.head()


# ### Resources Data

# In[13]:


resources.head()


# ## Distribution of Project Subject Category

# In[14]:


project_subject_category = projects['Project Subject Category Tree'].value_counts().head(10)
project_subject_category.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project Subject Categories')


# ## Distribution of Project Subject Sub Category

# In[15]:


project_subject_sub_category = projects['Project Subject Subcategory Tree'].value_counts().head(10)
project_subject_sub_category.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project Subject Sub-Categories')


# ## Distribution of Donor Cities

# In[16]:


donor_cities = donors['Donor City'].value_counts().head(10)
donor_cities.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Donor Cities')


# ## Distribution of Donor State

# In[17]:


donor_states = donors['Donor State'].value_counts().head(10)
donor_states.iplot(kind='bar', xTitle = 'Donor State', yTitle = "Count", title = 'Donor States')


# ## Distribution of School State

# In[18]:


school_states = schools['School State'].value_counts().head(50)
school_states.iplot(kind='bar', xTitle = 'School State', yTitle = "Count", title = 'School States')


# ## View Project Current Status Colum Insight

# In[20]:


projects['Project Current Status'].describe()


# In[21]:


projects['Project Current Status'].unique()


# ## Fully Funded Projects

# In[22]:


#fully_funded_projects.describe()
#projects[projects['Project Current Status'] == "Fully Funded"].head()
projects[projects['Project Current Status'] == "Fully Funded"].head(10)


# ## Project Title Insight

# In[23]:


projects['Project Title'].describe()


# ## Distribution of Project Titles

# In[24]:


project_titles = projects['Project Title'].value_counts().head(50)
project_titles.iplot(kind='bar', xTitle = 'Project Titile', yTitle = "Count", title = 'Project Title')


# ## Project Cost Insight

# In[27]:


#How to skip currency symbol and convert to numeric type
#df1['Avg_Annual'] = df1['Avg_Annual'].str.replace(',', '')
#df1['Avg_Annual'] = df1['Avg_Annual'].str.replace('$', '')
#df1['Avg_Annual'] = df1['Avg_Annual'].convert_objects(convert_numeric=True)

projects['Project Cost'] = projects['Project Cost'].str.replace(',', '')
projects['Project Cost'] = projects['Project Cost'].str.replace('$', '')
projects['Project Cost'] = projects['Project Cost'].convert_objects(convert_numeric=True)

print('Describe Project Cost: ')
print(projects['Project Cost'].describe())
print('View Some Project Cost data')
print(projects['Project Cost'].head())
print('Minimum Project Cost: ', projects['Project Cost'].min())
print('Maximum Project Cost: ', projects['Project Cost'].max())
print('Median Project Cost: ', projects['Project Cost'].median())
print('Total Project Cost: ', projects['Project Cost'].sum())
fully_funded_projects = projects[projects['Project Current Status'] == "Fully Funded"]
print('Total Funded Project Cost: ', fully_funded_projects['Project Cost'].sum())

