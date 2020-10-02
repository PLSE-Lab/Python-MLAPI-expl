#!/usr/bin/env python
# coding: utf-8

# # Preface
# <p>I am totally new to this field. Let see how far I can go before May 30th! <br/> Please suggest me, correct me, walk next to me or upvote this kernel if you feel like. <br/>This will be an on going kernel, **stay tuned!** : ) </p>
# 

# # Credits & Inspirational Kernels
# Apart from Kaggle's Learn sections, below are the links and kernels that are helping and inspiring me understanding and learning:
# * [Rare Visualization Tools](https://www.kaggle.com/kanncaa1/rare-visualization-tools)
# * [Handling-Missing Values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)

# # Table of Contents
# 1. [Introduction](http://)
# 2. [Import Libraries](http://)
# 3. [Directory List](http://)
# 4. [Read Data](http://)
# 5. [Projects Data - Overview & Insight](http://)<br/>
#      5.1. [Projects Data Overview](http://)<br/>
#      5.2 [Projects Table Info](http://)<br/>
#      5.3 [Missing Data in Projects Table](http://)<br/>
#      5.4 [Percentage of Missing Data in Projects Table](http://)<br/>
# 6. [Donations Data - Overview & Insight](http://)<br/>
#      6.1 [Donations Data Overview](http://)<br/>
#      6.2 [Donations Table Info](http://)<br/>
#      6.3 [Missing Data in Donations Table](http://)<br/>
# 7. [Donors Data - Overview & Insight](http://)<br/>
#      7.1 [Donors Data Overview](http://)<br/>
#      7.2 [Donors Table Info](http://)<br/>
#      7.3 [Missing Data in Donations Table](http://)<br/>
#      7.4 [Percentage of Missing Data in Donations Table](http://)<br/>
# 8. [Teachers Data - Overview & Insight](http://)<br/>
#      8.1 [Teachers Data Overview](http://)<br/>
#      8.2 [Teachers Table Info](http://)<br/>
#      8.3 [Missing Data in Teachers Table](http://)<br/>
#      8.4 [Percentage of Missing Data in Teachers Table](http://)<br/>
# 9. [Schools Data - Overview & Insight](http://)<br/>
#      9.1 [Schools Data Overview](http://)<br/>
#      9.2 [Schools Table Info](http://)<br/>
#      9.3 [Missing Data in Schools Table](http://)<br/>
#      9.4 [Percentage of Missing Data in Schools Table](http://)<br/>
# 10. [Resources Data - Overview & Insight](http://)<br/>
#      10.1 [Resources Data Overview](http://)<br/>
#      10.2 [Resources Table Info](http://)<br/>
#      10.3 [Missing Data in Resources Table](http://)<br/>
#      10.4 [Percentage of Missing Data in Resources Table](http://)<br/>

# # 1. Introduction
# <p>DonorChoose.org is an online platform, started by a Bronx history teacher in 2000, where classrooms in America has the opportunity to raise money. DonorsChoose.org has raised $685 million for America's classrooms. To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.</p>

# # 2. Import libraries 

# In[ ]:


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


# # 3. Directory List

# In[ ]:


print(os.listdir("../input"))


# # 4. Read Data

# In[ ]:


teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv')
resources = pd.read_csv('../input/Resources.csv')


# # 5. Projects Data - Overview & Insights

# ## 5.1 Projects Data Overview

# In[ ]:


projects.head()


# ## 5.2 Projects Table Info

# In[ ]:


projects.info()


# ## 5.3 Missing Data in Projects Table

# In[ ]:


print(projects.isnull().sum())
msno.matrix(projects)
plt.show()


# ## 5.4 Percentage of Missing Data in Projects Table

# In[ ]:


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

# # 6. Donation Data - Overview & Insights

# ## 6.1 Donations Data Overview

# In[ ]:


donations.head()


# ## 6.2 Donations Table Info

# In[ ]:


donations.info()


# ## 6.3 Missing Data in Donations Table

# In[ ]:


print(donations.isnull().sum())


# <p>There is no missing values in donations table. </p>

# ##### Thoughts on Donations Data:
# We can find which projects are getting the most donations. 

# # 7. Donor Data - Overview & Insights

# ## 7.1 Donors Data Overview

# In[ ]:


donors.head()


# ## 7.2 Donors Table Info

# In[ ]:


donors.info()
#donors.describe()


# ## 7.3 Missing Data in Donors Table

# In[ ]:


print('Missing Data Overview in Donors Table')
print(donors.isnull().sum())
msno.matrix(donors)
plt.title('Missing Data in Donors Table')
plt.show()


# ## 7.4 Percentage of Missing Data in Donors Table

# In[ ]:


# how many total missing values do we have?
missing_values_count = donors.isnull().sum()
total_cells = np.product(donors.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('% of Missing Values in Donors Table:')
print((total_missing/total_cells) * 100, "%")


# # 8. Teachers Data - Overview & Insights

# ## 8.1 Teachers Data Overview

# In[ ]:


teachers.head()


# ## 8.2 Teachers Table Info

# In[ ]:


teachers.info()


# ## 8.3 Missing Data in Teachers Table

# In[ ]:


print('Missing Data Overview in Teachers Table')
print(teachers.isnull().sum())
msno.matrix(teachers)
plt.title('Missing Data in Teachers Table')
plt.show()


# # 9. Schools Data - Overview & Insights

# ## 9.1 Schools Data Overview

# In[ ]:


schools.head()


# ## 9.2 Schools Table Info

# In[ ]:


schools.info()


# ## 9.3 Missing Data in Schools Table

# In[ ]:


print('Missing Data Overview in Schools Table')
print(schools.isnull().sum())
msno.matrix(schools)
plt.title('Missing Data in Schools Table')
plt.show()


# # 10. Resources Data - Overview & Insights

# ## 10.1 Resources Data Overview

# In[ ]:


resources.head()


# ## 10.2 Resources Data Info

# In[ ]:


resources.info()


# ## 10.3 Missing Data in Resources Table

# In[ ]:


print('Missing Data Overview in Resources Table')
print(resources.isnull().sum())
msno.matrix(resources)
plt.title('Missing Data in Resources Table')
plt.show()


# ## Distribution of Project Subject Category

# In[ ]:


project_subject_category = projects['Project Subject Category Tree'].value_counts().head(10)
project_subject_category.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project Subject Categories')


# ## Distribution of Project Subject Sub Category

# In[ ]:


project_subject_sub_category = projects['Project Subject Subcategory Tree'].value_counts().head(10)
project_subject_sub_category.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project Subject Sub-Categories')


# ## Distribution of Donor Cities

# In[ ]:


donor_cities = donors['Donor City'].value_counts().head(10)
donor_cities.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Donor Cities')


# ## Distribution of Donor State

# In[ ]:


donor_states = donors['Donor State'].value_counts().head(10)
donor_states.iplot(kind='bar', xTitle = 'Donor State', yTitle = "Count", title = 'Donor States')


# ## Distribution of School State

# In[ ]:


school_states = schools['School State'].value_counts().head(50)
school_states.iplot(kind='bar', xTitle = 'School State', yTitle = "Count", title = 'School States')


# ## View Project Current Status Colum Insight

# In[ ]:


projects['Project Current Status'].describe()


# In[ ]:


projects['Project Current Status'].unique()


# ## Fully Funded Projects

# In[ ]:


#fully_funded_projects.describe()
#projects[projects['Project Current Status'] == "Fully Funded"].head()
projects[projects['Project Current Status'] == "Fully Funded"].head(10)


# ## Project Title Insight

# In[ ]:


projects['Project Title'].describe()


# ## Distribution of Project Titles

# In[ ]:


project_titles = projects['Project Title'].value_counts().head(50)
project_titles.iplot(kind='bar', xTitle = 'Project Titile', yTitle = "Count", title = 'Project Title')


# ## Project Cost Insight

# In[ ]:


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

