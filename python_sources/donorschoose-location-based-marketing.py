#!/usr/bin/env python
# coding: utf-8

# ![DonorsChoose.org](https://cdn.donorschoose.net/images/logo/dc-logo.png)
# # Overview
# Based on reviewing the data, I propose two possible campaigns for existing donors:
# 1. Notifying donors of another project for the teacher that they previously funded.
# 2. Notifying donors that another teacher in their state needs help.
# 
# Keep reading below to see why I came to that conclusion. Most of the cells are hidden in the published view to help readability but edit or fork the notebook to see all the analysis.
# 
# I would appreciate feedback or questions! Thanks.
# 

# In[ ]:


import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
#from wordcloud import WordCloud, STOPWORDS
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams["figure.figsize"] = [16, 12]
#import chardet
from subprocess import check_output
data_directory = '../input/'

#print(check_output(["ls", "../input/"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", data_directory]).decode("utf8").strip().split('\r\n')
# helpful character encoding module
#filenames = filenames.split('\n')
#dfs = dict()
#for f in  filenames:
#    dfs[f[:-4]] = pd.read_csv(data_directory+ f)


# In[ ]:


for filename in filenames:
    print(filename)


# ## Data Issues
# The Project dataset contains issues involving the Project Essay field not having a matching double quotes " character at the end of the field.
# 
# This has a couple of implications:
# * We may want to focus only on the ones that were funded to see which key features are applicable
# * We may not care about all the errored rows as long as we get the majority of the rows for projects that actually get donations
# 
# For now I am ignoring both error lines and warning lines

# In[ ]:


donations = pd.read_csv(data_directory + 'Donations.csv')
donors = pd.read_csv(data_directory + 'Donors.csv')
projects = pd.read_csv(data_directory + 'Projects.csv', error_bad_lines=False, warn_bad_lines=False)
#Issue with project as of 5/7/2018 - ParserError: Error tokenizing data. C error: Expected 15 fields in line 10, saw 18
#May have some strange characters within the free text.
#Does not affect most of the rows
resources = pd.read_csv(data_directory + 'Resources.csv', error_bad_lines=False, warn_bad_lines=False)
#Error in rsources - ParserError: Error tokenizing data. C error: Expected 5 fields in line 1172, saw 8
schools = pd.read_csv(data_directory + 'Schools.csv', error_bad_lines=False, warn_bad_lines=False)
#ParserError: Error tokenizing data. C error: Expected 9 fields in line 59988, saw 10
#teachers = pd.read_csv(data_directory + 'Teachers.csv')


# ## High-level Stats of Uncleansed Data
# * 1.3 Million Projects
# * 400,000 Teachers
# * 70,000 Schools
# * 4.8 Million Donations
# * 2.1 Million Donors
# 

# ## Cleansing Project Data
# Ensuring that at least the keys of the data are proper length (32) as this table is a cross-reference of three main entities - Projects Donors and Teachers

# In[ ]:


plt.hist(projects['Project ID'].map(str).apply(len))


# In[ ]:


projects_cleansed = projects.loc[(projects['Project ID'].map(str).apply(len) == 32)
                                & (projects['School ID'].map(str).apply(len) == 32)
                                & (projects['Teacher ID'].map(str).apply(len) == 32)
                                ]
#projects_cleansed.count


# In[ ]:


teacher_projects = pd.DataFrame(projects_cleansed[['Teacher ID','Project ID']].groupby(['Teacher ID']).count())


# In[ ]:


teacher_projects = teacher_projects.rename(columns={'Project ID':'project_count'})
teacher_projects[['project_count']] = teacher_projects[['project_count']].apply(pd.to_numeric)


# In[ ]:


frq, edges = np.histogram(teacher_projects['project_count'],bins = 250)

fig, ax = plt.subplots()
ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
plt.xlabel('Number of Projects')
plt.ylabel('Number of Teachers')
plt.title('Do teachers have more than one project?')
plt.show()


# Overwhelming majority of teachers have very few projects. Let's zoom in to see how many have just one project. This will determine if we need to notify the donor that a new project from the teacher they sponsor is available.

# In[ ]:


frq, edges = np.histogram(teacher_projects.query('project_count <= 5'),bins = 5)

fig, ax = plt.subplots()
ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
plt.xticks(range(1, 6))
plt.xlabel('Number of Projects')
plt.ylabel('Number of Teachers')
plt.title('Zoomed teachers have more than one project?')
plt.show()


# Yes, so there may be potential for a campaign to notify of other new or existing projects that the specific teacher posts. This will build longer term relationships with the teacher.

# In[ ]:


donors.sample


# In[ ]:


schools.sample


# In[ ]:


donations.sample


# ## Relationships
# Need to understand the relationships between donors and teachers.

# In[ ]:


donor_teacher = donations.merge(projects_cleansed,left_on='Project ID',right_on='Project ID', how='inner')


# In[ ]:


donor_teacher.sample


# In[ ]:


donors_same_teacher = pd.DataFrame(donor_teacher[['Teacher ID','Donor ID','Project ID']].groupby(['Teacher ID','Donor ID']).count())
donors_same_teacher = donors_same_teacher.rename(columns={'Project ID':'donor_teacher_count'})
donors_same_teacher[['donor_teacher_count']] = donors_same_teacher[['donor_teacher_count']].apply(pd.to_numeric)
#donors_same_teacher.sample


# In[1]:


frq, edges = np.histogram(donors_same_teacher.query('donor_teacher_count <= 5'),bins = 5)

fig, ax = plt.subplots()
ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
plt.xticks(range(1, 6))
plt.xlabel('Donations to Same Teacher')
plt.ylabel('Donations')
plt.title('Do donors sponsor the same teacher twice?')
plt.show()


# ## Answer - Not often but sometimes. Worth pursuing?

# In[ ]:


donations_teacher_donor = donor_teacher.merge(donors,left_on='Donor ID',right_on='Donor ID', how='inner')


# In[ ]:


#donations_teacher_donor.sample


# In[ ]:


donations_teacher_donor_school = donations_teacher_donor.merge(schools,left_on='School ID',right_on='School ID', how='inner')


# In[ ]:


#donations_teacher_donor_school.sample


# In[ ]:


donations_teacher_donor_school = donations_teacher_donor_school.rename(columns={'Donor State':'donor_state'})
donations_teacher_donor_school = donations_teacher_donor_school.rename(columns={'School State':'school_state'})


# In[ ]:


#donations_teacher_donor_school[donations_teacher_donor_school.donor_state == donations_teacher_donor_school.school_state].sample


# ## Location Based Donating!
# Based on the join, 2.9 Million out of 4.4 Million donations (65%) occur where the donor resides in the same state as the school. This seems correlated. Let's use a pivot table and a heat map to see more.

# In[ ]:


#def same_state (row):
#    if [donations_teacher_donor_school.donor_state == donations_teacher_donor_school.school_state]:
#        return 1
#    return 0


# In[ ]:


#donations_teacher_donor_school['same_state'] = donations_teacher_donor_school.apply (lambda row: same_state (row), axis=1)#


# In[ ]:


states_grouped = donations_teacher_donor_school.groupby(['donor_state','school_state'],as_index=False)['Donor ID'].count()


# In[ ]:


states_grouped.sample


# In[ ]:


pivoted_states = pd.pivot_table(states_grouped, index='donor_state', columns='school_state', values='Donor ID')


# In[ ]:


pivoted_states.sample


# In[ ]:


fig, ax = plt.subplots(figsize=(15,15))
#mask = np.zeros_like(pivoted_states)
sns.heatmap(pivoted_states, cbar_kws={'label': 'Number of Donations'}, cmap="Greens", linewidth=.02)
ax.set_title('Donations from Donor States and School States')
plt.show()


# # Thank you for reading!
# Comments? Questions? Suggestions? 
# Type below.

# In[ ]:


location_outliers = pivoted_states.loc[:, ['North Dakota']]


# In[ ]:


location_outliers.sample


# In[ ]:


#This takes too long to run!!!
#import seaborn as sns 
#correlation = donations_state['donor_state'].corr(donations_state['school_state'])
#sns.heatmap(correlation)


# In[ ]:


#plt.matshow(donations_state.corr())


# In[ ]:


#import plotly.plotly as py
from mpl_toolkits.basemap import Basemap


# In[ ]:


map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
resolution = 'i', area_thresh = 0.1,
llcrnrlon=-136.25, llcrnrlat=56.0,
urcrnrlon=-134.25, urcrnrlat=58)


map.drawcoastlines()
map.drawcountries()
map.drawmapboundary()

lon = -135.3318
lat = 57.0799
x,y = map(lon, lat)
x2, y2 = map(lon+0.5,lat+0.5)

plt.arrow(x,y,x2-x,y2-y,fc="k", ec="k", linewidth = 4, head_width=10, head_length=10)
plt.show()


# In[ ]:




