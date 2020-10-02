#!/usr/bin/env python
# coding: utf-8

# <h1>**Analysis Of Trends**</h1>
# Work in progress! Do help me by providing suggestions and corrections.<br>
# -**Sushanth Ikshwaku** 
# 
# 

# <h2>1) Load Libraries</h2>

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from multiprocessing import Pool

get_ipython().run_line_magic('matplotlib', 'inline')

import os
# Any results you write to the current directory are saved as output.


# <h2>2)Read the data</h2>

# In[3]:


donations = pd.read_csv('../input/Donations.csv',index_col="Donation ID")
donors = pd.read_csv('../input/Donors.csv',low_memory= False, index_col= "Donor ID")
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines = False, index_col="School ID")
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False, warn_bad_lines = False, index_col="Teacher ID")
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"], index_col="Project ID")
resources = pd.read_csv('../input/Resources.csv', low_memory= False, error_bad_lines=False, warn_bad_lines=False, index_col = "Project ID")


# *** Schools:**

# In[4]:


schools.head()


# *** Projects:**

# In[5]:


projects.head()


# *** Resources:**

# In[6]:


resources.head()


# *** Teachers:**

# In[7]:


teachers.head()


# *** Donations:**

# In[8]:


donations.head()


# *** Donors:**

# In[9]:


donors.head()


# <h2>Graph 1: Number of Schools in each State</h2>

# In[10]:


plt.rcParams["figure.figsize"] = [20,6]
schools.groupby("School State")["School State"].count().plot.bar()


# <h3>Table with Donor State and School State</h3>

# In[12]:


donationSchoolDonor = pd.merge(donations[["Project ID","Donor ID"]], donors[["Donor State"]], left_on="Donor ID", right_index=True)
donationSchoolDonor = pd.merge(donationSchoolDonor, projects[["School ID"]], left_on="Project ID", right_index=True)
donationSchoolDonor = pd.merge(donationSchoolDonor, schools[["School State"]], left_on="School ID", right_index=True)
donationSchoolDonor.head()


# In[23]:


sameStateDonor = donationSchoolDonor.groupby("School State").agg(lambda x: len(x[(x['School State'] == x["Donor State"])]) / len(x))["Donor ID"]
diffStateDonor = donationSchoolDonor.groupby("School State").agg(lambda x: len(x[(x['School State'] != x["Donor State"])]) / len(x))["Donor ID"]


# <h2>Graph 2: Number of Schools in each State and Percent of Donors from same State</h2>

# In[49]:


red_patch = mpatches.Patch(color='red', label='Same State')
blue_patch = mpatches.Patch(color='blue', label='Number of Schools in a State')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis
ax1.plot(sameStateDonor, color="red")
ax2.plot(schools.groupby("School State")["School Name"].count())
ax1.tick_params(axis='x',rotation=90)
plt.legend(handles=[red_patch,blue_patch])


# <h2>Graph 2: Number of Schools in each State and Percent of Donors from Different State</h2>

# In[50]:


red_patch = mpatches.Patch(color='red', label='diff State Donor')
blue_patch = mpatches.Patch(color='blue', label='Number of Schools in a State')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis
ax1.plot(diffStateDonor, color="red")
ax2.plot(schools.groupby("School State")["School Name"].count())
ax1.tick_params(axis='x',rotation=90)
plt.legend(handles=[red_patch,blue_patch])


# <h2>Deduction 1:</h2>
# States with less number of schools (also maybe with less population) have a high percentage of donors who are Out-State 
