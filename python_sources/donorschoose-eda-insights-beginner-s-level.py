#!/usr/bin/env python
# coding: utf-8

#  # DonorsChoose : Exploratory Data Analysis & Insights - Beginner's Level
#  
#  ## About DonorsChoose
# Founded in 2000 by a Bronx history teacher, DonorsChoose.org has raised $685 million for America's classrooms. Teachers at three-quarters of all the public schools in the U.S. have come to DonorsChoose.org to request what their students need, making DonorsChoose.org the leading platform for supporting public education.
#  
# To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.
# 
#  ## Motivation for this Notebook
# This is my first attempt at a detailed notebook on EDA steps for any data science problem so constructive criticism is welcome. I will also try to give detailed explanation of the various steps that I will perform in the notebook for helping anyone who is at a beginner stage just like me. 

# ## 1. Preparing the Data
# ### 1.1. Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")


# ### 1.2. Importing Datasets

# In[ ]:


df_donors = pd.read_csv('../input/Donors.csv')
df_donations = pd.read_csv('../input/Donations.csv')
df_teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
df_projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False)
df_resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
df_schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)

# SMALL DATAFRAMES
df_donors_sm = df_donors.iloc[1:200,]


# In the above code I have also generated some small dataframes (df_donors_sm). There are only for my usage so that when I'm developing the code for graphs, I can quickly check with a small dataset.

# ## 2.Data Preview & Statistical Description

# ### 2.1 Donors Data

# #### 2.1.1. Preview

# In[ ]:


df_donors.head()


# #### Insight
# - "Donor Is Teachers" can be a binary variable.
# - There might be a link between "Donor Data"  and "Teacher Data". **(Intuition)**
# - There might be a link between "Donor Data" and "Donations Data". **(Intuition)**

# #### 2.1.2 Statistical Description

# In[ ]:


df_donors.describe()


# #### Insight
# - Most stats are straight forward and visible.
# - "Donor ID" is a key element as it is unique (freq=1).
# - Chicago is the top most "Donor City".
# - California is the top most "Donor State".
# - From the above description we see that total donor count is 2122640. However, in "Donor City" and "Donor Zip" columns we can see that the count is 1909543 and 1942580 respetively. Hence, we have some **missing data in "Donors data"**.

# ### 2.2 Donations Data

# #### 2.2.1 Preview

# In[ ]:


df_donations.head()


# #### Insight
# - Donor Data and Donations Data can be merged over "Donor ID".
# - Donation Included Optional Donation can be a binary variable.

#  #### 2.2.2 Statistical Description

# In[ ]:


df_donations.describe()


# #### Insight
# - Minimum "Donation Amount" is \$0.01.
# - Maximum "Donation Amount" is \$60000.
# - Average "Donation Amount" is \$60.66.

# ### 2.3 Teachers Data

# #### 2.3.1 Preview

# In[ ]:


df_teachers.head()


# #### Insight
# - Since we have "Teacher Prefix", we can explore the donations of teachers according to their sex by engineering a new binary feature from these prefixes.

# #### 2.3.2 Statistical Description

# In[ ]:


df_teachers.describe()


# #### Insight
# - There are 6 different "Teacher Prefixes" available.
# - "Teacher ID" is a key element as it is unique (freq=1).

# ### 2.4 Projects Data

# #### 2.4.1 Preview

# In[ ]:


df_projects.head()


# #### Insight
# - We can observe that School Data and Teacher Data can be merged with Projects Data over "School ID" and "Teacher ID" respectively.
# - We can explore the different kind of projects that take are taken up, we might be able to perform some word frequency analysis.

# #### 2.4.2 Statistical Description

# In[ ]:


print("Total Rows of Data :",len(df_projects))
print(df_projects["Project Type"].describe())
print(df_projects["Project Subject Category Tree"].describe())
print(df_projects["Project Grade Level Category"].describe())
print(df_projects["Project Resource Category"].describe())
print(df_projects["Project Current Status"].describe())
print(df_projects.describe())


# #### Insight
# - There is a lot of **missing data**.
# - There are 3 unique "Project Type".
# - There are 51 unique "Project Subject Category".
# - There are 17 unique "Project Resource Category".
# - There are 5 unique "Project Grade Level Category".
# - There 3 unique "Project Current Status".
# - Minimum "Project Cost" is \$35.29
# - Maximum "Project Cost" is \$60000.00
# - Average "Project Cost" is \$255737.70

# ### 2.5 Resources Data

# #### 2.5.1 Preview

# In[ ]:


df_resources.head()


# #### Insight
# - We can observe that Resource Data can be merged with Projects Data over Project ID.

# #### 2.5.2 Statistical Description

# In[ ]:


print(df_resources["Resource Vendor Name"].describe())
print(df_resources.describe())


# #### Insight
# - Different statistics about "Resource Quantity" & "Resource Unit Price" can be observed.
# - There are 31 unique Resource Vendor Name.

# ### 2.6 Schools Data

# #### 2.6.1 Preview

# In[ ]:


df_schools.head()


# #### Insight
# - We can observe that Schools Data has "School State" & "School Zip" and Donors also had "Donor State" & "Donor Zip". We might be able to further explore the data according to the states and zip.

# #### 2.6.2 Statistical Description

# In[ ]:


print(df_schools["School Metro Type"].describe())
print(df_schools["School State"].describe())
print(df_schools.describe())


# #### Insight
# - There are 5 unique "School Metro Type".
# - Since we see 51 unique "School State", there is a need to further clean this data.
# - Different statistics about "School Percentage Free Lunch"can be observed.

# ## 3. Creating Merged Tables (after above insights)

# ### 3.1 Donations & Donor Merge

# In[ ]:


df_donations_donors = df_donations.merge(df_donors, on='Donor ID', how='inner')
df_donations_donors.head()


# ## 4. Donors Data Exploration

# ### 4.1 State wise Donor Distribution 

# In[ ]:



sns.countplot(y='Donor State', data=df_donors_sm, color='c', order=pd.value_counts(df_donors['Donor State']).iloc[:10].index);


# - Above is a plot for top 10 states with highest number of Donors.

# 
