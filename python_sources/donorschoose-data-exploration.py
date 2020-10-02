#!/usr/bin/env python
# coding: utf-8

# <img src='https://blog.bonus.ly/hubfs/donorschoose-org-logo.png?t=1525381787204'>

# Founded in 2000 by a Bronx history teacher, DonorsChoose.org has raised $685 million for America's classrooms. Teachers at three-quarters of all the public schools in the U.S. have come to DonorsChoose.org to request what their students need, making DonorsChoose.org the leading platform for supporting public education.
# 
# To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.
# 
# In the second Kaggle Data Science for Good challenge, DonorsChoose.org, in partnership with Google.org, is inviting the community to help them pair up donors to the classroom requests that will most motivate them to make an additional gift. To support this challenge, DonorsChoose.org has supplied anonymized data on donor giving from the past five years. The winning methods will be implemented in DonorsChoose.org email marketing campaigns.

# <img src='http://stuffonix.com/wp-content/uploads/2017/09/donorschoose-how-it-work.jpg'>

# ### Problem Statement
# 
# DonorsChoose.org has funded over 1.1 million classroom requests through the support of 3 million donors, the majority of whom were making their first-ever donation to a public school. If DonorsChoose.org can motivate even a fraction of those donors to make another donation, that could have a huge impact on the number of classroom requests fulfilled.
# 
# A good solution will enable DonorsChoose.org to build targeted email campaigns recommending specific classroom requests to prior donors. Part of the challenge is to assess the needs of the organization, uncover insights from the data available, and build the right solution for this problem. Submissions will be evaluated on the following criteria:
# 
# - Performance - How well does the solution match donors to project requests to which they would be motivated to donate? DonorsChoose.org will not be able to live test every submission, so a strong entry will clearly articulate why it will be effective at motivating repeat donations.
# 
# 
# - Adaptable - The DonorsChoose.org team wants to put the winning submissions to work, quickly. Therefore a good entry will be easy to implement in production.
# 
# 
# - Intelligible - A good entry should be easily understood by the DonorsChoose.org team should it need to be updated in the future to accommodate a changing marketplace.

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np

# For offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Setting the plot size
plt.rcParams["figure.figsize"] = [12, 9]


# # Loading Data

# ### Load Donations file

# In[3]:


dtypes_donations = {'Project ID':'object','Donation ID':'object','Donor ID':'object',
                    'Donation Included Optional Donation':'category','Donation Amount':'float64','Donor Cart Sequence':'int64'}
donations = pd.read_csv('../input/Donations.csv',dtype=dtypes_donations, parse_dates=['Donation Received Date'])


# In[5]:


# Sneak peek into 3 random records of donations dataset
donations.sample(2)


# In[6]:


donations['Donation Amount'].describe().apply(lambda x: format(x, 'f'))


# #### Donation Amount statistical data:
# - Mean - 60
# - Median - 25
# - Minimum - 0.01
# - Maximum - 60000
# - Pop. Standard Deviation - 166

# In[7]:


donations['Donor Cart Sequence'].describe().apply(lambda x: format(x, 'f'))


# #### Donor Cart Sequence statistical data:
# - Mean - 143
# - Median - 2
# - Minimum - 1
# - Maximum - 18116
# - Pop. Standard Deviation - 872

# #### Top 10 Donors by the decreasing order of their Donor Cart Sequence

# In[8]:


d2 = donations[['Donor ID', 'Donor Cart Sequence']].groupby('Donor ID', as_index=False)['Donor Cart Sequence'].max().sort_values(by='Donor Cart Sequence', ascending=False).head(10)
d2


# ### Load Donors file

# In[9]:


dtypes_donors = {'Donor ID':'object','Donor City':'object','Donor State':'category','Donor Is Teacher':'category',
                 'Donor Zip':'object'}
donors = pd.read_csv('../input/Donors.csv', low_memory=False, dtype=dtypes_donors)


# In[10]:


# Sneak peek into 3 random records of donors dataset
donors.sample(2)


# ### Perform inner join of <code>donations</code> and <code>donors</code> dataframes on 'Donor ID'

# In[11]:


donations = pd.merge(donations, donors, on='Donor ID', how='inner')
donations = donations.reset_index(drop=True)


# In[13]:


# Sneak peek into 3 random records of the merged dataset
donations.sample(2)


# ### Load Projects file

# In[16]:


dtypes_projects = {'Project ID':'object','School ID':'object','Teacher ID':'object','Teacher Project Posted Sequence':'int64',
                   'Project Type':'category','Project Title':'object','Project Subject Category Tree':'object',
                   'Project Grade Level Category':'category','Project Resource Category':'category','Project Cost':'object',
                   'Project Posted Date':'object','Project Current Status':'category','Project Fully Funded Date':'object'}
projects = pd.read_csv('../input/Projects.csv', usecols=list(dtypes_projects.keys()), dtype=dtypes_projects,
                       parse_dates=['Project Posted Date','Project Fully Funded Date'], error_bad_lines=False, 
                       warn_bad_lines=False)


# In[17]:


projects.sample(2)


# #### Strip unnecessary characters of 'Project Cost' column & convert it to float datatype

# In[18]:


projects['Project Cost'] = projects['Project Cost'].str.replace('$', '')
projects['Project Cost'] = projects['Project Cost'].str.replace(',', '')
projects['Project Cost'] = projects['Project Cost'].astype('float64')


# #### Split multiple categories of 'Project Subject Category Tree' column and convert them to categorical/indicator variables
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html

# In[19]:


cat_df = pd.get_dummies(projects['Project Subject Category Tree'].str.split(',').apply(pd.Series))

cat_df['Applied Learning'] = (cat_df['0_Applied Learning'].add(cat_df['1_ Applied Learning'])).astype('category')
cat_df['Health & Sports'] = (cat_df['0_Health & Sports'].add(cat_df['1_ Health & Sports'])).astype('category')
cat_df['History & Civics'] = (cat_df['0_History & Civics'].add(cat_df['1_ History & Civics'])).astype('category')
cat_df['Literacy & Language'] = (cat_df['0_Literacy & Language'].add(cat_df['1_ Literacy & Language'])).astype('category')
cat_df['Math & Science'] = (cat_df['0_Math & Science'].add(cat_df['1_ Math & Science'])).astype('category')
cat_df['Music & The Arts'] = (cat_df['0_Music & The Arts'].add(cat_df['1_ Music & The Arts'])).astype('category')
cat_df['Special Needs'] = (cat_df['0_Special Needs'].add(cat_df['1_ Special Needs'])).astype('category')
cat_df['Warmth'] = (cat_df['0_Warmth'].add(cat_df['1_ Warmth'])).astype('category')
cat_df['Care & Hunger'] = (cat_df['1_ Care & Hunger'].add(cat_df['2_ Care & Hunger'])).astype('category')

cat_df.drop(cat_df.columns.tolist()[:18], axis=1, inplace=True)


# In[23]:


cat_df.sample(2)


# ### Concatenate columns of <code>project</code> and <code>cat_df</code> & drop the 'Project Subject Category Tree' column

# In[24]:


projects = pd.concat([projects, cat_df], axis=1)
projects.drop(['Project Subject Category Tree'], axis=1, inplace=True)


# In[25]:


projects.info()


# ### Perform inner join of <code>donations_projects</code> and <code>donations</code> dataframes on 'Project ID'

# In[26]:


donations_projects = pd.merge(donations, projects, on='Project ID', how='inner')
donations_projects = donations_projects.reset_index(drop=True)


# In[27]:


donations_projects.sample(2)


# ### Load Schools file

# In[28]:


dtypes_schools = {'School ID':'object','School Metro Type':'category','School Percentage Free Lunch':'float64',
                  'School State':'category','School Zip':'object','School City':'object'}
schools = pd.read_csv('../input/Schools.csv', usecols=list(dtypes_schools.keys()),
                      warn_bad_lines=False, error_bad_lines=False, dtype=dtypes_schools)


# In[29]:


schools.sample(2)


# ### Perform inner join of <code>donations_projects</code> and <code>schools</code> dataframes on 'School ID'

# In[30]:


donations_projects = pd.merge(donations_projects, schools, on='School ID', how='inner')
donations_projects = donations_projects.reset_index(drop=True)


# In[49]:


donations_projects.info()


# In[31]:


donations_projects.sample(2)


# In[52]:


df = donations_projects.groupby("Donor State")['Donation Amount'].agg('sum').nlargest(10).apply(lambda x: format(x, 'f')).astype('float64').apply(lambda x: x/1000000).round(1)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(range(0, 50, 5), fontsize=14)
plt.xlabel("Donor State", fontsize=16)  
plt.ylabel("Donation Amount (in million)", fontsize=16)
plt.title('Top 10 States by their Donation amounts (in million)', fontsize=18)
_ = df.plot('bar', ax=ax)


# In[50]:


df = donations_projects['Donor State'].value_counts().nlargest(10)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.xticks(fontsize=14)
plt.yticks(range(0, 800000, 100000), fontsize=14)
plt.xlabel("Donor State", fontsize=16)  
plt.ylabel("Number of Donations", fontsize=16)
plt.title('Top 10 States by their number of Donations', fontsize=18)
_ = df.plot('bar', ax=ax)


# In[37]:


df = donations_projects.iloc[:, 22:30].agg('sum').apply(lambda x: x/100000).sort_values(ascending=False)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.xticks(range(0, 28, 2), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Number of Donations (in 100k)", fontsize=15)  
#plt.ylabel("Project Subject Category", fontsize=15)
plt.title('Number of Donations by Project Subject Category', fontsize=17)
_ = df.plot('barh', ax=ax)


# In[ ]:


df = donations_projects['Project Resource Category'].value_counts().apply(lambda x: x/100000)
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.xticks(range(0, 19, 1), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Number of Donations (in 100k)", fontsize=16)  
#plt.ylabel("Project Resource Category", fontsize=16)
plt.title('Number of Donations by Project Resource Category', fontsize=18)
_ = df.plot('barh', ax=ax)


# In[42]:


fig = {
    "data": [
        {
            "values": donors['Donor Is Teacher'].value_counts().tolist(),
            "labels": ['Donor Is Not Teacher', 'Donor Is Teacher'],
            "type": "pie",
            "textfont": {
                "size": 16
            }
        }],
    "layout": {
        "title": "What % of Donors are Teachers?",
        "font": {
            "size": 14
        }
    }
}
iplot(fig)


# In[43]:


fig = {
    "data": [
        {
            "values": donations['Donation Included Optional Donation'].value_counts(),
            "labels": ['Included Optional Donation', 'Excluded Optional Donation'],
            "type": "pie",
            "textfont": {
                "size": 16
            }
        }],
    "layout": {
        "title": "What % of Donations included optional donation?",
        "font": {
            "size": 14
        }
    }
}
iplot(fig)


# In[45]:


fig = {
    "data": [
        {
            "values": projects['Project Type'].value_counts(),
            "labels": ['Teacher-Led', 'Professional Development', 'Student-Led'],
            "type": "pie",
            "textfont": {
                "size": 16
            }
        }],
    "layout": {
        "title": "Proportion of Projects by Project Type",
        "font": {
            "size": 14
        }
    }
}
iplot(fig)


# In[46]:


fig = {
    "data": [
        {
            "values": donations_projects['Project Grade Level Category'].value_counts().tolist(),
            "labels": donations_projects['Project Grade Level Category'].value_counts().axes[0],
            "type": "pie",
            "textfont": {
                "size": 16
            }
        }],
    "layout": {
        "title": "Proportion of Donations by Project Grades",
        "font": {
            "size": 14
        }
    }
}
iplot(fig)

