#!/usr/bin/env python
# coding: utf-8

# ## Which Projects attract Donors?
# 
# **A little description about DonosrChoose.org**<br>
# DonorsChoose.org has funded over 1.1 million classroom requests through the support of 3 million donors, the majority of whom were making their first-ever donation to a public school. If DonorsChoose.org can motivate even a fraction of those donors to make another donation, that could have a huge impact on the number of classroom requests fulfilled.<br>
# 
# **Brief description of notebook**<br>
# In order to analyse the data provided by DonorsChose.org, this notebook has been divided into following sections:<br>
# 
# 1. Import libraries<br>
# 2. Loading and Reading files<br>
# 3. Finding missing data<br>
# 4. Donation by people<br>
#     4.1 Total Donation<br>
#     4.2 What is the Total Amount donated by people? How much donated by teachers?<br>
#     4.3 Amount Donated by different states<br>
#     4.4 How many Donations include optional Donation?<br>
#     4.5 After how many days first donation for a project has been received by different states?<br>
# 5. Get Some info about projects<br>
#     5.1 What is the status of Current Projects?<br>
#     5.2 What are the Project Categories/sub-categories/Grade level category where schools need funds? What are their Resources?<br>
#     5.3 What is the type of project?<br>
#     5.4 Who are providing the above Resources?<br>
# 6. Description of Schools<br>
#     6.1 Where these schools are located (Rural/ Urban)?<br>
#     6.2 Which state's schools provide free lunch and to what proportion?<br>
# 7. Projects First posted by Teachers?<br>
# 8. Summary
# 
# 
# 
# ### More to come!! Stay Tunned!!
# ### If this is helpful, Please Upvote
# **This will motivate me to write more**
# 

# ## 1. Import Libraries

# In[1]:


import pandas as pd 
import numpy as np
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
from numpy import array
from matplotlib import cm
import missingno as msno
from wordcloud import WordCloud, STOPWORDS

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


# ## 2. Reading Files

# In[3]:


donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,                       parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False,                        warn_bad_lines=False)


# In[4]:


# Merging projects and donations dataframe
project_donations = pd.merge(projects, donations, on='Project ID')


# ## 3. Finding missing data
# White lines in the diagram shows the missing data.

# In[ ]:


print('Missing values in Projects DataFrame')
msno.matrix(projects)


# In[ ]:


print('Missing values in Donors DataFrame')
msno.matrix(donors)


# In[ ]:


print('Missing values in Resources DataFrame')
msno.matrix(resources)


# In[ ]:


print('Missing values in Schools DataFrame')
msno.matrix(schools)


# In[ ]:


print('Missing values in donations DataFrame')
msno.matrix(donations)


# In[ ]:


print('Missing values in teachers DataFrame')
msno.matrix(teachers)


# ## 4. Donation by people

# ### 4.1 Total Donations

# In[47]:


fig = plt.Figure(figsize=(12,12))
sns.distplot(donations['Donation Amount'].dropna())
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Amount Spent')
plt.show()


# ### 4.2 What is the Total Amount donated by people? How much donated by teachers?

# In[48]:


teachers_donor_id = donors[donors['Donor Is Teacher'] == 'Yes']['Donor ID']
amount_teachers_donated = donations[donations['Donor ID']                                    .isin(teachers_donor_id)]['Donation Amount']                                    .sum()
    
total_donation_amount = donations['Donation Amount'].dropna().sum()


# In[176]:


teacher_or_not = ['Teachers', 'All']
amount = [amount_teachers_donated, total_donation_amount]

sns.barplot(teacher_or_not, amount)
        
plt.xlabel('People', fontsize=18)
plt.ylabel('Amount', fontsize=18)
plt.title('Amount Donated by Teachers v/s Total', fontsize=20)
plt.show()


# (1/4)th of the Total Amount donated by people of different states and cities are Teachers by profession.

# ### 4.3 Amount Donated by different states

# In[17]:


fig, ax = plt.subplots(1,2, figsize=(25,12))

donor_states = donors['Donor State'].dropna().value_counts()               .sort_values(ascending=False).head(20)
    
sns.barplot(donor_states.values, donor_states.index, ax=ax[0])
for index, value in enumerate(donor_states.values):
        ax[0].text(0.8, index, value, color='k', fontsize=12)
        
ax[0].set_xlabel('Donors', fontsize=18)
ax[0].set_ylabel('States', fontsize=18)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=15)
ax[0].set_title('Donors from different States', fontsize=20)



donor_cities = donors['Donor City'].dropna().value_counts()               .sort_values(ascending=False).head(20)
    
sns.barplot(donor_cities.values, donor_cities.index, ax=ax[1])
for index, value in enumerate(donor_cities.values):
        ax[1].text(0.8, index, value, color='k', fontsize=12)
        
ax[1].set_xlabel('Donors', fontsize=18)
ax[1].set_ylabel('City', fontsize=18)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=15)
ax[1].set_title('Donors from dfferent cities', fontsize=20)
plt.show()


# **Top five states from where people donated money are:**<br>
# - California (294695 people)
# - New York (137957 people)
# - Texas(134449 people)
# - Florida (108828 people)
# - illinois (104381 people)
# 
# **Top five cities from where people donated money are:**<br>
# - Chicago (34352 people)
# - New york (27863 people)
# - Brookline (22330 people)
# - Los Angels (18320 people)
# - San Francisco (16925 people)

# ### 4.4 How many Donations include optional Donation?
# Optional donation means there may be requirement for other funds except the classroom requirement.

# In[189]:


donations['Donation Included Optional Donation'].value_counts().plot.bar()
plt.title('Optional Donation Yes/No ?', fontsize=15)
plt.xlabel('Yes/No', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.show()


# ### 4.5 After how many days first donation for a project has been received by different states? 

# In[8]:


project_posted_date = pd.to_datetime(project_donations['Project Posted Date'])
donation_receive_date = pd.to_datetime(project_donations['Donation Received Date'])
days = (donation_receive_date - project_posted_date).dt.days
days.describe()


# In[46]:


sns.boxplot(days)
plt.xlabel('Days', fontsize=14)
plt.title('Days taken to receive first donation', fontsize=16)
plt.show()


# For most of the projects, the time taken to get first donation is between **1 to 10 days.**

# ## 5. Get some Info about projects!!

# ### 5.1 Project's Status and Time taken to get funded!!
# ### 5.1.1 What is the status of Current Projects?

# In[62]:


project_current_status = projects['Project Current Status'].value_counts()                        [['Fully Funded', 'Expired', 'Live']]
    
plt.pie(project_current_status.values, labels=list(project_current_status.index),        autopct='%1.1f%%', shadow=True)

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Current Status of Projects', fontsize=20)

plt.show()


# ### 5.1.2 Number of times a donor has donated 

# In[69]:


categories = ['Literacy & Language', 'Literacy & Language, Math & Science', 'Math & Science',
              'Music & The Arts', 'Literacy & Language, Special Needs', 'Applied Learning',
              'Applied Learning, Literacy & Language']

donors = ['Donor1', 'Donor2', 'Donor3', 'Donor4', 'Donor5', 'Donor6', 'Donor7', 'Donor8',
          'Donor9', 'Donor10', 'Donor11', 'Donor12', 'Donor13', 'Donor14', 'Donor15',
          'Donor16', 'Donor17', 'Donor18', 'Donor19', 'Donor20']

donation_ids = donations['Donor ID'].value_counts().head(20)
only_donation_ids = donation_ids.index
df = {}
df['categories'] = []
df['donors'] = []
df['count'] = []
for i, id in enumerate(only_donation_ids):
    project_category = project_donations[project_donations['Donor ID'] == id]                        ['Project Subject Category Tree']
    for category in categories:
        try:
            df['count'].append(project_category.str.replace('"', '')                               .str.replace("'", '')                               .str.lstrip().value_counts()[category])
            
            df['categories'].append(category)
            df['donors'].append(donors[i])
        except:
            df['count'].append(0)
            df['categories'].append(category)
            df['donors'].append(donors[i])
    


# In[89]:


fig, ax = plt.subplots(1, 2, figsize=(25, 14))

donation_ids = donations['Donor ID'].value_counts().head(20)
sns.barplot(donation_ids.values, donors, ax=ax[1])

ax[1].set_ylabel('Donors', fontsize=20)
ax[1].set_xlabel('Number of times Donor Donated', fontsize=20)
ax[1].set_title('Number of times Donations made by Donor', fontsize=25)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=20)
ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=20)

for index, value in enumerate(donation_ids):
        ax[1].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)


        
df = pd.DataFrame(df)
data_df = df.pivot('categories', 'donors', 'count')
sns.heatmap(data_df, cmap='YlGnBu', fmt='2.0f', linewidths=.5, ax=ax[0])
ax[0].set_ylabel('Categories', fontsize=20)

ax[0].set_xlabel('Donors', fontsize=20)
ax[0].set_title('Subject Categories for which Donors donated mostly!', fontsize=25)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, fontsize=20)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=20)


plt.show()


# From the above two graphs, we can see that<br>
# 1. Donor1, Donor2, Donor3 has donated money many times for many different projects as seen from the second plot. <br>
# 2. And comparing with the first plot, we can see that<br>
#     - Donor1 has contributed maximum number of times for projects under category **"Music & The Arts"**.
#     - Donor2 has also contributed maximum number of times for **"Music & The Arts"** but also many times for **Math & Science, Literacy & Language.**
# 3. **Most of the people prefer to donate for Music & The Arts, Math & Science or Literacy & Language.**

# In[90]:


print('Mapping of Donors with their Donor ID')
pd.DataFrame({'Donors': donors, 'Donor ID': donation_ids.index})


# ### 5.2 What are the Project Categories/sub-categories/Grade level category where schools need funds? What are their Resources?
# 

# ### 5.2.1  Let's first see whether the data is cleaned or not for Project Categories?

# In[174]:


list(projects['Project Subject Category Tree'].unique()[0:10])


# In the Project Subject Category column of "projects" DataFrame, the categories has some unwanted special characters which makes the two same cateogories different. **For example,**<br>
# 1) Math & Science<br>
# 2) Math & Science'<br>
# 
# Both 1) and 2) are same but the special character (") make them different while producing plots and **some has blank spaces in the begining,<br> for example ' Special Needs"'**.<br> In most of the work, people forget to clean the data. **So we need to remove them while doing our analysis**.
# 
# ### 5.2.2 Visualizing the categories!

# In[175]:


project_category = projects['Project Subject Category Tree'].str.replace('"', '')                    .str.replace("'", '')                    .str.lstrip().value_counts().head(30)
        
project_subcategory = projects['Project Subject Subcategory Tree']                    .value_counts().head(30)
    
project_grade_level_category = projects['Project Grade Level Category']                                .value_counts().head(5)


# In[130]:


f1 = plt.figure(1, figsize=(25,25))
ax1 = plt.subplot2grid((2, 2), (0, 0))
sns.barplot(project_category.values, project_category.index, ax=ax1)
ax1.set_xlabel('Count', fontsize=23)
ax1.set_ylabel('Category', fontsize=23)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=20)
ax1.set_title('What are Project Categories?', fontsize=25)


f2 = plt.figure(1, figsize=(25,25))
ax2 = plt.subplot2grid((2, 2), (1, 0))
sns.barplot(project_subcategory.values, project_subcategory.index, ax=ax2)
ax2.set_xlabel('Count', fontsize=23)
ax2.set_ylabel('Subcategory', fontsize=23)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=20)
ax2.set_title('What are Project Sub Categories?', fontsize=25)


f3 = plt.figure(1, figsize=(25,25))
ax3 = plt.subplot2grid((2, 2), (0, 1))
wc = WordCloud(background_color="white", max_words=500, 
               stopwords=STOPWORDS, width=1000, height=1000)
wc.generate(" ".join(projects['Project Resource Category'].dropna()))

ax3.imshow(wc)
ax3.axis('off')
ax3.set_title('Resource Category', fontsize=25)


f2 = plt.figure(1, figsize=(25,25))
ax4 = plt.subplot2grid((2, 2), (1, 1))
patches, texts, autotexts = ax4.pie(project_grade_level_category.values,                                    labels=list(project_grade_level_category.index),                                    autopct='%1.1f%%', shadow=True)

[ _.set_fontsize(20) for _ in texts]

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Grade Level Category', fontsize=25)

plt.show()


# **Top Project Categories where different schools are working invloves:**<br>
# - Literacy and Language
# - Maths and Science
# - Music and Art
# - Social Needs, etc
# 
# **Top Project Categories where different schools are working invloves:**<br>
# Sub categoy of any project will be under Project category. So in the plot, it is clear that top sub categories where schools are working on are:<br>
# - Literacy and Mathematics
# - Language
# - Writing, etc
# 
# **What are Top resources schools are getting?**<br>
# - Technology
# - Books
# - Supplies
# - Computers, etc
# 
# **Top Grade level cateogies with respect to number of schools working**<br>
# - Grade PreK-2
# - Grade 3-5
# - Grade 6-8
# - Grade 9-12

# ### 5.3 What is the type of project?
# Let's see how many different types of projects are?

# In[9]:


project_types = projects['Project Type'].value_counts().head(3)
fig = plt.figure(figsize=(8, 5))
sns.barplot(project_types.index, project_types.values, palette="Blues_d")
plt.xlabel('Project Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Types of Projects', fontsize=16)
plt.show()


# ### 5.4 Who are providing the Resources?
# We have seen that schools get resources like books, computer, tablets, etc. We should fing who are the Resource Vendors?

# In[138]:


resource_vendors = resources['Resource Vendor Name'].value_counts().head(15)

fig = plt.figure(figsize=(8,8))
sns.barplot(resource_vendors.values, resource_vendors.index)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Vendors', fontsize=15)
plt.title('Who are Resource Vendors?', fontsize=18)
plt.show()


# **Amazon, Lakeshore Learning Materials, AKJ Education** are among the top vendors of resources upon which schools are relying.

# ## 6. Description of Schools
# ### 6.1 Where these schools are located (Rural/ Urban)?

# In[143]:


school_metro_type = schools['School Metro Type'].value_counts().drop('unknown')
fig = plt.figure(figsize=(8,8))
plt.pie(school_metro_type.values, labels = school_metro_type.index, autopct='%1.1f%%')

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Metero Type of Schools', fontsize=20)
plt.show()


# Most of the schools belong to **sub-urban or Urban** regions with less proportion from rural regions.

# ## 6.2 Which state's schools provide free lunch and to what proportion?

# In[168]:


schools_give_lunch = schools[['School Percentage Free Lunch', 'School State']]
schools_give_lunch.groupby('School State')['School Percentage Free Lunch']                  .describe().sort_values(by='mean', ascending=False).head(5)


# **mean** value of percentage of lunch provided at different schools of particular district can give an idea of how much state wise free luch is provided?<br>
# **Top 5 States which provide large proportion of free lunch are**<br>
# - District of Columbia
# - Mississipppi
# - Louisiana
# - New Mexico
# - Oklahoma

# ### 7. Projects First posted by Teachers!!

# In[91]:


f, ax = plt.subplots(1, 2, figsize=(15, 8))


# Projects posted every year
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
num_projects_posted = teachers.groupby(teachers['Teacher First Project Posted Date']                                       .dt.year)                                       .count()['Teacher First Project Posted Date']

ax[0].plot(num_projects_posted)
ax[0].set_xlabel('Years', fontsize=14)
ax[0].set_ylabel('Count', fontsize=14)
ax[0].set_title('Projects posted by Teachers as per Year', fontsize=16)



# Projects posted in 2018
new_index = ['0', 'Jan', 'Feb', 'March', 'April', 'May']
num_projects_posted_2018 = teachers.groupby(teachers[teachers['Teacher First Project Posted Date']                                            .dt.year == 2018]                                            ['Teacher First Project Posted Date']                                           .dt.month)                                           .count()['Teacher ID']

ax[1].plot(num_projects_posted_2018)
ax[1].set_xlabel('Months', fontsize=14)
ax[1].set_ylabel('Count', fontsize=14)
ax[1].set_xticks(np.arange(6))
labels = [new_index[i] for i, item in enumerate(ax[1].get_xticklabels())]

ax[1].set_xticklabels(labels)

ax[1].set_title('Projects posted by Teachers as per months for year 2018', fontsize=16)
plt.show()


# The number of project posted by teachers got **increased from 2002 to 2016** and then there is **some decrease in 2017**. In 2018, until May, there has been many projects posted approximately 25000 with **maximum projects being posted in April** as seen from second plot. It is less in May because the data contains only the starting days of the month May.

# ## 8. Summary
# 1. Schools get funds for their projects from various sources and help students to complete the projects.<br>
# 2. Most of the schools are from urban or sub urban regions and they get funds from many people include (1/4th) of fund from Teachers.
# 3. Amazon, Lakeshore Learning Materials are among top vendors of resources.
# 4. People from different states give donation to help project completion.
# **5. We get some pattern from the analysis like what projects mostly attract donors?**<br>
#     * Music and Arts<br>
#     * Literacy & Science<br>
# 6. From section 7, we analysed that the projects posted by teachers are increasing from 2002 to 2016 and then there is a little fall in projects posted and again rise in 2018 in the month of April.<br>
# 
# **I will keep updating the notebook, so please stay Tunned!!**<br>
# If this notebook is helpful, then **Please UpVote!!**
