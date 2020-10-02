#!/usr/bin/env python
# coding: utf-8

# ![](data:image/png;base64,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)

# # Introduction
# 
# 
# Donorschoose.org is a nonprofit organization that allows individuals to donate directly to public school classroom projects. Teachers from public schools post request for funding project with a short essay describing it. Donors all around the world can look at these projects when they login to Donorschoose.org and donate to projects of their choice. The idea is to have personalized recommendation webpage for all the donors, which will show them the projects, which they prefer, like and love to donate. Implementing the recommender system for DonorsChoose.org website will improve user experience and help more projects to meet their funding goals. 
# 
# 

# * <a href='#1'>1. Loading Libraries</a>  
# * <a href='#2'>2. Read Data</a>  
# * <a href='#3'>3. Sample data</a>  
# * <a href='#4'>4.Analyzing each data set</a>
#     *        <a href='#4.1'>4.1 Analyzing Resource Data set</a>
#     *        <a href='#4.2'>4.2 Analyzing School Data set</a>
#     *        <a href='#4.3'>4.3 Analyzing Donors Data set</a>
#     *        <a href='#4.4'>4.4 Analyzing Donations Data set</a>
#     *        <a href='#4.5'>4.5 Analyzing Teachers Data set</a>
#     *        <a href='#4.6'>4.6 Analyzing Projets Data set</a>

# ## <a id='1'>1. Loading Libraries</a>

# In[124]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# linear algebra
import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

#visualization
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from wordcloud import WordCloud, STOPWORDS

from scipy.stats import kurtosis
from scipy.stats import skew
stopwords = set(STOPWORDS)
from textblob import TextBlob



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## <a id='2'>2. Read Data</a>

# In[2]:


resource = pd.read_csv('../input/Resources.csv')
schools = pd.read_csv('../input/Schools.csv')
donors = pd.read_csv('../input/Donors.csv')
donations = pd.read_csv('../input/Donations.csv')
teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')


# In[3]:


#function for displaying top 20 values using plotly bar plot

def vert_bar_plot(df, col, title):
    trace = go.Bar(
    x = df[col].value_counts()[:20].index,
    y = df[col].value_counts()[:20].values,
    text = df[col].value_counts()[:20].values,
    textposition = 'auto',
    marker = dict(
        color = 'rgb(153,153,255)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5            
        ),
    ),
    opacity = 0.6  
    )
    layout = dict(
    title=title,
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
#state codes list which will be used for displaying state stats in maps
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}


# <a href='#3'>3. Sample Data</a> 
# 

# **Resource**

# In[4]:


resource.head()


# **Schools**

# In[5]:


schools.head()


# **Donors**

# In[6]:


donors.head()


# **Donations**

# In[7]:


donations.head()


# **Teachers**

# In[8]:


teachers.head()


# **Projects**

# In[10]:


projects.head()


# ## <a id='4.1'>4.1. Analyzing Resource Data set</a>

# ## Exploring missing datas in Resource data set

# In[45]:


msno.matrix(resource)
plt.show()
print("Missing datas\n" , pd.isnull(resource).sum())


# ## Top Resource Vendor Names

# In[12]:


vert_bar_plot(resource, 'Resource Vendor Name', 'Top 20 Resource Vendors')


# Top 3 resource vendors contribute more than 70% of resource. 
# 
# Amazon Business tops vendor names list with 44% followed by Lakeshore Learning Materials with 14% and AKJ Education  with 13%

# ## Distribution of Resource Unit Price

# In[20]:


plt.figure(figsize = (12,8))
plt.scatter(range(resource.shape[0]), np.sort(resource['Resource Unit Price'].values))
plt.xlabel('Resource unit Price')
plt.show()

plt.figure(figsize = (16,8))
sns.distplot(resource['Resource Unit Price'].dropna().apply(np.sqrt))
plt.show()


# In[22]:


print("Skewness ", skew(resource['Resource Unit Price'].dropna()))
print("Kurtosis ", kurtosis(resource['Resource Unit Price'].dropna()))


# Resource unit price is positively skewed with skewness 79.75 and kurtosis 26893.28

# 
# ## <a id='4.2'>4.2. Analyzing School Data set</a>

# 
# ## Missing values in Schools

# In[46]:


msno.matrix(schools)
plt.show()
print("Shape of Schools " , schools.shape)
print("Missing datas\n" , pd.isnull(schools).sum())


# ### Distribution of School Metro Type

# In[24]:


data = [go.Pie( labels = schools['School Metro Type'].value_counts().index, values = schools['School Metro Type'].value_counts().values  )]
layout = dict(
    title='Distribution of School Metro Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# More than half of the schools are located in urban and suburban areas

# ### Schools in each City

# In[25]:


vert_bar_plot(schools, 'School City', 'No.of Schools in each City')


# ### schools in different Districts

# In[26]:


vert_bar_plot(schools, 'School District', 'No .of Schools in different Districts')


# In[34]:


schools_75_free_lunch = schools[schools['School Percentage Free Lunch'] > 75.0]
metro_free_lunch = schools_75_free_lunch.groupby('School Metro Type').agg({'School Metro Type' : 'count', 'School Percentage Free Lunch' : 'count'})
data = [go.Pie( labels = metro_free_lunch.index, values = metro_free_lunch['School Percentage Free Lunch']  )]
layout = dict(
    title='Schools providing free lunch with more than 75% based on Metro Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Larger number of urban schools provide free lunch for students than other metro type

# In[36]:


vert_bar_plot(schools_75_free_lunch, 'School City', 'No.of Schools providing free lunch with more than 75% based on city')


# In[40]:


schools_25_free_lunch = schools[schools['School Percentage Free Lunch'] < 25.0]
metro_free_lunch = schools_25_free_lunch.groupby('School Metro Type').agg({'School Metro Type' : 'count', 'School Percentage Free Lunch' : 'count'})
data = [go.Pie( labels = metro_free_lunch.index, values = metro_free_lunch['School Percentage Free Lunch']  )]
layout = dict(
    title='Schools providing free lunch with less than 25% based on Metro Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Larger number of suburban schools are providing less than 25% free lunch to there student

# In[44]:


vert_bar_plot(schools_25_free_lunch, 'School City', 'No.of Schools providing less than 25% free lunch  based on city')


# Surprisingly New York City which provides higher percentage of free lunch is also listed in cities providing lower percentage of free lunch

# In[41]:


school_free_lunch = schools.groupby('School State').agg({'School Name' : 'count', 'School Percentage Free Lunch' : 'mean'}).reset_index()
school_free_lunch.columns = ['state' , 'school', 'free_lunch']
for col in school_free_lunch.columns:
    school_free_lunch[col] = school_free_lunch[col].astype(str)
school_free_lunch['text'] = school_free_lunch['state'] + '<br>' + '% of free lunch provided :' + school_free_lunch['free_lunch']
school_free_lunch['code'] = school_free_lunch['state'].map(state_codes)


# In[42]:


data = [dict( 
            type = 'choropleth',
            autocolorscale = True,
            locations = school_free_lunch['code'],
            z = school_free_lunch['free_lunch'].astype(float),
            locationmode = 'USA-states',
            text = school_free_lunch['text'].values,
            colorbar = dict(
                title = '% of free lunch'
            )
            )]

layout = dict(
        title = "Percentage of free lunch by states",
        geo = dict(
        scope = 'usa',
        projection = dict( type='albers usa' ),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
        )

fig = dict(data=data, layout=layout)
iplot(fig)


# In[43]:


school_free_lunch['text'] = school_free_lunch['state'] + '<br>' + '# of schools :' + school_free_lunch['school']
data = [dict( 
            type = 'choropleth',
            autocolorscale = True,
            locations = school_free_lunch['code'],
            z = school_free_lunch['school'].astype(float),
            locationmode = 'USA-states',
            text = school_free_lunch['text'].values,
            colorbar = dict(
                title = '# of Schools'
            )
            )]

layout = dict(
        title = "Number of Schools in Different state",
        geo = dict(
        scope = 'usa',
        projection = dict( type='albers usa' ),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
        )

fig = dict(data=data, layout=layout)
iplot(fig)


# 
# ## <a id='4.3'>4.3. Analyzing Donors Data set</a>

# ### Missing values in Donors

# In[49]:


msno.matrix(donors)
plt.show()
print("Shape of Donors " , donors.shape)
print("Missing datas\n" , pd.isnull(donors).sum())


# ### Top Donors by City

# In[50]:


vert_bar_plot(donors, 'Donor City', 'Top Donors by cities')


# ### Distribution of Donor teachers

# In[51]:


data = [go.Pie( labels = donors['Donor Is Teacher'].value_counts().index, values = donors['Donor Is Teacher'].value_counts().values  )]
layout = dict(
    title='Distribution of Donor teachers',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## <a id='4.4'>4.4 Analyzing Donations Data set</a>

# In[53]:


msno.matrix(donors)
plt.show()
print("Shape of Donations " , donations.shape)
print("Missing datas\n" , pd.isnull(donations).sum())


# ### Top 20 Donors

# In[55]:


vert_bar_plot(donations, 'Donor ID', 'Top Donors')


# ### Distribution of Repeating Donors

# In[56]:


repeating_donors = donations.groupby('Donor ID').agg({'Donor ID' : 'count', 'Donation Amount': 'sum'})
non_repeating = repeating_donors[repeating_donors['Donor ID'] == 1].shape[0]
two_times = repeating_donors[(repeating_donors['Donor ID']>2) & (repeating_donors['Donor ID'] < 5)].shape[0]
five_times = repeating_donors[(repeating_donors['Donor ID']>= 5) & (repeating_donors['Donor ID'] < 10)].shape[0]
ten_times = repeating_donors[repeating_donors['Donor ID']>= 10].shape[0]


labels = ['Non repeating','2 times','5 times','10 times']
values = [non_repeating,two_times,five_times,ten_times]

data = [go.Pie(labels=labels, values=values)]
layout = dict(
    title='Distribution of Repeating Donors',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Donation Amount by repeating Donors

# In[59]:


repeating_donors['Donation Amount'].sum()
non_repeating_donations = repeating_donors['Donation Amount'][repeating_donors['Donor ID'] == 1].sum()
two_times_donations = repeating_donors['Donation Amount'][(repeating_donors['Donor ID']>2) & (repeating_donors['Donor ID'] < 5)].sum()
five_times_donations = repeating_donors['Donation Amount'][(repeating_donors['Donor ID']>= 5) & (repeating_donors['Donor ID'] < 10)].sum()
ten_times_donations = repeating_donors['Donation Amount'][repeating_donors['Donor ID']>= 10].sum()

labels = ['Non repeating','2 times','5 times','10 times']
values = [non_repeating_donations,two_times_donations,five_times_donations,ten_times_donations]

trace = go.Bar(
    x = labels,
    y = values,
    text = values,
    textposition = 'auto',
    marker = dict(
        color = 'rgb(153,153,255)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5            
        ),
    ),
    opacity = 0.6  
    )
layout = dict(
    title='Donation Amount by repeating Donors',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

data = [go.Pie(labels=labels, values=values)]
layout = dict(
    title='Donation Amount by repeating Donors',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Around 102 million dollars (41%) of donation have been donated by donors who have donated more than 10 times and around 80 million dollars have been donated by non repeating donors

# In[60]:


optional_donations = donations.groupby('Donation Included Optional Donation').agg({'Donation Amount': 'sum'}).reset_index()


# ### Distribution of donation amount base on optional donations

# In[63]:


data = [go.Pie( labels = optional_donations['Donation Included Optional Donation'], values = optional_donations['Donation Amount'])]
layout = dict(
    title='Distribution of donation Amount based on optional donations',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[62]:



data = [go.Pie( labels = donations['Donation Included Optional Donation'].value_counts().index, values = donations['Donation Included Optional Donation'].value_counts().values  )]
layout = dict(
    title='Distribution of optional donations',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Distribution of donation amount

# In[65]:


plt.figure(figsize = (12,8))
plt.scatter(range(donations.shape[0]), np.sort(donations['Donation Amount'].values))
plt.xlabel('Distribution of donation amount')
plt.show()

plt.figure(figsize = (16,8))
sns.distplot(donations['Donation Amount'].apply(np.log))
plt.show()


# ### Donor cart sequences

# In[67]:


#Donor Cart Sequence
vert_bar_plot(donations, 'Donor Cart Sequence', 'Donor sequences')


# ## <a id='4.5'>4.5 Analyzing Teachers Data set</a>

# ### Distribution of Teacher Prefixes

# In[70]:


data = [go.Pie( labels = teachers['Teacher Prefix'].value_counts().index, values = teachers['Teacher Prefix'].value_counts().values  )]
layout = dict(
    title='Distribution of Teacher Prefixes',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Teacher Gender Distribution

# In[79]:


female = teachers[teachers['Teacher Prefix'] == 'Mrs.']['Teacher ID'].count() + teachers[teachers['Teacher Prefix'] == 'Ms.']['Teacher ID'].count()
male = teachers[teachers['Teacher Prefix'] == 'Mr.']['Teacher ID'].count()
x = ['female' , 'male']
y = [female, male]
plt.figure(figsize = (12,8))
sns.barplot(x=x, y=y)
plt.title('Teacher Gender Distribution')
plt.show()


# In[89]:


teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
teachers['posted_year'] = teachers['Teacher First Project Posted Date'].dt.year
teachers['posted_month'] = teachers['Teacher First Project Posted Date'].dt.month
teachers['posted_day'] = teachers['Teacher First Project Posted Date'].dt.dayofweek


# In[90]:


vert_bar_plot(teachers, 'posted_month', 'No of projects posted by month')


# In[91]:


vert_bar_plot(teachers, 'posted_day', 'No of projects posted by day')


# In[92]:


year_wise = teachers.groupby('posted_year').agg({'Teacher ID' : 'count'}).reset_index()
data = [go.Scatter(x=year_wise.posted_year, y=year_wise['Teacher ID'])]
layout = dict(
    title='Teacher projects posted by year',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[93]:


projects['Project Cost'] = projects['Project Cost'].str.replace(',', '')
projects['Project Cost'] = projects['Project Cost'].str.replace('$', '')
projects['Project Cost'] = projects['Project Cost'].astype(float)


# ## <a id='4.6'>4.6. Analyzing Projects Dataset</a>

# ### Project type Distribution

# In[98]:


vert_bar_plot(projects, 'Project Type', 'Project type Distribution')

data = [go.Pie( labels = projects['Project Type'].value_counts().index, values = projects['Project Type'].value_counts().values  )]
layout = dict(
    title='Distribution of Project Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Distribution of Project Subject Category

# In[99]:


vert_bar_plot(projects, 'Project Subject Category Tree', 'Project Subject Category')


# In[100]:


### Distribution of Project Subject Subcategory


# In[101]:


vert_bar_plot(projects, 'Project Subject Subcategory Tree', 'Project Subject Subcategory')


# In[102]:


#Project Grade Level Category
vert_bar_plot(projects, 'Project Grade Level Category', 'Project Grade Level Category')


# In[103]:


#Project Resource Category
vert_bar_plot(projects, 'Project Resource Category', 'Project Resource Category')


# In[104]:


projects['Project Posted Date'] = pd.to_datetime(projects['Project Posted Date'])
projects['posted_year'] = projects['Project Posted Date'].dt.year
projects['Project Fully Funded Date'] = pd.to_datetime(projects['Project Fully Funded Date'])
projects['funded_year'] = projects['Project Fully Funded Date'].dt.year


# In[106]:


posted = projects.groupby('posted_year').agg({'Teacher ID' : 'count'}).reset_index()
funded = projects.groupby('funded_year').agg({'Teacher ID' : 'count'}).reset_index()
posted = go.Scatter(x=posted.posted_year, y=posted['Teacher ID'])
funded = go.Scatter(x=funded.funded_year, y=funded['Teacher ID'])
layout = dict(
    title='Projects posted and funded',
    )
data = [posted, funded]
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[107]:


#Project Current Status
data = [go.Pie( labels = projects['Project Current Status'].value_counts().index, values = projects['Project Current Status'].value_counts().values  )]
layout = dict(
    title='Distribution of Project Current Status',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[109]:


t = projects.groupby('Project Current Status').agg({'Project Cost' : 'mean'}).reset_index()
plt.figure(figsize = (12,8))
sns.barplot(x="Project Current Status", y="Project Cost", data=t)
plt.show()


# In[110]:


#Project Grade Level Category
t = projects.groupby('Project Grade Level Category').agg({'Project Cost' : 'mean'}).reset_index()
plt.figure(figsize = (12,8))
sns.barplot(x="Project Grade Level Category", y="Project Cost", data=t)
plt.show()


# Project cose for Grades 9 -12 is comparatively higher than other grades

# In[111]:


#Project Resource Category
t = projects.groupby('Project Resource Category').agg({'Project Cost' : 'mean'}).reset_index()
plt.figure(figsize = (12,8))
sns.barplot(x="Project Cost", y="Project Resource Category", data=t)

plt.show()


# In[113]:


projects.info()


# ### Common words used in Project Title

# In[121]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(projects['Project Title']))

fig = plt.figure(figsize = (14,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[122]:


wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(projects['Project Essay']))

fig = plt.figure(figsize = (14,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:




