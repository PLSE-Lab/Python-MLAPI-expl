#!/usr/bin/env python
# coding: utf-8

# <img src="https://3.bp.blogspot.com/-Tgg-4T00JyQ/VDGkUaOm_pI/AAAAAAAAFvU/JoiNszeL4xE/s1600/Screen%2BShot%2B2013-12-04%2Bat%2B11.53.12%2BPM.png" width='600' heigh='100' />

# ## [Table of Contents](#here)
# ___
# 
# * [Introduction](#introdction)
#     * [Summary](#summary)
#     * [Problem Statement](#problem)
#     
# * [Load Packages & Data](#data-packages)
#     * [Load Packages](#packages)
#     
#     * [List Data](#list)
#     
#     * [Load DonorsChoose.org Data](#data)
#     
# * [Analysis](#analysis)
# 
#     * [Teachers](#teachers)
#         * [Partial View of Data](#teachers-data)
#     * [Shools](#schools)
#         * [Partial View of Data](#teachers-schools)
#     * [Projects](#projects)
#         * [Partial View of Data](#teachers-projects)
#     * [Resources](#resource)
#         * [Partial View of Data](#teachers-resource)
#     * [Donors](#donors)
#         * [Partial View of Data](#teachers-donors)
#     * [Donations](#donations)
#         * [Partial View of Data](#teachers-donations)
#         
# * [Machine Learning](#machine-learning)
# 
# * [Naratives](#naratives)

# # Introduction <a class='anchor' id='introduction'></a>

# ## Summary <a class='anchor' id='summary'></a>
# ___
# 
# Founded in 2000 by a Bronx history teacher, DonorsChoose.org has raised $685 million for America's classrooms. Teachers at three-quarters of all the public schools in the U.S. have come to DonorsChoose.org to request what their students need, making DonorsChoose.org the leading platform for supporting public education.
# 
# To date, 3 million people and partners have funded 1.1 million DonorsChoose.org projects. But teachers still spend more than a billion dollars of their own money on classroom materials. To get students what they need to learn, the team at DonorsChoose.org needs to be able to connect donors with the projects that most inspire them.
# 
# In the second Kaggle Data Science for Good challenge, DonorsChoose.org, in partnership with Google.org, is inviting the community to help them pair up donors to the classroom requests that will most motivate them to make an additional gift. To support this challenge, DonorsChoose.org has supplied anonymized data on donor giving from the past five years. The winning methods will be implemented in DonorsChoose.org email marketing campaigns.

# ## Problem Statement <a class='anchor' id='problem'></a>
# ___
# DonorsChoose.org has funded over 1.1 million classroom requests through the support of 3 million donors, the majority of whom were making their first-ever donation to a public school. If DonorsChoose.org can motivate even a fraction of those donors to make another donation, that could have a huge impact on the number of classroom requests fulfilled.
# 
# A good solution will enable DonorsChoose.org to build targeted email campaigns recommending specific classroom requests to prior donors. Part of the challenge is to assess the needs of the organization, uncover insights from the data available, and build the right solution for this problem. Submissions will be evaluated on the following criteria:
# 
# * **Performance** - How well does the solution match donors to project requests to which they would be motivated to donate? DonorsChoose.org will not be able to live test every submission, so a strong entry will clearly articulate why it will be effective at motivating repeat donations.
# 
# * **Adaptable** - The DonorsChoose.org team wants to put the winning submissions to work, quickly. Therefore a good entry will be easy to implement in production.
# 
# * **Intelligible** - A good entry should be easily understood by the DonorsChoose.org team should it need to be updated in the future to accommodate a changing marketplace.

# # Load Packages & Data <a class='anchor' id='data-packages'></a>
# ___
# We have to load packages and Data, pandas will be used to load and manipulate data, we will do the analysis based on the data provided by DonorsChoose.org 

# ## Load Packages <a class='anchor' id='packages'></a>
# Lets load packages and make the defaults colors and package settings for display

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os as os
import squarify
import plotly.graph_objs as go
import plotly.tools as tools
import plotly.offline as ply
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from mpl_toolkits.mplot3d import axes3d
from wordcloud import WordCloud, STOPWORDS
from plotnine import *

# default notebook setting 
color = sns.color_palette()
ply.init_notebook_mode(connected=True)
pd.options.display.max_columns = 1000
get_ipython().run_line_magic('matplotlib', 'inline')


# ### List Data <a class='anchor' id='list'></a>
# Lets see what input data directory have 

# In[3]:


os.listdir('../input/')


# ### Load DonorsChoose.org Data <a class='anchor' id='data'></a>
# Now that we have data lets load it for analysis

# In[4]:


resources = pd.read_csv('../input/Resources.csv', low_memory=False)
teachers = pd.read_csv('../input/Teachers.csv', low_memory=False)
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', low_memory=False)
donations = pd.read_csv('../input/Donations.csv', low_memory=False)
projects = pd.read_csv('../input/Projects.csv', low_memory=False)
# error_bad_lines=False


# # Analysis <a class='anchor' id='analysis'></a>
# ___
# This Section provide analysis of all the data provided by DonorChoose.org, As outlined above we will start with the teachers analysis first then we will go to other dataset

# ## Teachers <a class='anchor' id='teachers'></a>
# This section will cover teachers dataset

# ### Partial View of Data  <a class='anchor' id='teachers-data'></a>
# Lets see the teacher's dataset how it looks like

# In[5]:


teachers.head()


# ### How many unique teachers in the past 16 years
# Lets have a look how many teachers registered on DonorChoose.org in the past 15 years and 8 months

# In[6]:


print("\x1b[1;31m We have {0} records \x1b[0m ".format(teachers.shape[0]))
min_date = teachers['Teacher First Project Posted Date'].min()
max_date = teachers['Teacher First Project Posted Date'].max()
num_teachers = len(pd.unique(teachers['Teacher ID']))
print('\x1b[1;31m' + ' From {0} to {1} we have '.format(min_date, max_date) + '\x1b[0m')
_ = venn2((num_teachers,0,0), set_labels=('Teachers',''))
_.get_label_by_id('01').set_text('')


# * From 2002 to this year we have 402900 teachers that registered their first projects on DonorsChoose.org

# ### Lets see the statistics for teachers gender based

# In[7]:


titles = teachers['Teacher Prefix'].value_counts()
trace = go.Table(
    header=dict(values=['Title','Total'],
                fill = dict(color='#cd5c5c'),
                align = ['left']),
    cells=dict(values=[titles.index, titles.values],
               fill = dict(color='#ffb6c1'),
               align = ['left']))

data = [trace] 
ply.iplot(data, filename = 'table')


# ### Visually  

# In[8]:


(ggplot(teachers.dropna()) + 
   aes(x='Teacher Prefix', fill='Teacher Prefix') +
   geom_bar() + 
   ggtitle("Number of Teachers by Title/Prefix") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# ### Visually 

# In[9]:


X=teachers['Teacher Prefix'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_chart')


# We have 
# * <span style="color:#000066">50.2%</span> Mrs (Married)
# * <span style="color:#660088">36.2%</span> Ms (single)
# * <span style="color:#006600">11.8</span> Mr
# * <span style="color:#F08080">1.84</span> Teacher
# * <span style="color:#F08080">0.00141%</span> Dr
# * <span style="color:#F08080">0.00695%</span> Mx  
# This shows lot of teachers creating projects are ladies mostly married women  

# ### Gender Identification 
# Cleaning up gender and classifying into three groups

# In[10]:


gender_identifier = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Not Specified", 
                     "Dr.":"Not Specified", "Mx.":"Not Specified" }

teachers["gender"] = teachers['Teacher Prefix'].map(gender_identifier)


# ### Visual Representation of Gender (Percentages %)

# In[11]:


X=teachers['gender'].value_counts()
colors = ['#F08080', '#1E90FF', '#FFFF99']

plt.pie(X.values, labels=X.index, colors=colors,
        startangle=90,
        explode = (0, 0, 0),
        autopct = '%1.2f%%')
plt.axis('equal')
plt.show()


# We have 
# * <span style="color:#F08080">86.36%</span> Female Teachers posted their first projects 
# * <span style="color:#1E90FF">11.79%</span> Male Teachers posted their first projects
# * and <span style="color:#006600">1.86%</span>  Unknown Gender Teachers posted their first projects

# ### The Set of the Gender

# In[12]:


females = len(teachers[teachers["gender"]=='Female'])
males = len(teachers[teachers["gender"]=='Male'])
unspecified = len(teachers[teachers["gender"]=='Not Specified'])


# ### The set of the represented visually 

# In[13]:


_ = venn2((females,unspecified,males), set_labels=('Females','Not specified', 'Males'))


# We have 
# * <span style="color:#F08080">347903</span> Females
# * <span style="color:#1e90ff">47480</span> Males
# * <span style="color:#006600">7489</span> Unknown gender  
# as exact figures numerically 

# ### Time Series 
# ___
# Let's see teachers first posted project since the first day DonorsChoose.org was launched till the latest data given

# In[14]:


teachers['count'] = 1
long_date_name = 'Teacher First Project Posted Date'
teachers_register = teachers[[long_date_name, 'count']].groupby([long_date_name]).count().reset_index()
teachers_register[long_date_name] = pd.to_datetime(teachers_register[long_date_name])
teachers_register = teachers_register.set_index(long_date_name)
max_date = max(teachers_register.index)
min_date = min(teachers_register.index)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)
ax = teachers_register.plot(color='blue')
ax.axvspan(min_date, '2002-12-31', color='red', alpha=0.3)
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('All time Projects posted by Date')
plt.show()


# * Red Bars indicates incomplete years in <span style="color:#F08080">2002</span> and this current year <span style="color:#F08080">2018</span>, as it indicates on the graph above at the end of <span style="color:#000066">2010</span> the DonorsChoose.org started having increase in the number of teachers posting their first projects 

# **Lets get the day in which it started having more than <span style="color:#006600">500</span> in the year <span style="color:#000066">2013</span> **

# In[15]:


increase_number = teachers_register.reset_index()
start = increase_number[increase_number[long_date_name].dt.year == 2013]
max_start = start[start['count']==max(start['count'])][long_date_name].values[0]
print('\x1b[1;31m' + ' The projects Started increasing with over 500 since 2013 on {0} '.format(max_start.astype(str)[:10]) + '\x1b[0m')


# * On 27 October 2013 DonorsChoose.org started having more than 500 new projects created by more than 500 new teachers  

# In[16]:


ax = teachers_register.plot(color='blue')
ax.axvspan(max_start, max_date, color='green', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('(2013 - 2018 snip) Projects Posted by Date ')
plt.show()


# * From above diagram, the green shaded bar period it shows the most registered projects per day exceeding <span style="color:#006600">2000</span> by teachers who post their first projects with <span style="color:#F08080">86%</span> chance that are ladies   

# In[17]:


ax = teachers_register[teachers_register.index>='2013-10-27'].plot(color='blue')
ax.axhline(500, color='green', linestyle='--')
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('Closer look (2013 -2018 snip) Projects posted by Date')
plt.show()


# * Slicing the data shows the nice flow between <span style="color:red">2013</span> and <span style="color:red">2018</span> with a potential in <span style="color:red">2018</span> to exceed <span style="color:green">Total yearly posted projects </span> breaking the record for the year <span style="color:red">2016</span>

# In[18]:


ax = teachers_register[(teachers_register.index>=min_date) & 
                       (teachers_register.index<='2002-12-31')].plot(color='blue', style='.-')
ax.axhline(6, color='green', linestyle='--')
ax.axvspan(min_date, '2002-12-31', color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('Closer look (2002 snip) Projects posted by Date')
plt.show()


# * In <span style="color:red">2002</span> there was not much happening, the maximum number of projects posted were <span style="color:red">13</span> for the whole year, well the platform was new and take it as a complement for that year and most importantly the world was not connected as of today. Social Media was not matured. 

# In[19]:


ax = teachers_register[teachers_register.index>='2018-01-01'].plot(color='blue')
ax.axhline(500, color='green', linestyle='--')
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('Closer look (2018 snip) Projects posted by Date')
plt.show()


# * The world today is more connected making the platform more efficiently to be used by many teachers, and social media is making teachers more connected and help the students to learn new tools and have more efficient ways to teach students at schools, <span style="color:red">2018</span> is already <span style="color:red">4</span> month old with this dataset and it is already making a good progress. 

# In[20]:


teachers_register['Year'] = teachers_register.reset_index()[long_date_name].dt.year
per_year = teachers[[long_date_name, 'count']].groupby(long_date_name).count().reset_index()
per_year[long_date_name] = pd.to_datetime(per_year[long_date_name])
per_year['Year'] = per_year[long_date_name].dt.year

ax = per_year[['Year', 'count']].groupby('Year').agg({'count':'sum'}).plot(color='blue', style='.-')
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Year')
ax.set_title('Projects poted by Year')
plt.show()


# * Yearly figure shows increase in the creation on new projects each year with a slightly decrease in <span style="color:red">2017</span> from <span style="color:red">2016</span>.

# ### Scatter Representation for every day projects posted by teachers

# In[21]:


(ggplot(teachers_register.reset_index())
 + geom_point(aes(long_date_name, 'count', fill='count'))
 + labs(y='number of Projects posted ')
 + ggtitle("Projects posted by Date") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# ###  Gender Based Analysis
# ___
# Lets see how teachers are posting projects based on gender

# In[22]:


gender_based = teachers[[long_date_name, 'gender', 'count']].groupby([long_date_name, 'gender']).count().reset_index()
gender_based[long_date_name] = pd.to_datetime(gender_based[long_date_name])

(ggplot(gender_based) 
 + aes(long_date_name, 'count', color='factor(gender)') 
 + geom_line()
 + ggtitle("Projects posted by Date - Gender Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# * Females are actively posting their first project more than Males teachers, teachers who did not specify their gender being third and creating their first projects after a while

# In[23]:


(ggplot(gender_based[gender_based.gender=='Male'])
 + aes(long_date_name, 'count', color='gender') 
 + geom_line()
 + ggtitle("Projects posted by Date - Males ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# * ** Closer View Time Series of Male Teachers posted projects **

# In[24]:


(ggplot(gender_based[gender_based.gender=='Female']) 
 + aes(long_date_name, 'count', color='gender') 
 + geom_line()
 + ggtitle("Projects posted by Date - Females ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# * ** Closer View Time Series of Female Teachers posted projects **

# In[25]:


(ggplot(gender_based[gender_based.gender=='Not Specified']) 
 + aes(long_date_name, 'count', color='gender') 
 + geom_line()
 + ggtitle("Projects posted by Date - Not Specified ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# * ** Closer View Time Series of Unknown Gender Teachers posted projects **

# ### Yearly Gender based Projects posted

# In[26]:


teachers[long_date_name] = pd.to_datetime(teachers[long_date_name])
teachers['Year'] = teachers[long_date_name].dt.year
gender_based_year = teachers[['Year', 'gender', 'count']].groupby(['Year', 'gender']).count().reset_index()

(ggplot(gender_based_year) 
 + aes('Year', 'count', color='factor(gender)') 
 + geom_point()
 + geom_line()
 + ggtitle("Projects posted by Year - Gender Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# * Yearly representation of time series still have the same view of daily representation difference being total on yearly and daily

# ### Projects distribution 

# In[27]:


(ggplot(gender_based) 
  + aes('count', fill='gender', color='gender') 
  + ggtitle("Projects posted on Gender ") 
  + geom_density(alpha=0.2)  
)


# ### Number of posted projects by males distribution

# In[28]:


(ggplot(gender_based[gender_based.gender=='Male']) 
   + aes('count', fill='gender', color='gender') 
   + ggtitle("Projects posted for Males") 
   + geom_density(alpha=0.2)  
)


# ### Number of posted projects by Females distribution

# In[29]:


(ggplot(gender_based[gender_based.gender=='Female']) 
  + aes('count', fill='gender', color='gender')
  +  ggtitle("Projects posted for Females ") 
  + geom_density(alpha=0.2)  
)


# ### Number of posted projects by Unknown Gender distribution

# In[30]:


(ggplot(gender_based[gender_based.gender=='Not Specified'])
  + aes('count', fill='gender', color='gender')
  + ggtitle("Projects posted for Unknown ") 
  + geom_density(alpha=0.2)  
)


# ___
# ## Schools <a class='anchor' id='schools'></a>
# This section will cover Schools dataset

# ### Partial View of data <a class='anchor' id='schools-data'></a>
# Lets see the structure of schools dataset

# In[31]:


schools.head()


# ### How many unique schools in the past 16 years
# Lets have a look how many schools registered on DonorChoose.org in the past 15 years and 8 months

# In[32]:


num_schools = len(pd.unique(schools['School ID']))
print('\x1b[1;31m' + 'We have {0}'.format(num_schools) + ' Schools \x1b[0m')
_ = venn2((num_schools,0,0), set_labels=('Schools',''))
_.get_label_by_id('01').set_text('')


# * We have 72993 Shools that are known on DonorsChoose.org with at least one or more projects

# ### Lets see the statistics for Shools Metro Types

# In[33]:


metro_types = schools['School Metro Type'].value_counts()
trace = go.Table(
    header=dict(values=['Metro Type','Total'],
                fill = dict(color='#cd5c5c'),
                align = ['left']),
    cells=dict(values=[metro_types.index, metro_types.values],
               fill = dict(color='#ffb6c1'),
               align = ['left']))

data = [trace] 
ply.iplot(data, filename = 'metro_table')


# We have 
# * <span style="color:red">22992</span> schools located in suburban metro  
# * <span style="color:green">22793</span> schools located in urban metro
# * <span style="color:blue"> 12973</span> schools located in rural metro
# * <span style="color:violet"> 8125</span> schools located in Unknown metro
# * <span style="color:brown"> 6110</span> schools located in town metro  

# ### Visual representation of School Metro Types

# In[34]:


X=schools['School Metro Type'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='school_pie_chart')


# We have 
# * <span style="color:#000066">31.5%</span> schools located in suburban
# * <span style="color:orange">31.2%</span> schools located in urban
# * <span style="color:green">17.8%</span> schools located in rural
# * <span style="color:red">11.1%</span> schools which are on unknown metro type
# * <span style="color:violet">8.37%</span> schools located in town 

# In[35]:


(ggplot(schools) + 
   aes(x='School Metro Type', fill='School Metro Type') +
   geom_bar() + 
   ggtitle("Number of Schoos by Metro type") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# ### School Percentage Free Lunch

# In[36]:


(ggplot(schools.dropna()) 
  + aes(x='School Percentage Free Lunch', fill='School Metro Type')
  + ggtitle("Distribution of School Percentage Free Lunch ") 
  + geom_histogram(binwidth=5)
)


# In[37]:


sns.kdeplot(schools['School Percentage Free Lunch'].dropna())
plt.title('School Percentage Free Lunch distribution')
plt.xlabel('Number of schools')
plt.ylabel('Percentage')
plt.show()


# In[38]:


sns.kdeplot(schools['School Percentage Free Lunch'].dropna(), cumulative=True)
plt.title('School Percentage Free Lunch - Cummulative distribution')
plt.xlabel('Number of schools')
plt.ylabel('Percentage')
plt.show()


# In[39]:


schools[schools['School Metro Type'] == 'rural']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 rural Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")


# * Top 10 disctricts in rural metro with schools on DonorChoose.org

# In[40]:


schools[schools['School Metro Type'] == 'urban']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Urban Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")


# * Top 10 disctricts in Urban metro with schools on DonorChoose.org

# In[41]:


schools[schools['School Metro Type'] == 'suburban']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Suburban Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")


# * Top 10 disctricts in Suburban metro with schools on DonorChoose.org

# In[42]:


schools[schools['School Metro Type'] == 'town']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Town Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")


# * Top 10 disctricts in rural metro with schools on DonorChoose.org

# In[43]:


schools[schools['School Metro Type'] == 'unknown']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Unknown Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")


# * Top 10 disctricts in Unknown metro with schools on DonorChoose.org

# In[44]:


schools['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Overall Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")


# * Top 10 disctricts in All metros with schools on DonorChoose.org

# ### Top 10 Cities with schools on DonorsChoose.org

# In[45]:


schools['School City'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Cities")
plt.ylabel("Number of schools")
plt.xlabel("City Name")


# ### Top 10 Counties with schools on DonorChoose.org

# In[46]:


schools['School County'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Counties")
plt.ylabel("Number of schools")
plt.xlabel("County Name")


# ### Which States are schools located in?

# In[47]:


(ggplot(schools) + 
   aes(x='School State', fill='School State') +
   geom_bar() + 
   coord_flip() +
   ggtitle("Number of Schoos by School State") +
   theme(figure_size=(12, 20), axis_text_x=element_text(rotation=90, hjust=1))
)


# * All 51 states have at least one school project registered on DonorsChoose.org, California is the leading State and is expected since is near Sillicon Valley, The schools push technology forward 

# ### Which State has more schools

# In[48]:


states = schools['School State'].value_counts()
x = 0.
y = 0.
width = 50.
height = 50.
type_list = states.index
values = states.values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = ['#41B5A3','#FFAF87','#FF8E72','#ED6A5E','#377771','#E89005','#C6000D','#000000','#05668D','#028090','#9FD35C','#02C39A','#F0F3BD','#41B5A3','#FF6F59','#254441','#B2B09B','#EF3054','#9D9CE8','#0F4777','#5F67DD','#235077','#CCE4F9','#1748D1','#8BB3D6','#467196','#F2C4A2','#F2B1A4','#C42746','#330C25'] * 2
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x'] + r['dx'], 
            y1 = r['y'] + r['dy'],
            line = dict(width = 1),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x'] + (r['dx']/2),
            y = r['y'] + (r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x'] + (r['dx']/2) for r in rects ], 
    y = [ r['y'] + (r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height = 1000, 
    width = 1000,
    xaxis = dict(showgrid=False,zeroline=False),
    yaxis = dict(showgrid=False,zeroline=False),
    shapes = shapes,
    annotations = annotations,
    hovermode = 'closest',
    font = dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
ply.iplot(figure, filename='state-treemap')


# * **California** is the leading, 
# * **Texas** being the second and 
# * **New York** being the third

# ### Maps 
# ___
# We are not provided with coordiantes of the schools so we will have to work that out ourselves 

# In[49]:


# Get the coordinates of the states from the link using pandas 
usa_codes = [['AL', 'Alabama'],
       ['AK', 'Alaska'], 
       ['AZ', 'Arizona'],
       ['AR', 'Arkansas'],
       ['CA', 'California'],
       ['CO', 'Colorado'],
       ['CT', 'Connecticut'],
       ['DE', 'Delaware'],
       ['FL', 'Florida'],
       ['GA', 'Georgia'],
       ['HI', 'Hawaii'],
       ['ID', 'Idaho'],
       ['IL', 'Illinois'],
       ['IN', 'Indiana'],
       ['IA', 'Iowa'],
       ['KS', 'Kansas'],
       ['KY', 'Kentucky'],
       ['LA', 'Louisiana'],
       ['ME', 'Maine'],
       ['MD', 'Maryland'],
       ['MA', 'Massachusetts'],
       ['MI', 'Michigan'],
       ['MN', 'Minnesota'],
       ['MS', 'Mississippi'],
       ['MO', 'Missouri'],
       ['MT', 'Montana'],
       ['NE', 'Nebraska'],
       ['NV', 'Nevada'],
       ['NH', 'New Hampshire'],
       ['NJ', 'New Jersey'],
       ['NM', 'New Mexico'],
       ['NY', 'New York'],
       ['NC', 'North Carolina'],
       ['ND', 'North Dakota'],
       ['OH', 'Ohio'],
       ['OK', 'Oklahoma'],
       ['OR', 'Oregon'],
       ['PA', 'Pennsylvania'],
       ['RI', 'Rhode Island'],
       ['SC', 'South Carolina'],
       ['SD', 'South Dakota'],
       ['TN', 'Tennessee'],
       ['TX', 'Texas'],
       ['UT', 'Utah'],
       ['VT', 'Vermont'],
       ['VA', 'Virginia'],
       ['WA', 'Washington'],
       ['WV', 'West Virginia'],
       ['WI', 'Wisconsin'],
       ['WY', 'Wyoming']]
us_states = pd.DataFrame(data=usa_codes, columns=['Code', 'State'])


# * We are representing data on maps, lets get to work 

# In[50]:


us_states = us_states.rename({'State': 'School State'})
counts = pd.DataFrame({'State': schools['School State'].value_counts().index, 
                       'Total': schools['School State'].value_counts().values})
maps_df = counts.merge(us_states, on='State', how='inner')


# In[51]:


maps_df['text'] = maps_df['State'] + '<br>  ' + (maps_df['Total']).astype(str)+' donations'
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = maps_df['Code'],
        z = maps_df['Total'].astype(float),
        locationmode = 'USA-states',
        text = maps_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(
        title = 'DonorsChoose.org Donations <br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
ply.iplot( fig, filename='d3-cloropleth-map' )


# ___
# ## Projects <a class='anchor' id='projects'></a>
# This section will cover projects dataset

# ### Partial View of Data <a class='anchor' id='projects-data'></a>
# Lets see the structure of teacher dataset

# In[52]:


projects.head(2)


# ### Lets see total Projects that we have

# In[53]:


print("\x1b[1;31m We have {0} records \x1b[0m ".format(projects.shape[0]))
min_date = pd.to_datetime(projects['Project Fully Funded Date']).min()
max_date = pd.to_datetime(projects['Project Fully Funded Date']).max()
num_projects = len(pd.unique(projects['Project ID']))
print('\x1b[1;31m' + ' From {0} to {1} we have '.format(min_date, max_date) + '\x1b[0m')
_ = venn2((num_projects, 0, 0), set_labels=('Projects',''))
_.get_label_by_id('01').set_text('')


# ### How many projects Are fully funded or not (Visually)

# In[54]:


(ggplot(projects) + 
   aes(x='Project Current Status', fill='Project Current Status') +
   geom_bar() + 
   ggtitle("Number of Projects by Project Current Status") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# ### Projects Type

# In[55]:


(ggplot(projects) + 
   aes(x='Project Type', fill='Project Type') +
   geom_bar() + 
   ggtitle("Number of Projects by Type") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# ### Projects Set

# In[56]:


expired = len(projects[projects["Project Current Status"]=='Expired'])
fully = len(projects[projects["Project Current Status"]=='Fully Funded'])
live = len(projects[projects["Project Current Status"]=='Live'])


# ### The set represented visually

# In[57]:


_ = venn2((expired,fully,live), set_labels=('Expired','Fully Funded', 'Live'))


# We have 
# * <span style="color:green">826764</span>  Fully Funded projects
# * <span style="color:red">241402</span> Expired projects 
# * <span style="color:blue">41851</span> Live Projects  
# More Projects are fully funded 

# In[58]:


X=projects['Project Current Status'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_gender')


# We have 
# * <span style="color:blue">74.5%</span> Fully funded projects
# * <span style="color:orange">21.7%</span> Expired projects
# * <span style="color:green">3.77%</span> Live Projects

# ### Project Resource Category

# In[59]:


(ggplot(projects.dropna()) + 
   aes(x='Project Resource Category', fill='Project Resource Category') +
   geom_bar() + 
   coord_flip() + 
   ggtitle("Number of Projects by Project Resource Category") +
   theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(12, 10))
)


# We have highlights of three Projects resource category
# * **Supplies**
# * **Techonology** (as expected)
# * and **Books** beign the third highest 

# In[60]:


(ggplot(projects.dropna()) + 
   aes(x='Project Grade Level Category', fill='Project Grade Level Category') +
   geom_bar() + 
   coord_flip() + 
   ggtitle("Number of Projects by Grade ") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# * Teachers are creating projects on very lower grades to teach learners about Technology and reading Books

# In[61]:


X=projects['Project Grade Level Category'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_grade')


# We have 
# * <span style="color:blue">38.9%</span> projects for Grades Prek - 2 
# * <span style="color:orange">32.8%</span> projects for Grades 3 - 5 
# * <span style="color:green">16.3%</span> projects for Grades 6 -8
# * <span style="color:red">11.9%</span> projects for Grades 9 - 12
# * <span style="color:violet">0.00477%</span> projects for unknown Grades 

# ### Number of Projects by Subjects 

# In[62]:


(ggplot(projects.dropna()) + 
   aes(x='Project Subject Category Tree', fill='Project Subject Category Tree') +
   geom_bar() + 
   coord_flip() + 
   ggtitle("Number of Projects by Subject ") +
   theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(10, 30))
)


# ### Teachers Project Posted Sequence Distribution

# In[63]:


sns.kdeplot(projects['Teacher Project Posted Sequence'].dropna())
plt.title('Teachers Posted Sequence distribution')


# ### Projects Cost Distribution

# In[64]:


sns.kdeplot(projects['Project Cost'], cumulative=True)
plt.xlabel('Project Cost')
plt.ylabel('Percentage of projects')
plt.show()


# * <span style="color:green">96%</span> of Projects cost less than <span style="color:blue">10000.00 dollars</span>

# ## Time series 

# In[65]:


project_date = 'Project Posted Date'
cost = 'Project Cost'

projects_cost = projects[[project_date, cost]].groupby([project_date]).agg({cost:'sum'}).reset_index()
projects_cost[project_date] = pd.to_datetime(projects_cost[project_date])
projects_cost = projects_cost.set_index(project_date)
max_date = max(projects_cost.index)

ax = projects_cost.plot(color='blue')
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Project Cost in dollars ($)')
ax.set_title('Projects Cost by Date')
plt.show()


# * For projects dataset, we have data from <span style="color:blue">2013</span> - <span style="color:red">2018 May</span>, The graph above shows projects created daily with Total Amount for all projects created on that date. <span style="color:red">2018</span> with the highest cost of projects created daily, though the data is incomplete for <span style="color:red">2018</span>. It shows that platforms is used by many teachers. But that does not mean all projects that are created are fully funded, so the numbers maybe different depending on Project Status.

# In[66]:


projects_cost['Year'] = projects_cost.reset_index()[project_date].dt.year
per_year = projects[[project_date, cost]].groupby(project_date).agg({cost:'sum'}).reset_index()
per_year[project_date] = pd.to_datetime(per_year[project_date])
per_year['Year'] = per_year[project_date].dt.year

ax = per_year[['Year', cost]].groupby('Year').agg({cost:'sum'}).plot(color='blue', style='.-')
ax.axhspan(200000, 250000, color='green', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Project Cost in dollars ($)')
ax.set_title('Projects Cost by Year')
plt.show()


# * Projects costs by year shows increase in amount of projects created each year, which makes sense since resources are increasing in prices and also population taking place at hand, the current year may exceed the year<span style="color:red"> 2017</span>

# In[67]:


grade = 'Project Grade Level Category'
grade_based = projects[[project_date, grade, cost]].groupby([project_date, grade]).agg({cost:'sum'}).reset_index()
grade_based[project_date] = pd.to_datetime(grade_based[project_date])

(ggplot(grade_based) 
 + aes(project_date, cost, color='Project Grade Level Category') 
 + geom_line()
 + ggtitle("Projects Cost by Date - Grade Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# In[68]:


fully = 'Project Fully Funded Date'
grade_based = projects[[fully, grade, cost]].groupby([fully, grade]).agg({cost:'sum'}).reset_index()
grade_based[fully] = pd.to_datetime(grade_based[fully])

(ggplot(grade_based.dropna()) 
 + aes(fully, cost, color='Project Grade Level Category') 
 + geom_line()
 + ggtitle("Projects Cost by Fully Funded Date - Grade Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# In[69]:


projects[project_date] = pd.to_datetime(projects[project_date])
projects['Year'] = projects[project_date].dt.year
grade_based_year = projects[['Year', grade, cost]].groupby(['Year', grade]).agg({cost:'sum'}).reset_index()

(ggplot(grade_based_year) 
 + aes('Year', cost, color='Project Grade Level Category') 
 + geom_point()
 + geom_line()
 + ggtitle("Projects Cost by Year - Grade Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# In[70]:


status = 'Project Current Status'
projects[project_date] = pd.to_datetime(projects[project_date])
projects['Year'] = projects[project_date].dt.year
grade_based_year = projects[['Year', status, cost]].groupby(['Year', status]).agg({cost:'sum'}).reset_index()

(ggplot(grade_based_year) 
 + aes('Year', cost, color=status) 
 + geom_point()
 + geom_line()
 + ggtitle("Projects Cost by Year - Status Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# In[71]:


p_type = 'Project Type'
projects[project_date] = pd.to_datetime(projects[project_date])
projects['Year'] = projects[project_date].dt.year
grade_based_year = projects[['Year', p_type, cost]].groupby(['Year', p_type]).agg({cost:'sum'}).reset_index()

(ggplot(grade_based_year) 
 + aes('Year', cost, color=p_type) 
 + geom_point()
 + geom_line()
 + ggtitle("Projects Cost by Year - Project Type ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# ### Stopwords 

# In[72]:


stop_words = set(STOPWORDS)
new_words = ("DONOTREMOVEESSAYDIVIDER", "go", "bk")
new_stops = stop_words.union(new_words)


# In[73]:


wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Need Statement'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')


# In[74]:


wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Title'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')


# In[75]:


wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Essay'].sample(50000).astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')


# In[76]:


wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Short Description'].sample(100000).astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')


# ___
# ## Resources <a class='anchor' id='resource'></a>
# This section Covers resources dataset

# ### Partial View of Data <a class='anchor' id='resources-data'></a>
# Lets see the structure of schools dataset

# In[77]:


resources.head()


# ### Which Vendor are providing with the resources

# In[78]:


vendor_resource = resources["Resource Vendor Name"].value_counts()
trace = go.Table(
    header=dict(values=['Vendor Name','Total Resources'],
                fill = dict(color='#cd5c5c'),
                align = ['left']),
    cells=dict(values=[vendor_resource.index, vendor_resource.values],
               fill = dict(color='#ffb6c1'),
               align = ['left']))

data = [trace] 
ply.iplot(data, filename = 'metro_table')


# In[ ]:


(ggplot(resources.dropna()) + 
   aes(x="Resource Vendor Name", fill="Resource Vendor Name") +
   geom_bar(size=20) + 
   coord_flip() +
   ggtitle("Number of elements by Phase") +
   theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(12, 20))
)


# ### Lets see Number of resources that are bought

# In[ ]:


sns.kdeplot(resources['Resource Quantity'].dropna(), cumulative=True)
plt.title("Distribution of Resource quatity")


# ### Lets see how resources are described 

# In[ ]:


wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(resources['Resource Item Name'].sample(60000).astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')


# ___
# ## Donors <a class='anchor' id='donors'></a>
# This section will cover the Donors dataset

# ### Partial View of Data <a class='anchor' id='donors-data'></a>
# Lets see the structure of donors dataset

# In[ ]:


donors.head()


# ### Donor is a Teacher/Not a Teacher

# In[ ]:


X=donors['Donor Is Teacher'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_grade')


# We have 
# * <span style="color:blue">90%</span> Donors that are not teachers
# * <span style="color:orange">10%</span> Donors that are teachers

# In[ ]:


(ggplot(donors) + 
   aes(x='Donor Is Teacher', fill='Donor Is Teacher') +
   geom_bar() + 
   ggtitle("Number of Donors that are Teachers and Not Teachers") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# ### Top 10 Cities with most donors 

# In[ ]:


donors['Donor City'].value_counts()[:10].plot(kind='bar')
plt.title('City by number of Donors')
plt.xlabel('City Name')
plt.ylabel('Number of Donors')


# ### Top 10 States with most Donors

# In[ ]:


donors['Donor State'].value_counts()[:10].plot(kind='bar')
plt.title('State by number of Donors')
plt.xlabel('State')
plt.ylabel('Number of Donors')


# ### States with most Donors

# In[ ]:


# Get the coordinates of the states from the link using pandas 
counts = pd.DataFrame({'State': donors['Donor State'].value_counts().index, 
                       'Total': donors['Donor State'].value_counts().values})
maps_df = counts.merge(us_states, on='State', how='inner')

maps_df['text'] = maps_df['State'] + '<br>  ' + (maps_df['Total']).astype(str)+' donations'
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = maps_df['Code'],
        z = maps_df['Total'].astype(float),
        locationmode = 'USA-states',
        text = maps_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(
        title = 'DonorsChoose.org Donors <br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
ply.iplot( fig, filename='d3-cloropleth-map' )


# * **California** has more donors than any other states in the US. It has many philanthropist.

# ___
# ## Donations <a class='anchor' id='donations'></a>
# This section will cover the donations dataset

# 
# ### Partial View of Data <a class='anchor' id='donations-data'></a>
# Lets see the structure of donations dataset

# In[ ]:


donations.head()


# In[ ]:


print(donations['Donation Amount'].min())
print(donations['Donation Amount'].max())


# * People donate with as little as <span style="color:green">0.01</span> cents to a maximum of  <span style="color:green">60000.00</span> dollars

# ### Lets see the number of donations that included additional donations and those that haven't

# In[ ]:


X=donations['Donation Included Optional Donation'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_grade')


# We have 
# * <span style="color:blue">85.4%</span> Included optional donation 
# * <span style="color:orange">14.6%</span> did not Included optional donation 

# In[ ]:


(ggplot(donations) + 
   aes(x='Donation Included Optional Donation', fill='Donation Included Optional Donation') +
   geom_bar() + 
   ggtitle("Number of Donations included additional donations/not") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)


# ### Lets extract meaning from time

# In[ ]:


donations['Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Hour'] = donations['Date'].dt.hour


# In[ ]:


donations.Hour.plot(kind='density')


# ### Lets see what time in hours most people Donate

# In[ ]:


donations.Hour.value_counts().sort_index().plot(kind='bar')
plt.title('Donations by Time in Hours')
plt.ylabel('Number of Donations')
plt.xlabel('Hour')


# ### Hourly donation distributions 

# In[ ]:


sns.distplot(donations.Hour.value_counts(sort=True).values, hist=False, norm_hist=True)


# * This is bimodal distributions it has two means with the same variance

# In[ ]:


donation_date = 'Donation Received Date'
options = 'Donation Included Optional Donation'
donations[donation_date] = pd.to_datetime(donations[donation_date])
donations['Date'] = donations[donation_date].dt.date
donations_time_based = donations[['Date', options, 'Donation Amount']].groupby(['Date', options]).agg({'Donation Amount':'count'}).plot()


# In[ ]:


#Donation Included Optional Donation 	Donation Amount 	Donor Cart Sequence

donation_date = 'Donation Received Date'
options = 'Donation Included Optional Donation'
donations[donation_date] = pd.to_datetime(donations[donation_date])
donations['Year'] = donations[donation_date].dt.year
donations_time_based = donations[['Year', options, 'Donation Amount']].groupby(['Year', options]).agg({'Donation Amount':'count'}).reset_index()

(ggplot(donations_time_based)
 + aes('Year', 'Donation Amount', color=options) 
 + geom_point()
 + geom_line()
 + ggtitle("Donation Amount by Year") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)


# ___
# # Machine Learning <a class='anchor' id='machine-learning'></a>

# In[ ]:


from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, train_test_split, validation_curve
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF 
from sklearn.metrics import accuracy_score, auc, classification_report, mean_squared_error, roc_curve


# In[ ]:


## Comming soon


# ___
# # Naratives

# According to https://stats.oecd.org the gender distribution for teachers is <span style="color:green">87.1%</span> females and <span style="color:blue">12.9%</span> males. The anlysis shows that we have <span style="color:green">86.4%</span> female teachers and <span style="color:blue">11.8%</span> male teachers while the rest is unspecified. This indicates that male and female teachers are more or less equally involved in submitting projects on DonorsChoose.org.
# 
# The time series graphs show that donations are nomarly distributed through each semester. This suggests that there is more donations towards the middle of each semester which might be an interesting time to suggest projects for donors. And graphs also show that there is more donations on average on the second semester of each year than the first semester. 
# 
# We have 
# * <span style="color:blue">85.4%</span> Included optional donation 
# * <span style="color:orange">14.6%</span> did not Included optional donation 
# 
# * People donate with as little as <span style="color:green">0.01</span> cents to a maximum of  <span style="color:green">60000.00</span> dollars 
# * **California** has more donors than any other states in the US. It has many philanthropist.
# 
# * Projects costs by year shows increase in amount of projects created each year, which makes sense since resources are increasing in prices and also population taking place at hand, the current year may exceed the year<span style="color:red"> 2017</span>
# 
# * For projects dataset, we have data from <span style="color:blue">2013</span> - <span style="color:red">2018 May</span>, The graph above shows projects created daily with Total Amount for all projects created on that date. <span style="color:red">2018</span> with the highest cost of projects created daily, though the data is incomplete for <span style="color:red">2018</span>. It shows that platforms is used by many teachers. But that does not mean all projects that are created are fully funded, so the numbers maybe different depending on Project Status.
# 
# We have 
# * <span style="color:#000066">31.5%</span> schools located in suburban
# * <span style="color:orange">31.2%</span> schools located in urban
# * <span style="color:green">17.8%</span> schools located in rural
# * <span style="color:red">11.1%</span> schools which are on unknown metro type
# * <span style="color:violet">8.37%</span> schools located in town 
# 
# * **California** is the leading, 
# * **Texas** being the second and 
# * **New York** being the third
# 
# * From 2002 to this year we have 402900 teachers that registered their first projects on DonorsChoose.org
# 
# * Red Bars indicates incomplete years in <span style="color:#F08080">2002</span> and this current year <span style="color:#F08080">2018</span>, as it indicates on the graph above at the end of <span style="color:#000066">2010</span> the DonorsChoose.org started having increase in the number of teachers posting their first projects 
# * On 27 October 2013 DonorsChoose.org started having more than 500 new projects created by more than 500 new teachers  
# * From above diagram, the green shaded bar period it shows the most registered projects per day exceeding <span style="color:#006600">2000</span> by teachers who post their first projects with <span style="color:#F08080">86%</span> chance that are ladies   
# * Slicing the data shows the nice flow between <span style="color:red">2013</span> and <span style="color:red">2018</span> with a potential in <span style="color:red">2018</span> to exceed <span style="color:green">Total yearly posted projects </span> breaking the record for the year <span style="color:red">2016</span>
# * In <span style="color:red">2002</span> there was not much happening, the maximum number of projects posted were <span style="color:red">13</span> for the whole year, well the platform was new and take it as a complement for that year and most importantly the world was not connected as of today. Social Media was not matured. 
# * The world today is more connected making the platform more efficiently to be used by many teachers, and social media is making teachers more connected and help the students to learn new tools and have more efficient ways to teach students at schools, <span style="color:red">2018</span> is already <span style="color:red">4</span> month old with this dataset and it is already making a good progress. 
# * Yearly figure shows increase in the creation on new projects each year with a slightly decrease in <span style="color:red">2017</span> from <span style="color:red">2016</span>.
# * Females are actively posting their first project more than Males teachers, teachers who did not specify their gender being third and creating their first projects after a while
