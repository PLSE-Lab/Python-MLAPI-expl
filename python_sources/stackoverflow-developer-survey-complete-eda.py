#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Stack Overflow is a privately held website, the flagship site of the Stack Exchange Network created in 2008 by Jeff Atwood and Joel Spolsky. It was created to be a more open alternative to earlier question and answer sites such as Experts-Exchange. The name for the website was chosen by voting in April 2008 by readers of Coding Horror, Atwood's popular programming blog.
# It features questions and answers on a wide range of topics in computer programming. 
# 
# Every year Stack Overflow asks the developer community about everything from their favorite technologies to their job preferences. This year also they have published their Annual Developer Survey results. The survey results have been made available on kaggle on which people can do analysis. 

# I will be doing some exploratory analysis of the data. Please let me know if you have any suggestions. 

# # Loading the Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
color = sns.color_palette()

import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

import cufflinks as cf
cf.go_offline()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Now, we will read the data file

# # Reading the input dataset

# In[3]:


df = pd.read_csv("../input/survey_results_public.csv")
print("Number of rows in the dataset are : {}".format(len(df)))


# Now, we will have a look at the top 5 rows of the data

# In[4]:


df.head()


# In[5]:


print("Number of columns in the input dataset : {}".format(len(df.columns)))


# # Exploratory Analysis

# We will move through the dataset and explore what the data says

# # 1) What are the countries from which most of the developers participated in the survey?? 

# In[6]:


country_dist = df['Country'].value_counts().head(6)
country_dist.iplot(kind='bar', xTitle='Country Name', yTitle='Num of developers', title='Most number of developers from these countries')


# So, we can see that most developers which participated are from USA, the followed by India and Germany. 

# # 2) How many developers contribute to open source projects?

# In[7]:


open_source_cnts = df['OpenSource'].dropna().value_counts()
df_new = pd.DataFrame({
    'label':open_source_cnts.index, 
    'values':open_source_cnts.values
})
df_new.iplot(kind='pie', labels='label', values='values', title='Percent of developers contributing to open source', color=['#ffff00', '#b0e0e6'])


# So, we can see that a lot of developers indeed contribute to open source projects.

# # 3) Is the developers job education dependent ??

# In[8]:


df['FormalEducation'].isnull().sum().sum()


# We can see 4152 developers have not filled their FormalEducation value. Formal Education stand for the highest level of formal education that the developer has completed..

# In[9]:


edu_cnts = df['FormalEducation'].dropna().value_counts()
edu_cnts / edu_cnts.sum() * 100


# So, we can see that 46% of the people have Bachelors degree. Also, Some people have never completed and formal education. SO, we can see that there is a variety of developers

# In[10]:


trace = go.Bar(
    y=edu_cnts.index[::-1],
    x=(edu_cnts/edu_cnts.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#00adff', '#f99372', '#fdded3', '#b0e0e6', '#ffff00', '#00fa9a', '#ffffcc', '#f2e6e9', '#fccbbb']
    ),
)

layout = dict(
    title='Level of Formal Education',
        margin=dict(
        l=500,
)
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # 4) What are the job profiles of developers? Let us explore that

# In[17]:


dev_type_sep = ";".join(df['DevType'].values.astype(str))
lst_dev = dev_type_sep.split(";")
lst_dev = pd.Series(lst_dev)
lst_dev = lst_dev.value_counts().head(7)


# In[18]:


def get_colors(n_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(n_colors)]
    return color


# In[19]:


trace0 = [
    go.Bar(
        x=lst_dev.index,
        y=(lst_dev / lst_dev.sum() * 100),
        marker=dict(color=get_colors(len(lst_dev)))
    )]

layout = go.Layout(
    title='Different Job profiles of developers',
    xaxis=dict(title='Job Profiles of Developers'),
    yaxis=dict(title='Number of developers')
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, we can see that most of the developers are back-end developers followed by full-stack and front-end developers.

# # 5) Are the developers satisfied with their jobs ??

# In[20]:


df['JobSatisfaction'].value_counts()


# In[21]:


sat_levels = df['JobSatisfaction'].value_counts()
trace0 = [
    go.Bar(
        x=sat_levels.index,
        y=sat_levels / sat_levels.sum() * 100,
        marker=dict(color=get_colors(len(sat_levels)))
    )]

layout = go.Layout(
    title='Job satisfaction level of developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, around 37% developers are moderately satisifed, while 18% are extremely satisfied. 

# # 6) What do the developers hope to do in the next five years ??

# In[23]:


df['HopeFiveYears'].dropna().value_counts()


# In[24]:


hope_five_yrs = df['HopeFiveYears'].value_counts()
hope_five_yrs.iplot(kind='bar', title='What developers hope to do in next 5 years', colors=get_colors(len(hope_five_yrs)))


# So, many of the developers  want to work in a different or more specialized technical role than they are in now. Also, may deveopers hope to work as a founder or their own company. 

# # 7) How many times the developers exercise within a typical week ??

# In[26]:


df['Exercise'].value_counts()


# In[27]:


exercise_num = df['Exercise'].value_counts()
df_new = pd.DataFrame({
    'label':exercise_num.index, 
    'values':exercise_num.values
})
df_new.iplot(kind='pie', labels='label', values='values', title='Times developers do exercise in a typical week', color=get_colors(len(sat_levels)))


# # 8) What are the languages that the developers have worked with?? 

# In[28]:


lang_used = ";".join(df['LanguageWorkedWith'].dropna().values.astype(str))
lst_lang = lang_used.split(";")
lst_lang = pd.Series(lst_lang)
lst_lang = lst_lang.value_counts().head(10)


# In[29]:


trace0 = [
    go.Bar(
        x=lst_lang.index,
        y=lst_lang / lst_lang.sum() * 100,
        marker=dict(color=get_colors(len(lst_lang)))
    )]

layout = go.Layout(
    title='Languages that developers have worked with',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, we can see that JavaScript, HTML, and CSS are the top 3 languages developers have worked with.

# # 9) What are the languages developers want to work on in the next year ?? 

# In[31]:


lang_used = ";".join(df['LanguageDesireNextYear'].dropna().values.astype(str))
lst_lang = lang_used.split(";")
lst_lang = pd.Series(lst_lang)
lst_lang = lst_lang.value_counts().head(10)


# In[32]:


trace0 = [
    go.Bar(
        x=lst_lang.index,
        y=lst_lang / lst_lang.sum() * 100,
        marker=dict(color=get_colors(len(lst_lang)))
    )]

layout = go.Layout(
    title='Languages that developers want to work with in the next year',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, may of the developers want to work with Python on the next year. This may be because of the vast use of python in data analytics which is increasing at a very fast rate nowadays

# # 10) What are the databases that the developers have worked with ??

# In[33]:


db_used = ";".join(df['DatabaseWorkedWith'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)


# In[34]:


trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)))
    )]

layout = go.Layout(
    title='Databases that the developers have worked with',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, MYSQL, SQL Server, PostGre SQL tops this list for the top 3 contenders

# # 11) What are the databases developers want to work with next year ??

# In[35]:


db_used = ";".join(df['DatabaseDesireNextYear'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)


# In[36]:


trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)))
    )]

layout = go.Layout(
    title='Databases that developers want to work with in the next year.',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, MongoDB has risen up the ranks, this may be due to the popularity of NOSql Databases which is increasing in general

# # 12) Let us see the libraries, frameworks and tools developers have worked with 

# In[37]:


db_used = ";".join(df['FrameworkWorkedWith'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)


# In[38]:


trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Frameworks that developers have worked with.',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# # 13)  Let us see the libraries, frameworks and tools developers want to work with in the next year. 

# In[39]:


db_used = ";".join(df['FrameworkDesireNextYear'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)


# In[40]:


trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Frameworks that developers want to work with in the next year.',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, we can see that the react and tensorflow popularity has increased and more people want to use these in the next year

# # 14) Let us see what IDEs are the most common among developers

# In[42]:


ide_used = ";".join(df['IDE'].dropna().values.astype(str))
lst_ide = ide_used.split(";")
lst_ide = pd.Series(lst_ide)
lst_ide = lst_ide.value_counts().head(10)


# In[43]:


trace0 = [
    go.Bar(
        x=lst_ide.index,
        y=lst_ide / lst_lang.sum() * 100,
        marker=dict(color=get_colors(len(lst_ide)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Most Common IDEs among developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# # 15) Let us now see what are the operating systems used by the developers 

# In[44]:


os_used = df['OperatingSystem'].value_counts()


# In[45]:


df_new = pd.DataFrame(
{
    'label': os_used.index,
    'values': os_used.values
})
df_new.iplot(kind='pie', labels='label', values='values',title='Operating_system used', colors=get_colors(len(os_used)))


# So, almost 50% of the developers use windows based operating system.

# # 16) Let us see the median salary in different currencies

# In[47]:


df_new = df[['Currency', 'Salary', 'SalaryType']]
# We will drop the rows which have null values
df_new = df_new.dropna()
df_new['Salary'] = [x.replace(",", "") for x in df_new['Salary']]
df_new['Salary'] = [float(x) for x in df_new['Salary'].values.astype(str)]
df_new = df_new[df_new['Salary'] != 0]
index_monthly = df_new[df_new['SalaryType'] == 'Monthly'].index
df_new.loc[index_monthly, 'Salary'] = df_new.loc[index_monthly, 'Salary'] * 12
index_weekly = df_new[df_new['SalaryType'] == 'Weekly'].index
df_new.loc[index_weekly, 'Salary'] = df_new.loc[index_weekly, 'Salary'] * 52


# In[48]:


df_new.groupby('Currency').median()


# So, median salary in India is 6 lakhs rupees, while in USA it is 90000 dollars, In Europe it is 40,000 euros

# # 17) Let us also have a look at the age distribution of the developers 

# In[50]:


df['Age'].dropna().value_counts()


# In[51]:


lst_age = df['Age'].dropna().value_counts()


# In[52]:


trace0 = [
    go.Bar(
        x=lst_age.index,
        y=lst_age / lst_age.sum() * 100,
        marker=dict(color=get_colors(len(lst_age)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Age Distribution of developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, almost 50% of the developers are in the age-group 25-34 years. 

# # 18) Let us find a correlation between age of the developers and whether they code as a hobby

# In[54]:


df_new = df[['Age', 'Hobby']]
df_new = df_new.dropna()
df_new['Hobby'] = [1 if x=='Yes' else 0 for x in df_new['Hobby']]
lst = df_new.groupby('Age').describe()
def func(row):
    return row['Hobby']['mean']*row['Hobby']['count']

lst['vals'] = lst.apply(func, axis=1)


# In[55]:


trace1 = go.Bar(
    x=lst.index,
    y=lst['Hobby']['count'],
    name='Count of the Deveopers'
)

trace2 = go.Bar(
    x=lst.index,
    y=lst['vals'],
    name='Count of those who code as a hobby'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# So, we can see that as the age increases, less and less developers code as  hobby, while the developers who are under the age of 18 are almost the same as those who code as a hobby.

# # 19) Also how age affects open source contribution

# In[57]:


df_new = df[['Age', 'OpenSource']]
df_new = df_new.dropna()
# df_new.head()
df_new['OpenSource'] = [1 if x=='Yes' else 0 for x in df_new['OpenSource']]
lst = df_new.groupby('Age').describe()
# lst.head()
def func(row):
    return row['OpenSource']['mean']*row['OpenSource']['count']

lst['vals'] = lst.apply(func, axis=1)


# In[58]:


trace1 = go.Bar(
    x=lst.index,
    y=lst['OpenSource']['count'],
    name='Count of the Developers'
)

trace2 = go.Bar(
    x=lst.index,
    y=lst['vals'],
    name='Count of those who do open source contributions'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # 20) Let us see how much time do developers spend in front of a computer

# In[59]:


df['HoursComputer'].value_counts()


# In[60]:


lst = df['HoursComputer'].value_counts()
trace0 = [
    go.Bar(
        x=lst.index,
        y=lst / lst.sum() * 100,
        marker=dict(color=get_colors(len(lst)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Time developers spend in front of a computer',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, almost 50% of the developers spend 9-12 hours in front of a computer

# # 21) Let us now see the distribution of employees according to company size

# In[63]:


lst = df['CompanySize'].value_counts()
trace0 = [
    go.Bar(
        x=lst.index,
        y=lst / lst.sum() * 100,
        marker=dict(color=get_colors(len(lst)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Company size of the developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, we can see that a number of employees are working in company sizes ranging upto 500. This may be due to startup culture which is going on at a good rate

# # 22) Let us see what the developers think about the future of AI

# In[64]:


df['AIFuture'].value_counts()


# In[65]:


lst = df['AIFuture'].value_counts()
trace0 = [
    go.Bar(
        x=lst.index,
        y=lst / lst.sum() * 100,
        marker=dict(color=get_colors(len(lst)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Developers thoughts about the future of AI',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)


# So, around 70% of the developers are excited about the possibilities rather than being worried about its dangers

# More to come in this notebook. 

# In[ ]:




