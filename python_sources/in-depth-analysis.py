#!/usr/bin/env python
# coding: utf-8

# ## Complete EDA
# 
# **This Year, as well, people have shared their views about the Technology, their work status, how they learn and build their career, what they want to do in future.**<br>
# 
# This notebook is going to be the in-depth analysis of the data. So **Stay tuned with the work.**

# ### 1. Load Libraries

# In[1]:


import numpy as np
import pandas as pd 
import copy
import datetime as dt
import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
import missingno as msn

import warnings
warnings.filterwarnings('ignore')


# ### 2. Reading Files

# In[2]:


data = pd.read_csv('../input/survey_results_public.csv')
schema = pd.read_csv('../input/survey_results_schema.csv')


# ### 3. Glimpse of Data

# In[3]:


data.head()


# In[4]:


schema.head()


# ### 4. Geography
# <b>Which country's Respondents have participated in the Survey?</b>

# In[4]:


countries = data['Country'].value_counts()
trace = [ dict(
        type = 'choropleth',
        locations = countries.index,
        locationmode = 'country names',
        z = countries.values,
        text = countries.values,
        colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],\
                      [0.4, 'rgb(188,189,220)'], [0.6, 'rgb(158,154,200)'],\
                      [0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(190,190,190)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Participation'),
      ) ]
layout = dict(
    title = 'Participation in survey from different countries',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=trace, layout=layout )
py.iplot( fig, validate=False, filename='Survey participation' )


# Most of the Participation are from United Sates followed by India.

# ### 5. Developer Roles
# #### Developer Type

# In[6]:


def find_dev_type():
    dev_type_count = dict()
    for i in range(len(data)):
        types = data['DevType'][i]
        if isinstance(types, str):
            dev_type = types.split(';')
            for j in dev_type:
                if j not in dev_type_count:
                    dev_type_count[j] = 1
                else:
                    dev_type_count[j] += 1
    return dev_type_count



def get_dev_name_count():
    dev_type_name = []
    dev_type_percentage = []
    dev_type_count = find_dev_type()
    total_devs = sum(list(dev_type_count.values()))
    
    for key in dev_type_count:
        dev_type_count[key] = dev_type_count[key]/total_devs*100
    sorted_dev_type_count = sorted(dev_type_count.items(), key=lambda x: x[1])

    for i in range(len(sorted_dev_type_count)):
        dev_type_name.append(sorted_dev_type_count[i][0])
        dev_type_percentage.append(sorted_dev_type_count[i][1])
    
    return dev_type_name, dev_type_percentage


dev_type_name, dev_type_percentage = get_dev_name_count()
trace1 = go.Bar(
    x = dev_type_name,
    y  = dev_type_percentage
)

layout = go.Layout(
    title = 'Percentage of Developer Type',
    height = 400,
    xaxis = dict(
        tickfont = dict(
            size = 9
        )
    ),
    margin = dict(l = 50, r = 70, b = 150, t = 50, pad = 4)
)

trace = [trace1]
fig = go.Figure(data = trace, layout=layout)
py.iplot(fig)


# Amost 20% of Respondents are Back-end Developers, almost 16% are Full Stack Developers, 13% are front end Developers.

# #### How many Among total participants do coding as Hobby? How many contribute to open Source projects?

# In[5]:


coding_practices = pd.DataFrame()

coding_practices['Hobby'] = data['Hobby'].value_counts()
coding_practices['open source'] = data['OpenSource'].value_counts()

layout = go.Layout({
         'yaxis': {'title': 'Count'}
})

coding_practices.iplot(kind='bar', layout=layout, subplots=True, shape=(1,2),
                       subplot_titles=('Coding as Hobby?',\
                                       'Contribution to Open Source Projects?'))


# Around 80,000 People do coding as Hobby which means they love to code. But the ratio of contribution to open source projects v/s not contributing to open source projects is less than 1 as less people do contribution.

# #### Student V/s Not Student in Coding as Hobby !

# In[8]:


student_coding_hobby = data[data['Student'] != 'No']['Hobby'].value_counts()
not_student_coding_hobby = data[data['Student'] == 'No']['Hobby'].value_counts()

trace1 = go.Bar(
    x = student_coding_hobby.index,
    y = student_coding_hobby.values,
    name = 'Student'
)

trace2 = go.Bar(
    x = not_student_coding_hobby.index,
    y = not_student_coding_hobby.values,
    name = 'not student'
)

trace = [trace1, trace2]
layout = go.Layout(
    barmode = 'group',
    title = 'Coding as Hobby- Student v/s Not Student ?',
    yaxis = dict(
        title = 'Count'
    )
)

fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# From the graph, we observe that **the respondents, who are not students, do Coding as Hobby more than students.** These People can be professional Developers, Software engineers, etc.   

# #### Student V/s Not Student in Contribution to Open Source Projects !

# In[6]:


student_open_source_contri = data[data['Student'] != 'No']['OpenSource']                             .value_counts()
not_student_open_source_contri = data[data['Student'] == 'No']['OpenSource']                                 .value_counts()

trace1 = go.Bar(
    x = student_open_source_contri.index,
    y = student_open_source_contri.values,
    name = 'Student'
)

trace2 = go.Bar(
    x = not_student_open_source_contri.index,
    y = not_student_open_source_contri.values,
    name = 'not student'
)

trace = [trace1, trace2]
layout = go.Layout(
    barmode = 'group',
    title = 'Open Source Contribution- Student v/s Not Student ?',
    yaxis = dict(
        title = 'Count'
    )
)

fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# As similar to coding habits, **most of the people who contribute to open source projects are not students.**

# ### 6. Experience

# #### Years of Coding and Professional Coding Experience

# In[10]:


years_coding = data['YearsCoding'].value_counts()
yrs_prof_coding = data['YearsCodingProf'].value_counts()

trace0 = go.Bar(
    x = years_coding.values,
    y = years_coding.index,
    orientation = 'h'
)
trace1 = go.Bar(
    x = yrs_prof_coding.values,
    y = yrs_prof_coding.index,
    orientation ='h'
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Years of Coding Experience',
                                                        'Professional Coding experience'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=650, width=1000)
fig['layout']['yaxis1']['tickfont'].update(size=8)
fig['layout']['yaxis2']['tickfont'].update(size=8)
py.iplot(fig)


# ### 7. Education

# #### How many are Students?

# In[11]:


is_student = data['Student'].value_counts()
is_student.iplot(kind='bar', title='How many are Students?')


# Most of the Respondents for stackoverflow survey are not students. They may either be working or not working indivduals.

# #### What is their education ?

# In[12]:


formal_education = data['FormalEducation'].value_counts()
undergrad_major = data['UndergradMajor'].value_counts()

trace1 = {
    'x': ['Bachelors Degree', 'Master Degree', 'Study without degree',
          'Secondary School', 'Associate Degree', 'Doctoral Degree',
          'Primary school', 'Professional Degree',
          'Never completed education'],
    'y': formal_education.values
}

trace2 = {
    'x': ['Comp Engineering', 'Other Engineering', 'IT', 'Natural Science',
          'Maths & Stats', 'web dev & design', 'Business Discipline',
          'Humanities Discipline', 'Social Science',
          'Fine Arts', 'Not a major', 'Health Science'],
    'y': undergrad_major.values
}

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Formal Education',
                                                        'Area of Education'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=650, width=1000)
py.iplot(fig)


# **Formal Education**-<br>
# Most of the poeple have Bachlor's and Master's Degree.<br>
# **What is the area of study?**<br>
# The area of formal education of most of respondents is computer enginerring, engineering, etc.

# ### 8. Employment

# #### Occupation of Respondents!!

# In[13]:


employment = data['Employment'].value_counts()
layout = go.Layout(
    height = 400,
    font=dict(size=12, color='black'),
    title = 'Employment Status of Respondents!!',
    margin = dict(l = 50, r = 70, b = 150, t = 50, pad = 4),
    xaxis = dict(
        tickfont = dict(
            size = 9
        )
    )
)
employment.iplot(kind='bar', layout=layout)


# So almost Respondents are Full time employed with very few as Independent Contractors or Freelancers 

# #### Job and Career Satisfication

# In[14]:


job_satisfaction = data['JobSatisfaction'].value_counts()
career_satisfaction = data['CareerSatisfaction'].value_counts()

trace0 = go.Bar(
    x = job_satisfaction.values,
    y = job_satisfaction.index,
    orientation = 'h'
)
trace1 = go.Bar(
    x = career_satisfaction.values,
    y = career_satisfaction.index,
    orientation ='h'
)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Job Satisfaction',
                                                        'Career Satisfaction'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig['layout'].update(height=650, width=1000)
fig['layout']['yaxis1']['tickfont'].update(size=10)
fig['layout']['yaxis2']['tickfont'].update(size=10)
fig['layout']['margin'] = dict(l = 200, r = 70, b = 100, t = 50, pad = 4)
py.iplot(fig)


# Most of the People are moderately satisfied with their job and career. So they might be looking for more interesting roles to do in life.

# #### What People hope for next five years?

# In[15]:


hope_five_years = data['HopeFiveYears'].value_counts()
trace0 = go.Bar(
    x = hope_five_years.values,
    y = hope_five_years.index,
    orientation = 'h'
)
layout = go.Layout(
    height = 400,
    font=dict(size=12, color='black'),
    title = 'What people hope in the next five years!!',
    margin = dict(l = 410, r = 50, b = 50, t = 50, pad = 4),
    yaxis = dict(
        tickfont = dict(
            size = 10
        )
    )
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# As expected, in previous section, we have found that people are moderately satisfied with their career. The above plot shows their goals for the next five years in order to get satsfied with their career.<br>
# **People, mostly, want to get better role than where thay are in present or want to have their own company.**

# #### Job Search Status!

# In[16]:


job_search_status = data['JobSearchStatus'].value_counts()
trace0 = go.Bar(
    x = job_search_status.values,
    y = job_search_status.index,
    orientation = 'h'
)
layout = go.Layout(
    height = 400,
    font=dict(size=12, color='black'),
    title = 'Job Search Status!!',
    margin = dict(l = 300, r = 50, b = 50, t = 50, pad = 4),
    yaxis = dict(
        tickfont = dict(
            size = 9
        )
    )
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# As most of the respondents are employed or working individuals, so depicts the above analysis that **mostly people are not looking for job actively but they are open to new opportunities.**

# #### Number of years working in last job

# In[17]:


last_new_job = data['LastNewJob'].value_counts()
trace0 = go.Bar(
    x = last_new_job.values,
    y = last_new_job.index,
    orientation = 'h'
)
layout = go.Layout(
    height = 400,
    font=dict(size=12, color='black'),
    title = 'Last New Job!!',
    margin = dict(l = 200, r = 50, b = 50, t = 50, pad = 4),
    yaxis = dict(
        tickfont = dict(
            size = 9
        )
    )
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# **People are in their last new job for less than one year which may be indication that people often switch jobs.**

# #### Reason for updating CV last Time

# In[18]:


# Reason for updating your CV
update_CV = data['UpdateCV'].value_counts()
trace0 = go.Bar(
    x = update_CV.values,
    y = update_CV.index,
    orientation = 'h'
)
layout = go.Layout(
    height = 400,
    font=dict(size=12, color='black'),
    title = 'Reason for updating CV when updated Last time?',
    margin = dict(l = 400, r = 50, b = 50, t = 50, pad = 4),
    yaxis = dict(
        tickfont = dict(
            size = 9
        )
    )
)
trace = [trace0]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# **Due to changes in skills, personal status people use to update their CV.** Other erason people updated their CV was when they got call from recruiter or got bad experience in the present job and want a change.

# ## More to come!!
