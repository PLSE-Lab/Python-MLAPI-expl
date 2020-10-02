#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


asean_countries_survey = ['Indonesia', 'Singapore', 'Thailand', 'Viet Nam', 'Malaysia', 'Philippines']
# Survey 2017
df_17 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding='ISO-8859-1')
df17_asean = df_17[df_17['Country'].isin(asean_countries_survey)]
df17_rest = df_17[~df_17['Country'].isin(asean_countries_survey)]
df_17['Area']=["Asean" if x in asean_countries_survey else "Others" for x in df_17['Country']]

# Survey 2018
df_18 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")
df18_asean = df_18[df_18['Q3'].isin(asean_countries_survey)]
df18_rest = df_18[~df_18['Q3'].isin(asean_countries_survey)]
df_18['Area']=["Asean" if x in asean_countries_survey else "Others" for x in df_18['Q3']]

# Survey 2019
df_19 = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")
df_19 = df_19.drop(0, axis=0)
df19_asean = df_19[df_19['Q3'].isin(asean_countries_survey)]
df19_rest = df_19[~df_19['Q3'].isin(asean_countries_survey)]
df_19['Area']=["Asean" if x in asean_countries_survey else "Others" for x in df_19['Q3']]


# # Motivation for Asean Data Enthusiast
# ***
# **Yusnardo Tendio | 16/11/2019**
# 
# Finally, the kaggle survey dataset competition has begun. This competition has been held three times. I see a decrease in interest in respondents so I want to raise this issue in order to add motivation to all Kagglers especially Asean Kagglers. I hope that this simple writing can provide very broad benefits.
# 
# I apologize if there is a grammar error in this paper because English is not my mother language.

# # Table of contents
# ***
# > 
# * [Introduction](#introduction)
# 
# * [1. Asean Respondent in Kaggle 2019 Survey](#respondent)
#     * [1.1 Comparing Total Respondent by Area](#respondent_by_area)
#     * [1.2 Comparing Respondent by Country in Asean](#respondent_by_asean)
#     * [1.3 Comparing Asean Respondent by Year](#respondent_by_year)
# * [2. Comparative Analysis of Asean & Other Kagglers](#comparative)
#     * [2.1 Age](#age)
#     * [2.2 Gender](#gender)
#     * [2.3 Highest Education Level](#education)
#     * [2.4 Popular Platform to Study Data Science](#platform)
#     * [2.5 Experience Writing Code to Analyse Data](#exp)
# * [3. Motivation from Newbie for Newbie](#motivation)
#     * [3.1 Job Oportunity](#job)
#     * [3.2 Salary Overview](#salary)
# * [4. Conclusion](#conclusion)
# * [References](#ref)

# # Introduction
# <a id="introduction"></a>
# ***
# 
# I see a decrease in interest based on the number of respondents compared to the previous year. So what is data science? Based on [Datajobs.com](https://datajobs.com/what-is-data-science), Data science is a multidisciplinary blend of data inference, algorithmm development, and technology in order to solve analytically complex problems [1]. 
# 
# The Association of Southeast Asian Nations, or ASEAN, was established on 8 August 1967 in Bangkok, Thailand, with the signing of the ASEAN Declaration (Bangkok Declaration) by the Founding Fathers of ASEAN, namely Indonesia, Malaysia, Philippines, Singapore and Thailand.
# 
# I hope with this kernel, everyone's motivation and passion increases. **Happy reading :)**

# # 1. Asean Respondent in Kaggle 2019 Survey
# <a id="respondent"></a>
# ***
# 
# What kind of distribution does Asean Kaglers have?

# ## Comparing Total Respondents
# <a id='respondent_by_area'> </a>
# ***
# 
# Surprisingly, There are only 3.36% of respondents from ASEAN. Reversed with Asian respondents who are the most respondents compared to other continents [2].

# In[ ]:


tmp = df_19.Area.value_counts()
labels = (np.array(tmp.index))
sizes = (np.array((tmp / tmp.sum())*100))

trace = go.Pie(labels = labels, values = sizes)
layout = go.Layout(
    title = 'Asean Respondent VS The Rest of The World'
)
data = [trace]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = "Compare_Respondent")


# ## Comparing Asean Respondent
# <a id='respondent_by_asean'></a>
# ***
# Knowing that there are only 3.36% of respondents from ASEAN, I dig deeper into which countries in ASEAN are the biggest contributions. I found that Indonesia, Singapore and Viet Nam were the most respondents from ASEAN. 
# 
# Actually, there are several other ASEAN countries that did not enter this survey such as the Philippines, Brunei Darussalam, Laos, and others. I hope that these countries can join the Kaggle Survey 2020.

# In[ ]:


tmp = df19_asean.Q3.value_counts()
labels = (np.array(tmp.index))
sizes = (np.array((tmp / tmp.sum())*100))

trace = go.Pie(labels = labels, values = sizes)
layout = go.Layout(
    title = 'Asean Respondent'
)
data = [trace]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = "Asean_Respondent")


# ## Comparing Asean Respondents by Year
# <a id='respondent_by_year'></a>
# ***
# This section I want to show the problems that I want to raise. It can be seen in the plot below that the respondents from Asean declined when compared to the 2018 Kaggle Survey so I wondered why this could happen.
# 
# It turned out that the total respondents in the 2019 Kaggle Survey overall also experienced a decline from the 2018 Kaggle Survey.
# 
# Hopefully with this writing, respondents in the 2019 Kaggle Survey have increased again, especially from ASEAN respondents.

# In[ ]:


import plotly.graph_objects as go

respond_2019_asean = df_19.Area.value_counts()['Asean']
respond_2018_asean = df_18.Area.value_counts()['Asean']
respond_2017_asean = df_17.Area.value_counts()['Asean']
data = [[2017, respond_2017_asean], [2018, respond_2018_asean], [2019, respond_2019_asean]]
custom = pd.DataFrame(data, columns = ['Year', 'Total_Asean_Respondent']) 

trace = go.Scatter(
    x = custom.Year,
    y = custom.Total_Asean_Respondent,
    mode = 'lines',
    name = 'Asean Respondent by Year'
)

layout = go.Layout(title = 'Asean Respondent by Year')
figure = go.Figure(data = trace, layout = layout)
figure.show()


# In[ ]:


respond_2019 = df_19['Q1'].count()
respond_2018 = df_18['Q1'].count()
respond_2017 = df_17['Country'].count()
data = [[2017, respond_2017], [2018, respond_2018], [2019, respond_2019]]
custom = pd.DataFrame(data, columns = ['Year', 'Total_Respondent']) 

trace = go.Scatter(
    x = custom.Year,
    y = custom.Total_Respondent,
    mode = 'lines',
    name = 'Total Respondent by Year'
)

layout = go.Layout(title = 'Total Respondent by Year')
figure = go.Figure(data = trace, layout = layout)
figure.show()


# # 2. Comparative Analysis of Asean & Other Kagglers
# <a id='comparative'></a>
# ***
# Very many young people who were respondents in this survey both Asean Kagglers and others. In this section, we know Data Science has good prospects because there is a lot of young people love to learn data :).

# ## Age
# <a id='age'></a>
# ***

# In[ ]:


import plotly.graph_objs as go

asean_age_percentage = ((df_19.groupby('Area').get_group('Asean')['Q1'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Asean')['Q1'].sort_index() .count())*100)
others_age_percentage = ((df_19.groupby('Area').get_group('Others')['Q1'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Others')['Q1'].sort_index() .count())*100)
    
    
x = df_19.groupby('Area').get_group('Asean')['Q1'].value_counts().sort_index().index
y1 = asean_age_percentage
y2 = others_age_percentage


trace1 = go.Bar(
    x = x,
    y = y1,
    name = 'Asean',
    marker = dict(
        color='rgb(49,130,189)'
    )
)
trace2 = go.Bar(
    x = x,
    y = y2,
    name = 'Others',
    marker = dict(
        color='rgb(55, 83, 109)'
    )
)

layout = go.Layout(
    title = 'Age',
    xaxis=dict(tickangle=-45, title='Age by Total Respondent Each Area'),
    barmode='group',
    yaxis=dict(
        title='percentage',
    )
)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
fig.show()


# ## Gender
# <a id='gender'></a>
# ***
# Predictably, men dominate in this survey in the Asean area. We hope that in the next year's survey strong women will emerge who also want to be serious about learning and community in Kaggle.

# In[ ]:


tmp = df19_asean.Q2.value_counts()
labels = (np.array(tmp.index))
sizes = (np.array((tmp / tmp.sum())*100))

trace = go.Pie(labels = labels, values = sizes)
layout = go.Layout(
    title = 'Gender of Asean Respondent'
)
data = [trace]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = "Gender_Asean_Respondent")


# ## Highest Education Level
# <a id='education'></a>
# ***
# In Education section, most of respondents is Bachelor or Master degree.

# In[ ]:


asean_education_percentage = ((df_19.groupby('Area').get_group('Asean')['Q4'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Asean')['Q4'].sort_index() .count())*100)
others_education_percentage = ((df_19.groupby('Area').get_group('Others')['Q4'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Others')['Q4'].sort_index() .count())*100)
    
    
x = df_19.groupby('Area').get_group('Asean')['Q4'].value_counts().sort_index().index
y1 = asean_education_percentage
y2 = others_education_percentage


trace1 = go.Bar(
    x = x,
    y = y1,
    name = 'Asean',
    marker = dict(
        color='rgb(49,130,189)'
    )
)
trace2 = go.Bar(
    x = x,
    y = y2,
    name = 'Others',
    marker = dict(
        color='rgb(55, 83, 109)'
    )
)

layout = go.Layout(
    title = 'Highest Education Level',
    xaxis=dict(tickangle=-20),
    barmode='group',
    yaxis=dict(
        title='percentage',
    )
)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
fig.show()


# ## Popular Platform To Study Data Science in Asean
# <a id='platform'></a>
# ***
# For beginners like me, this section presents several platforms that can be used to increase knowledge about Data Science. Hopefully the plot below can provide a good platform for all of you to learn.

# In[ ]:


Q13_1 = df19_asean.Q13_Part_1.value_counts()
Q13_2 = df19_asean.Q13_Part_2.value_counts()
Q13_3 = df19_asean.Q13_Part_3.value_counts()
Q13_4 = df19_asean.Q13_Part_4.value_counts()
Q13_5 = df19_asean.Q13_Part_5.value_counts()
Q13_6 = df19_asean.Q13_Part_6.value_counts()
Q13_7 = df19_asean.Q13_Part_7.value_counts()
Q13_8 = df19_asean.Q13_Part_8.value_counts()
Q13_9 = df19_asean.Q13_Part_9.value_counts()
Q13_10 = df19_asean.Q13_Part_10.value_counts()
Q13_11 = df19_asean.Q13_Part_11.value_counts()
Q13_12 = df19_asean.Q13_Part_12.value_counts()

Q13_index = [Q13_1.index[0], Q13_2.index[0], Q13_3.index[0], Q13_4.index[0],
            Q13_5.index[0], Q13_6.index[0], Q13_7.index[0], Q13_8.index[0],
            Q13_9.index[0], Q13_10.index[0], Q13_11.index[0], Q13_12.index[0]]
Q13_value = [Q13_1[0], Q13_2[0], Q13_3[0], Q13_4[0],
            Q13_5[0], Q13_6[0], Q13_7[0], Q13_8[0],
            Q13_9[0], Q13_10[0], Q13_11[0], Q13_12[0]]

Q13 = pd.Series(Q13_value, index = Q13_index)

tmp = Q13
labels = (np.array(tmp.index))
sizes = (np.array((tmp / tmp.sum())*100))

trace = go.Pie(labels = labels, values = sizes)
layout = go.Layout(
    title = 'Popular Platform To Study Data Science in Asean'
)
data = [trace]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = "Platform_Asean_Respondent")


# In[ ]:


Q13_1 = df19_rest.Q13_Part_1.value_counts()
Q13_2 = df19_rest.Q13_Part_2.value_counts()
Q13_3 = df19_rest.Q13_Part_3.value_counts()
Q13_4 = df19_rest.Q13_Part_4.value_counts()
Q13_5 = df19_rest.Q13_Part_5.value_counts()
Q13_6 = df19_rest.Q13_Part_6.value_counts()
Q13_7 = df19_rest.Q13_Part_7.value_counts()
Q13_8 = df19_rest.Q13_Part_8.value_counts()
Q13_9 = df19_rest.Q13_Part_9.value_counts()
Q13_10 = df19_rest.Q13_Part_10.value_counts()
Q13_11 = df19_rest.Q13_Part_11.value_counts()
Q13_12 = df19_rest.Q13_Part_12.value_counts()

Q13_index = [Q13_1.index[0], Q13_2.index[0], Q13_3.index[0], Q13_4.index[0],
            Q13_5.index[0], Q13_6.index[0], Q13_7.index[0], Q13_8.index[0],
            Q13_9.index[0], Q13_10.index[0], Q13_11.index[0], Q13_12.index[0]]
Q13_value = [Q13_1[0], Q13_2[0], Q13_3[0], Q13_4[0],
            Q13_5[0], Q13_6[0], Q13_7[0], Q13_8[0],
            Q13_9[0], Q13_10[0], Q13_11[0], Q13_12[0]]

Q13 = pd.Series(Q13_value, index = Q13_index)

tmp = Q13
labels = (np.array(tmp.index))
sizes = (np.array((tmp / tmp.sum())*100))

trace = go.Pie(labels = labels, values = sizes)
layout = go.Layout(
    title = 'Popular Platform To Study Data Science in Non-Asean'
)
data = [trace]
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = "Platform_Non_Asean_Respondent")


# ## Experience Writing Code to Analyse Data
# <a id='exp'></a>
# ***
# And yes, you need to know how to code. Most of the respondent is 0 - 2 years experience in coding. Let's study together, and be surprised to survey next year that the respondents have good coding skills

# In[ ]:


asean_exp_percentage = ((df_19.groupby('Area').get_group('Asean')['Q15'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Asean')['Q15'].sort_index() .count())*100)
others_exp_percentage = ((df_19.groupby('Area').get_group('Others')['Q15'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Others')['Q15'].sort_index() .count())*100)
    
    
x = df_19.groupby('Area').get_group('Asean')['Q15'].value_counts().sort_index().index
y1 = asean_exp_percentage
y2 = others_exp_percentage


trace1 = go.Bar(
    x = x,
    y = y1,
    name = 'Asean',
    marker = dict(
        color='rgb(49,130,189)'
    )
)
trace2 = go.Bar(
    x = x,
    y = y2,
    name = 'Others',
    marker = dict(
        color='rgb(55, 83, 109)'
    )
)

layout = go.Layout(
    title = 'Experience Writing Code to Analyse Data',
    xaxis=dict(tickangle=-20),
    barmode='group',
    yaxis=dict(
        title='percentage',
    )
)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
fig.show()


# # 3. Motivation from Newbie for Newbie
# <a id='motivation'></a>
# ***
# So there are some reason why we go this path. I want to show you that the work that awaits you is vast and varied, the following plot below can provide a brief overview of your future job prospects.

# ## Job Oportunity
# <a id='job'></a>
# ***

# In[ ]:


asean_job_percentage = df_19['Q5'].value_counts().sort_index()
    
    
x = df_19['Q5'].value_counts().sort_index().index
y = asean_job_percentage


trace = go.Bar(
    x = x,
    y = y,
    name = 'Asean',
    marker = dict(
        color='rgb(49,130,189)'
    )
)

layout = go.Layout(
    title = 'Job Oportunity',
    xaxis=dict(tickangle=-20),
    barmode='group',
    yaxis=dict(
        title='percentage',
    )
)
fig = go.Figure(data=trace, layout=layout)
fig.show()


# ## Salary
# <a id='salary'></a>
# ***
# Predictable, most of the respondents have \\$ 0-999 year. I think this is happening because so many newbies are responding. But don't be worry there is a lot of people can reach \> \\$ 500.000. This is **FANTASTIC**, right? 

# In[ ]:


asean_salary_percentage = ((df_19.groupby('Area').get_group('Asean')['Q10'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Asean')['Q10'].sort_index() .count())*100)
others_salary_percentage = ((df_19.groupby('Area').get_group('Others')['Q10'].value_counts().sort_index()  / 
                         df_19.groupby('Area').get_group('Others')['Q10'].sort_index() .count())*100)
    
    
x = df_19.groupby('Area').get_group('Asean')['Q10'].value_counts().sort_index().index
y1 = asean_salary_percentage
y2 = others_salary_percentage


trace1 = go.Bar(
    x = x,
    y = y1,
    name = 'Asean',
    marker = dict(
        color='rgb(49,130,189)'
    )
)
trace2 = go.Bar(
    x = x,
    y = y2,
    name = 'Others',
    marker = dict(
        color='rgb(55, 83, 109)'
    )
)

layout = go.Layout(
    title = 'Salary Overview',
    xaxis=dict(tickangle=-20),
    barmode='group',
    yaxis=dict(
        title='percentage',
    )
)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
fig.show()


# # 4. Conclusion
# <a id='conclusion'></a>
# ***
# Yap, this is the most important part. So in this notebook i want to show you all that Data is most powerful weapon in your hand.
# 1. Respondents from Asean are only 3.3 % from all of the respondents. This shows that data science is still lacking interest in ASEAN. And I hope with this notebook a significant improvement can occur <br>
# 
# 2. The prospect of data going forward is extraordinary. <br>
# 
# 3. Let's study together
# 

# # References
# <a id='ref'></a>
# ***
# [1] [https://datajobs.com/what-is-data-science](https://datajobs.com/what-is-data-science) <br>
# [2] [michau96 - kagglers-continent-fight-2019](https://www.kaggle.com/michau96/kagglers-continent-fight-2019)
