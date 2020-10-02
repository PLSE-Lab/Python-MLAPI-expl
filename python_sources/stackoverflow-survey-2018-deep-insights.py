#!/usr/bin/env python
# coding: utf-8

# # ** This notebook mainly focuses on different unique visualization techniques and how we can use them to depict some great insights. Hope you like it.**
# 
# ## So lets get started with importing the libraries.

# In[ ]:


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
from pandas import Series

import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
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
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from wordcloud import WordCloud, STOPWORDS
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


df1=pd.read_csv('../input/survey_results_public.csv')
df1.head()


# In[ ]:


df1.shape


# ## Lets take a look at the Student Category

# In[ ]:


cnt_srs = df1['Student'].value_counts().head()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Student Category',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Students")


# ## Lets have a look at the students' category

# In[ ]:


temp = df1['Hobby'].value_counts()
trace = go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(
        color=temp.values,
        colorscale = 'Picnic'
    ),
)

layout = go.Layout(
    title="Students' Ratio taking up coding as a Hobby or not"
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="sc")


# In[ ]:


df_country=pd.DataFrame(df1.Country.value_counts().reset_index())


# In[ ]:


df_country.columns = ['Country', 'No_of_users']


# In[ ]:


df_country=df_country.head(10)


# ## Lets have a look at how many people are satisfied or not!

# In[ ]:


labels = df1['JobSatisfaction']
values = df1['JobSatisfaction'].value_counts()

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='basic_pie_chart')


# ## What's the avg. experience of coding of people on StackOverflow?

# In[ ]:


cnt_srs = df1['YearsCodingProf'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="blue",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Average years of coding experience'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Code")  


# In[ ]:


df1.head(1)


# ## Lets take a look from which country there are maximum participants on StackOverflow

# In[ ]:


colors = ['#91BBF4', '#91F4F4', '#F79981', '#F7E781', '#C0F781','rgb(32,155,160)', 'rgb(253,93,124)', 'rgb(28,119,139)', 'rgb(182,231,235)', 'rgb(35,154,160)']

n_phase = len(df_country['Country'])
plot_width = 200

# height of a section and difference between sections 
section_h = 100
section_d = 10

# multiplication factor to calculate the width of other sections
unit_width = plot_width / max(df_country['No_of_users'])

# width of each funnel section relative to the plot width
phase_w = [int(value * unit_width) for value in df_country['No_of_users']]

height = section_h * n_phase + section_d * (n_phase - 1)

# list containing all the plot shapes
shapes = []

# list containing the Y-axis location for each section's name and value text
label_y = []

for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)
        
label_trace = go.Scatter(
    x=[-200]*n_phase,
    y=label_y,
    mode='text',
    text=df_country['Country'],
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)
 
# For phase values
value_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=df_country['No_of_users'],
    textfont=dict(
        color='rgb(200,200,200)',
        size=12
    )
)

data = [label_trace, value_trace]
 
layout = go.Layout(
    title="<b>Top 10 Countries having maximum Users</b>",
    titlefont=dict(
        size=12,
        color='rgb(203,203,203)'
    ),
    shapes=shapes,
    height=600,
    width=800,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)
 
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## So US and India have maximum participants on StackOverflow

# ### What are the major languages people have worked with?

# In[ ]:


df_language=df1.dropna(subset=['LanguageWorkedWith'])


# In[ ]:


stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df_language['LanguageWorkedWith'])


# ### Can you see C++? I can't

# In[ ]:


df1.head(1)


# In[ ]:


stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df1['DevType'])


# In[ ]:


df2=pd.read_csv('../input/survey_results_schema.csv')
df2.head()


# In[ ]:


stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df2['Column'])


# In[ ]:


df1.head(2)


# ### What is the education background of most of the Stackoverflow users

# In[ ]:


df1['UndergradMajor'].unique()


# In[ ]:


edu_back = {'Mathematics or statistics': "Science (PCM/B)", 'A natural science (ex. biology, chemistry, physics)': "Science (PCM/B)",            'Computer science, computer engineering, or software engineering':'Computer Science/IT', 'Fine arts or performing arts (ex. graphic design, music, studio art)':           'Fine Arts', 'Information systems, information technology, or system administration':'Computer Science/IT',            'Another engineering discipline (ex. civil, electrical, mechanical)':'Engineering', 'A business discipline (ex. accounting, finance, marketing)':'BBA/MBA',           'A social science (ex. anthropology, psychology, political science)':'Social Science', 'Web development or web design':'Computer Science/IT',            'A humanities discipline (ex. literature, history, philosophy)':'Humanities', 'A health science (ex. nursing, pharmacy, radiology)':'Pharmaceuticals',            'I never declared a major':'No Degree'}

df1['Degree'] = df1['UndergradMajor']
df1 = df1.replace({"Degree": edu_back})


# In[ ]:


df_deg_count = pd.DataFrame(df1['Degree'].value_counts()).reset_index()
df_deg_count


# In[ ]:


trace = go.Bar(
    y=df_deg_count['Degree'],
    x=df_deg_count['index'],
    orientation = 'v',
    marker=dict(
        color=df_deg_count['Degree'].values[::-1],
        colorscale = 'viridis',
        reversescale = True
    ),
)

layout = dict(
    title='Education Background of StackOverflow Users',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="EduBack")


# In[ ]:


df1['Employment'].unique()


# In[ ]:


empl_stat = {'Employed part-time':'Employee (Full/Part Time)', 'Employed full-time': 'Employee (Full/Part Time)', 'Independent contractor, freelancer, or self-employed'            :'Freelancer/Contractor','Not employed, and not looking for work':'Student/Researcher',  'Not employed, but looking for work':'Unemployed', 'Retired':'retired'}

df1['emp_stat'] = df1['Employment']
df1 = df1.replace({"emp_stat": empl_stat})


# In[ ]:


df1['emp_stat'].value_counts()


# In[ ]:


trace = go.Pie(labels=['Employee (Full/Part Time)','Freelancer/Contractor', 'Unemployed', 'Student/Researcher', 'Retired'],
                             values= [75857,9282, 5805, 4132, 227], hole=0.3)
                          

layout = dict(
    title='Employment Status of StackOverflow Users',
    )

data = [trace]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="EmplStat")

