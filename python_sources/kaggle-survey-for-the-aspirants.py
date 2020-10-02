#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
plotly.offline.init_notebook_mode()
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

sur_schema = pd.read_csv('../input/SurveySchema.csv')
free_response = pd.read_csv('../input/freeFormResponses.csv')
mcq_response = pd.read_csv('../input/multipleChoiceResponses.csv')
# Any results you write to the current directory are saved as output.


#                          Participate in Surveys, help the community
# 
# I personally feel surveys are the best ways to reach many people and gather their unbiased opinions. However, participation is still a challenge. 
# 
# How many of you fill surveys seriously? :) 
# 
# Kaggle has made the data of its recent survey public and in this notebook I will try to show how this survey can be an inspiration to all aspirants of data science and I hope, it motivates a lot of people to see how their participation in surveys can be useful to others.
# 
# I have been a full stack web application developer and now am a data science aspirant. And hence, I would wish to see what it takes to be a data scientist. 
# To do that, let's check from the respondants, how long have the data scientists been in their role.
# 
# The survey had respondents with below roles:

# In[ ]:


mcq_response['Q6'].unique()


# I will focus right now only on Data Analyst and Data Scientist roles. 
# **Though not all data scientist/analysts use kaggle or take part of surveys, but these numbers can give pretty much idea of the active community in data science**

# In[ ]:


def filter_datascientists(f):
    return f[f['Q6'].isin(['Data Scientist','Data Analyst'])]

ds = filter_datascientists(mcq_response)
ds.dropna()
ms = ds.groupby(['Q8']).count().reset_index('Q8')

cf.go_offline()
series = ds['Q8'].value_counts()
series.iplot(kind='bar', yTitle='Number of respondants', title='Data Scientists/Analysts by Experience in the role')

def draw_pie(ser,title,labels):
    cf.go_offline()
    trace = plotly.graph_objs.Pie(labels=labels, values=ser)
    data = plotly.graph_objs.Data([trace])
    layout = plotly.graph_objs.Layout(title=title,showlegend=False)
    figure=plotly.graph_objs.Figure(data=data,layout=layout)
    plotly.offline.iplot(figure)


# This is inspiring!. Out of 6059 users, (1927+1131=3058) almost 50% of them have been in this role over just couple of years. Around 32% have just 0-1 years of experience.
# I am curious how long were they preparing for this role.
# May be let's check, how long they have been using machine learning methods or writing code to analyze data and what was their background/education.

# In[ ]:


labels = ds['Q8'].unique()
draw_pie(series,'Data Scientists/Analysts by Experience in the role',labels)


# In[ ]:


ds['Q8'].unique()


# In[ ]:


def new_datascientists(f):
    return f[f['Q8'].isin(['0-1','1-2'])]
new_ds = new_datascientists(ds)
print(new_ds['Q24'].unique())
print('--------------------------')
print(new_ds['Q25'].unique())
ser_exp_writing = new_ds['Q24'].value_counts()
ser_exp_ml = new_ds['Q25'].value_counts()
labels = new_ds['Q24'].unique()


# In[ ]:


draw_pie(ser_exp_writing,"Experience writing code to analyze data",labels)
draw_pie(ser_exp_ml,"Experience using machine learning methods",labels)


# Around 28% respondants say, they have written code for analysing data for over just 0-2 years. 
# For machine learning methods the percentage is almost 20% which is good to know. 
# Many responded 3-5 years and one reason could be they have been either working towards a Master's or a PHD degree.  Let's find out which stream they were pursuing their degree in. I am expecting majority to be in computer science or mathematics background.

# In[ ]:


students = new_ds[new_ds['Q24'] == '3-5 years']
students['study'] = students['Q4'] + students['Q5']
cts = students['study'].value_counts()
draw_pie(cts,"Educational background of <br> respondants writing code <br> for data analysis for 3-5 years",students['study'].unique())
students = new_ds[new_ds['Q25'] == '3-4 years']
students['study'] = students['Q4'] + students['Q5']
cts = students['study'].value_counts()
draw_pie(cts,"Educational background of <br> respondants using ML <br> for 3-4 years",students['study'].unique())


# Next steps - 
# How did the ones who did not come from mathematics/computer science background learn and enter the field?
# To do this we can check what courses did the respondents who have been up to 2 years in the data science role and belonged to a non-computer science or mathematics background take.

# In[ ]:


ds['Q18'].unique()


# In[ ]:


new_datascientist_lang = ds['Q24'].value_counts()
draw_pie(new_datascientist_lang,"Language suggestions by Data scientists in early years",ds['Q18'].unique())


#  **WORK IN PROGRESS............**

# In[ ]:




