#!/usr/bin/env python
# coding: utf-8

# ![https://imgur.com/a/RNxiRy0](http://)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import squarify
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2
from subprocess import check_output


# In[ ]:


data = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv", skiprows=[1])


# In[ ]:


data.head()


# # Analysis

# In[ ]:


print('Total number of respondents:', data.shape[0])
print('Total number of Countries with respondents: ', data["Q3"].nunique())
print("Country with most respondents:", data['Q3'].value_counts().index[0], 'with', data['Q3'].value_counts().values[0], 'respondents')
data['Q3'].value_counts().sort_values(ascending=True).plot(kind='bar', title='Where are Kagglers residing?', figsize=(10,10))
plt.show()


# # Gender Split

# In[ ]:


plt.subplots(figsize=(22,12))
sns.countplot(y=data['Q1'],order=data['Q1'].value_counts().index)
plt.show()


# # About Age

# In[ ]:


sns.barplot(data['Q2'].value_counts().index, data['Q2'].value_counts().values, palette='inferno')
plt.title('Age ranges of Kagglers')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# It is seen that, Most people are between age 25 and 29. In second number we have more people from range 22-24. It means that overall more are between age range 20-30. Lets look for some more insights.

# # Respondents by Country

# In[ ]:


response_count=data['Q3'].value_counts()[:15].to_frame()
sns.barplot(response_count['Q3'],response_count.index,palette='inferno')
plt.title('Top 15 countries by Number of Respondents')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# United States and India are on top of the List and far away from 3rd position i.e China. Fourth position is covered by the respondents who did not fill the field for Country.

# # Salary Report

# Data Scientists are said to be one of the most paying individuals. Let see what survey say about them.

# In[ ]:


sns.barplot(data['Q9'].value_counts().index, data['Q9'].value_counts().values, palette='inferno')
plt.title('Salaries of Kagglers')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(20,8)
plt.show()
data['Q9'].value_counts()


# WoW.... It is seen that most respondents do not show their Salaries. Most people have salary between 0-10,000. Only 23 Kagglers have salary range between 400,000 and 500,000 interesting.... Lets explore some more.

# # Major Subjects

# In[ ]:


data['Q5'].value_counts()
data['Q5'].value_counts().plot.bar(figsize=(15,10), fontsize=12)
plt.xlabel('UG Majors', fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('UG Majors - Split of respondents', fontsize=20)
plt.show()


# It is seen that most of the respondents have Computer Science as their Major subject. Engineering and Staticians took 2nd and 3rd level respectively. We have also evaluate that what Kagglers use to describe themselves. Lets have some more interesting and fun insights.

# # Education

# In[ ]:


temp_series=data['Q4'].value_counts()
labels=(np.array(temp_series.index))
sizes=(np.array((temp_series/temp_series.sum())*100))
trace=go.Pie(labels=labels, values=sizes, hole=0.5)
layout=go.Layout(title='Education distribution')
dataEducation=[trace]
fig=go.Figure(data=dataEducation, layout=layout)
py.iplot(fig, filename='Education')


# Interesting..... Very interesting. Almost half of respondents have Master's Degree. One third of respondents have Bachelor's Degree. Almost 1/7th half of people have Doctorate Degree. In other 10% respondents, some have college degree, some with Professional degree and some do not prefer to tell. 

# # Job Title

# In[ ]:


data['Q6'].value_counts()
data['Q6'].value_counts().plot.bar(figsize=(15,10), fontsize=12)
plt.xlabel('Job Title', fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('Job Title - Split of respondents', fontsize=20)
plt.show()


# Most respondents are Students, Data Scientists, Software Engineers and Data Analysts. Interesting. Lets have some more fun.

# # Industry

# Let's have a look in which industry most Kagglers are contributing their talent.

# In[ ]:


data['Q7'].value_counts().sort_values(ascending=True).plot(kind='bar', title='In which industry Kagglers work?', figsize=(10,10))
plt.show()


# Top three industries are Computer/IT, Students and Educational Department. It means that most of the Kagglers are Tech geeks. 

# **Check Back Soon, and Upvote if you enjoy it.**

# In[ ]:





# In[ ]:




