#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# interactive visualization
import plotly as py
from plotly.offline import iplot
import plotly.tools as tls

import cufflinks as cf

import plotly.graph_objs as go


# In[ ]:


data = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')


# In[ ]:


data.head()


# In[ ]:


df = data.copy()


# In[ ]:


df.drop("Unnamed: 0",axis=1, inplace=True)


# In[ ]:


df['course_students_enrolled']= df['course_students_enrolled'].str.replace('k', '*1000')
df['course_students_enrolled']= df['course_students_enrolled'].str.replace('m', '*1000000')
df['course_students_enrolled'] = df['course_students_enrolled'].map(lambda x: eval(x))


# ## Interactive Data Visualization 

# In[ ]:


py.offline.init_notebook_mode(connected=True)
cf.go_offline()


# ## Course Certificate Count

# In[ ]:


# print(cf.getThemes())
cf.set_config_file(theme='ggplot')
df['course_Certificate_type'].iplot(kind='hist',title='Course Certificate Type',
                                   xTitle='Course Type', yTitle='Counts')


# ### In this we can say that (Maximum -> Course , than Specialization and Minimum -> Professional Certificate)

# ## Rating Counts

# In[ ]:


# print(cf.getThemes())
cf.set_config_file(theme='polar')
df['course_rating'].iplot(kind='hist',title='Course Rating ',bargap=0.2,
                                   xTitle='Rating', yTitle='Counts')


# ### From this plot we can see that we Rating in around (4.4 to 4.8)
# ### Most of the courses get rating around 4.6 to 4.8 

# In[ ]:


# print(cf.getThemes())
cf.set_config_file(theme='pearl')
df['course_difficulty'].iplot(kind='hist',title='Course Difficulty',
                             xTitle='Course Type',yTitle='Count')


# ### WE have most of the course for the Beginner -> Intermediat -> Mixed -> and then Advanced

# In[ ]:


df.head()


# ## Students Enrolled with Respect to Course Difficulty

# In[ ]:


# print(cf.getThemes())
cf.set_config_file(theme='ggplot')
df.iplot(x='course_difficulty',y='course_students_enrolled',kind='bar')


# ### Most of students get enrolled on the Beginner level course and then Mixed and Intermediate and very less in Advanced Course

# ## Students Enrolled with respect to Course Certificate Type

# In[ ]:


# print(cf.getThemes())
cf.set_config_file(theme='pearl')
df.iplot(x='course_Certificate_type',y='course_students_enrolled',
        kind='bar')


# ### We can see that we have more students who enrolled in Course Certificate type and then Specialization and then professional certificate

# ## Which University or which Company provide the most courses

# In[ ]:


course_org = df.groupby('course_organization').count().reset_index()


# In[ ]:


trace = go.Pie(labels = course_org['course_organization'], values =course_org['course_students_enrolled'] )
data = [trace]
fig = go.Figure(data = data)
iplot(fig)


# ## With the help of our visualization we can say that 
# - University Of Pennsylvania (1)
# - University of michigan (2)
# - Google Cloud (3)
# - Duke University (4)
# - Johns Hopkins University (5)
# 
# and so on ....
