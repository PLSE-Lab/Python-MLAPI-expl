#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go


# In[ ]:


import plotly.express as px


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')


# In[ ]:


df.shape


# In[ ]:


df.head(4)


# In[ ]:


df.isna().sum()


# In[ ]:


df.info(memory_usage='deep')


# ### Course Certificate Type

# In[ ]:


df.course_Certificate_type.value_counts().plot(kind='bar')


# ### Course Difficulty

# In[ ]:


df.course_difficulty.value_counts().plot(kind='bar')


# In[ ]:


df.course_students_enrolled = df.course_students_enrolled.apply(lambda x : float(str(x).replace('k', '').replace('m',''))*1000)


# ### Top 10 courses

# In[ ]:


top_10 = df.nlargest(10, 'course_students_enrolled')


# In[ ]:


fig = px.bar(top_10, x='course_title', y='course_students_enrolled')
fig.update_layout(
    title = 'Top 10 courses by number of students enrolled',
    xaxis_title="Courses",
    yaxis_title="Students enrolled",
)
fig.show()


# #### As we can see Data Science is most popular Course

# ## Students enrolled By Course Difficulty

# In[ ]:


course_dif = df.groupby('course_difficulty')['course_students_enrolled'].sum().reset_index()
fig = px.bar(course_dif.sort_values(by = 'course_students_enrolled', ascending=False), x='course_difficulty', y='course_students_enrolled')
fig.update_layout(
    title = 'Top 10 courses by number of students enrolled',
    xaxis_title="Courses",
    yaxis_title="Students enrolled",
)
fig.show()


# #### Beginners are more enrolled in courses.
