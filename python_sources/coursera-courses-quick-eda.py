#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas_profiling import ProfileReport


# In[ ]:


courses = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')
courses.head()


# In[ ]:


courses_profile = ProfileReport(courses)


# In[ ]:


courses_profile.to_widgets()


# In[ ]:


courses_profile.to_notebook_iframe()

