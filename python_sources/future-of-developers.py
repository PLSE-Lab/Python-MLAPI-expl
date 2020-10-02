#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
data=pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")


# In[ ]:


data


# In[ ]:


#top 5 Country with most number of upcoming developers


# In[ ]:


data['Country'].value_counts().head(5)


# In[ ]:


#Which Gender will be dominating in this field


# In[ ]:


data['Gender'].value_counts().head(2).plot(kind='pie')


# In[ ]:


# Top 15 age group using stack overflow


# In[ ]:


data['Age'].value_counts().head(15).plot(kind='bar')


# In[ ]:


data['OpenSourcer'].value_counts()


# In[ ]:


data_india=data[data['Country']=='India']


# In[ ]:


data_usa=data[data['Country']=='United States']


# In[ ]:


data['OpenSourcer'].value_counts().plot(kind='pie')


# In[ ]:


data_india['OpenSourcer'].value_counts()


# In[ ]:


# ONLY 1562 developers from India , contribute to open source on a regular basis out 0f 9061


# In[ ]:


data['UndergradMajor'].value_counts().head(5).plot(kind='pie')


# In[ ]:


# students with a computer science background in their undergrad forms the majority 


# In[ ]:


data_india['EdLevel'].value_counts()


# In[ ]:


data_usa['EdLevel'].value_counts()


# In[ ]:


#It clearly shows that people in USA in their bachelors degree are more towards development and contributing towards open source

