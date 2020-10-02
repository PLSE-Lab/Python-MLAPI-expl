#!/usr/bin/env python
# coding: utf-8

# #### Medical Data - Attempt to see what is inside of this to explore

# In[1]:


# Import necessary tools 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
import seaborn as sns


# In[2]:


df = pd.read_csv("../input/data.csv")


# In[3]:


df.head()


# ####  Diseases and thier count

# In[4]:


Disease = df['disease'].value_counts()
Disease.iplot(kind = 'bar', theme = 'solar',colors = 'Blue', xTitle = 'Disease Names', yTitle = 'No of patients', title = 'Diseases Frequency'
     )


# #### Any relationship with age and disease type ???. 

# Only the data of birth of the patient is given. from the DOB string, year is separated out and entered as a column

# In[5]:


Year_of_birth = [ ]
for str in list(df['dob']):
    year = int(str.split('-')[0])
    Year_of_birth.append(year)
df['YOB'] = Year_of_birth
df.head()


# Considering 2017 as the reference,let us calculate the age of the patients

# In[6]:


df['AGE'] = 2017 - df['YOB']


# #### Disease and Gender distrubution

# In[49]:


disease = list(df['disease'].unique())

for x in disease:
    trace = df[df['disease'] == x].groupby('gender').count()['AGE']
    trace.iplot(kind = 'bar', title = x, theme = 'solar')
    


# #### Observation :  Men are more sick than women !!!!

# In[47]:


df.groupby('gender').count()['id'].iplot(kind = 'bar', theme = 'solar')


# In[53]:


df.groupby('ancestry').count()['id'].iplot(kind = 'bar', theme = 'solar')


# #### Any relationship with age and disease type ???. 

# In[71]:


df['AGE'].value_counts().iplot(kind = 'bar', theme = 'solar')


# In[ ]:




