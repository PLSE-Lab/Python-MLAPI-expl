#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


#Load Dataset
ed=pd.read_csv('../input/xAPI-Edu-Data.csv')


# In[ ]:


ed.head()


# In[ ]:


ed.describe()


# In[ ]:


ed.shape


# In[ ]:


ed.info()


# In[ ]:


ed['Class'].value_counts()


# In[ ]:


ed.groupby(['gender'])['Class'].count().reset_index()


# In[ ]:


ed.groupby(['NationalITy'])['Class'].count().reset_index().sort(['Class'],ascending=False)


# In[ ]:


ed.groupby(['GradeID'])['Class'].count().reset_index().sort(['GradeID'])


# In[ ]:


ed.groupby(['GradeID','Semester'])['Class'].count().reset_index().sort(['GradeID'])


# In[ ]:





# In[ ]:




