#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read csv and find columns


# In[ ]:


df = pd.read_csv('../input/elsect_summary.csv')


# In[ ]:


texas = df[df['STATE']=='Texas']


# In[ ]:


texas.columns


# In[ ]:


#format data for pairplot


# In[ ]:


pplot_texas = texas[['TOTAL_REVENUE', 'FEDERAL_REVENUE','STATE_REVENUE', 'LOCAL_REVENUE', 'TOTAL_EXPENDITURE']]


# In[ ]:


sns.pairplot(pplot_texas)


# In[ ]:


#Visualization of TEA income and spending over the years


# In[ ]:


plt.scatter(x=texas['YEAR'],y=texas['STATE_REVENUE'],c='red')
plt.scatter(x=texas['YEAR'],y=texas['LOCAL_REVENUE'],c='green')
plt.scatter(x=texas['YEAR'],y=texas['FEDERAL_REVENUE'],c='blue')
plt.scatter(x=texas['YEAR'],y=texas['TOTAL_REVENUE'],c='black')
plt.scatter(x=texas['YEAR'],y=texas['TOTAL_EXPENDITURE'],c='orange',marker='x')
plt.legend()
plt.title('Education Spending in 10 Millions')

