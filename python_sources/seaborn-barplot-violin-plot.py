#!/usr/bin/env python
# coding: utf-8

# In[36]:




import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))


# In[37]:


heart_disease=pd.read_csv('../input/heart.csv')


# In[38]:


heart_disease.head(10)


# Check if any value is nan in the dataset

# In[39]:


heart_disease.isnull().values.any()


# Lets plot some bar plot using seabon
# we need to import the reuired libraries

# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


heart_disease.columns


# In[42]:


#ax = sns.barplot(x = 'target',y='age',data = heart_disease)
def bar_plot(x,y,hues=None,ci=95):
    ax = sns.barplot(x,y,data = heart_disease,hue = hues)


# In[60]:


import warnings
warnings.filterwarnings("ignore")
bar_plot(heart_disease['target'],heart_disease['age'],heart_disease['sex'],ci=78)


# In[61]:


bar_plot(heart_disease['target'],heart_disease['trestbps'],heart_disease['cp'])


# In[62]:


bar_plot(heart_disease['target'],heart_disease['chol'],heart_disease['sex'])


# In[63]:


sns.barplot(x = heart_disease['target'],y = heart_disease['fbs'],data= heart_disease,hue='cp')


# In[64]:


ax = sns.barplot(x='target',y = 'thalach',data= heart_disease,hue='thal')


# In[65]:


ax = sns.barplot(x = 'cp',y = 'chol',data = heart_disease,hue='target',palette='husl')


# let us create some seaborn violin plot as well

# In[49]:


x = 'target'
y = ['age','trestbps','chol','thalach','oldpeak']
hues = ['sex','cp','fbs']


# In[50]:


def violin_plot(x,y,hues):
    ax= sns.violinplot(x,y,data = heart_disease,hue=hues)
    plt.show()


# In[66]:


violin_plot(x,y[0],hues[0])


# In[67]:


violin_plot(x,y[2],hues[1])


# From the above combinations we can create differenet volin plot 

# In[68]:


violin_plot(x,y[2],hues[2])


# In[69]:


violin_plot(heart_disease['fbs'],y[1],hues[1])

