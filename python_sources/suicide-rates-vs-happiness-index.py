#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


suicide_data = pd.read_csv('../input/data.csv')


# In[9]:


suicide_data.head()


# In[10]:


suicide_data.columns = suicide_data.iloc[0]


# In[11]:


suicide_data.head()


# In[12]:


suicide_data = suicide_data[1:]


# In[13]:


suicide_data.drop([2015.0, 2010.0,2000.0], axis=1,inplace=True)


# In[14]:


suicide_data.rename(index=str, columns={2016.0: "Suicide rates per 1Lakh population for 2016"},inplace=True)


# In[15]:


suicide_data.head()


# In[16]:


suicide_data = suicide_data[(suicide_data['Sex'] == 'Both sexes')]


# In[17]:


suicide_data.drop(['Sex'], axis=1,inplace=True)


# In[18]:


suicide_data.head()


# In[21]:


happiness_data = pd.read_csv('../input/2016.csv')


# In[22]:


happiness_data.head()


# In[23]:


happiness_data = happiness_data[['Country','Happiness Rank']]


# In[24]:


happiness_data.head()


# In[25]:


suicide_data.index = suicide_data['Country']
happiness_data.index = happiness_data['Country']


# In[26]:


suicide_data.drop(['Country'], axis=1,inplace=True)
happiness_data.drop(['Country'],axis=1,inplace=True)


# In[27]:


suicide_data.head()


# In[28]:


happiness_data.head()


# In[29]:


#inner join
compiled_data = pd.merge(suicide_data, happiness_data, left_index=True, right_index=True) 


# In[30]:


compiled_data.head()


# In[31]:


compiled_data = compiled_data.sort_values(by=['Happiness Rank'])
compiled_data.head()


# In[32]:


sns.scatterplot(x='Happiness Rank',y='Suicide rates per 1Lakh population for 2016',data=compiled_data)


# In[33]:


sns.lmplot(x='Happiness Rank',y='Suicide rates per 1Lakh population for 2016',data=compiled_data)


# In[34]:


top_40 = compiled_data[0:40]
top_40.head()


# For the top 40 ranked countries - **the more happier a country is ranked the more are the suicide rates :-** 

# In[35]:


sns.lmplot(x='Happiness Rank',y='Suicide rates per 1Lakh population for 2016',data=top_40)


# Not any correlation between suicide rates and happiness index :- 

# In[36]:


compiled_data.corr()


# In[38]:


happiness_index_other_variables = pd.read_csv('../input/2016.csv')


# In[39]:


happiness_index_other_variables.head()


# In[40]:


happiness_index_other_variables.index = happiness_index_other_variables['Country']
happiness_index_other_variables.drop(['Country'],axis=1,inplace=True)


# In[41]:


happiness_index_other_variables.head()


# In[42]:


happiness_index_suicide_rankings = pd.merge(suicide_data, happiness_index_other_variables, left_index=True, right_index=True) 


# In[43]:


happiness_index_suicide_rankings.head()


# In[44]:


happiness_index_suicide_rankings.corr()


# **Suicide rates are more related to the Health Index(Life Expectancy) - Lesser the score under Health corresponds to a higher suicide rate in the country.**

# In[45]:


sns.lmplot(x='Health (Life Expectancy)',y='Suicide rates per 1Lakh population for 2016',data=happiness_index_suicide_rankings)


# In[ ]:




