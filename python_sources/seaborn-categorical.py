#!/usr/bin/env python
# coding: utf-8

# # **CATEGORICAL PLOTS**

# In[ ]:


import seaborn as sns


# In[ ]:


df = sns.load_dataset('tips')


# In[ ]:


df.head()


# **COUNT** **PLOT**

# In[ ]:


sns.countplot('sex', data=df)


# In[ ]:


sns.countplot('smoker', data=df)


# In[ ]:


sns.countplot('day', data=df)


# In[ ]:


sns.countplot(y='sex', data=df)


# **BAR** **PLOT**

# In[ ]:


sns.barplot(x='total_bill', y='sex', data=df)


# In[ ]:


sns.barplot(y='total_bill', x='sex', data=df)


# **BOX PLOT**

# In[ ]:


sns.boxplot('sex', 'total_bill', data=df)


# In[ ]:


sns.boxplot('sex', 'total_bill', data = df, palette='rainbow')


# In[ ]:


sns.boxplot(data=df)


# In[ ]:


sns.boxplot(data=df, x='total_bill', y='day', hue='smoker')


# **VIOLIN PLOT**

# In[ ]:


sns.violinplot(x='total_bill', y='day', data=df, palette='rainbow')

