#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


mcq_df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
mcq_df.shape


# In[ ]:


mcq_df.describe()


# ## **Gender**

# In[ ]:


sns.countplot(y=mcq_df['GenderSelect'])


# > Most of our respondents are male. Our data is highly skewed towards men.

# ## **Participants Country wise**

# In[ ]:


coun_df = pd.DataFrame(mcq_df['Country'].value_counts())
coun_df['country'] = coun_df.index
coun_df.columns = ['participants','country']
coun_df = coun_df.reset_index().drop('index',axis=1)
plt.figure(figsize=(11,10))
sns.barplot(x='participants', y='country', data=coun_df)


# ## **Popular Language**

# In[ ]:


lang = pd.DataFrame(mcq_df['LanguageRecommendationSelect'].value_counts())
lang['language'] = lang.index
lang.columns = ['count','language']
lang = lang.reset_index().drop('index', axis=1)
plt.figure(figsize=(11,5))
sns.barplot(x='count', y='language', data=lang)


# Python is most popular with around 7000 votes

# ## **Popular Language by Country**

# In[ ]:


x = mcq_df[mcq_df['Country'].notnull() & ((mcq_df['LanguageRecommendationSelect']=='Python')
                               | (mcq_df['LanguageRecommendationSelect']=='R'))]
plt.figure(figsize=(9,10))
sns.countplot(y='Country', hue='LanguageRecommendationSelect', data=x)


# As we can see, Python is still dominating in almost all countries

# ## **Choice of Language by Job Title**

# In[ ]:


d = mcq_df[(mcq_df['CurrentJobTitleSelect'].notnull())&((mcq_df['LanguageRecommendationSelect']=='Python')
                                                 | (mcq_df['LanguageRecommendationSelect']=='R'))]
plt.figure(figsize=(8, 10))
sns.countplot(y='CurrentJobTitleSelect', hue='LanguageRecommendationSelect', data=d)


# Almost every profession prefer to use Python as a language except statistician. As they prefer to use R
