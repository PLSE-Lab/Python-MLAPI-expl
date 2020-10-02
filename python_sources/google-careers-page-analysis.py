#!/usr/bin/env python
# coding: utf-8

# # Google Job Listings Data Analysis
# ### From needed skills to most in demand type of work. Gain insight into what Google is looking for in their hiring process

# # Table of Contents
# 1. [What programming languages are most in demand at Google?](#Step1)
# 2. [Which Degrees are Most Popular at Google?](#Step2)
# 3. [How Many Years of Experience is Google Looking For?](#Step3)
# 4. [What Experience Level is Most in Demand by Job Category?](#Step4)
# 5. [Where is Google Hiring?](#Step5)
# 6. [Which Job Categories Are Hiring The Most?](#Step6)
# 7. [What Are The Most In Demand Locations for These Jobs?](#Step7)
# 
# *Note: this notebook has been forked from and features data collected by Niyamat Ullah

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
import re

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/job_skills.csv')
# data.head()


# <h1>What programming languages are most in demand at Google? <a name="Step1"></a></h1>

# In[ ]:


minimum_qualifications = data['Minimum Qualifications'].tolist()
minimum_qualifications_string = "".join(str(v) for v in minimum_qualifications).lower()


# In[ ]:


programming_language_list = ['python', 'java', 'c++', 'php', 'javascript', 'objective-c', 'ruby', 'perl','c','c#', 'sql','kotlin']

word_count = dict((x,0) for x in programming_language_list)
for w in re.findall(r"[\w'+#-]+|[.!?;']", minimum_qualifications_string):
    if w in word_count:
        word_count[w] += 1
        
print(word_count)


# In[ ]:


programming_language_popularity = sorted(word_count.items(), key = lambda kv: kv[1], reverse=True)


# In[ ]:


programming_lang_df = pd.DataFrame(programming_language_popularity, columns = ['Languages', 'Popularity'])
#df formatting
programming_lang_df['Languages'] = programming_lang_df.Languages.str.capitalize()
programming_lang_df = programming_lang_df[::-1]


# In[ ]:


# programming_lang_df.head()


# In[ ]:


programming_lang_df.plot.barh(x = 'Languages', y = 'Popularity', figsize= (10,8), legend = False)
plt.suptitle('Programming Language Demand at Google', fontsize = 18)
plt.xlabel("")
plt.yticks(fontsize = 14)
plt.show


# <h1>Which Degrees are Most Popular at Google? <a name="Step2"></a></h1>

# In[ ]:


degree_list = ["ba", "bs", "bachelor's", "ms", "mba", "master", "master's", "phd"]
wordcount = dict((x,0) for x in degree_list)
for w in re.findall(r"[\w']+|[.,!?;']", minimum_qualifications_string):
    if w in wordcount:
        wordcount[w] += 1
        
print(wordcount)


# In[ ]:


degree_demand = sorted(wordcount.items(), key = lambda kv: kv[1], reverse = True)
degree_df = pd.DataFrame(degree_demand, columns = ['Degrees', 'Popularity'])
degree_df['Degrees'] = degree_df.Degrees.str.upper()
degree_df = degree_df[::-1]
degree_df.plot.barh(x = "Degrees", y = "Popularity", figsize = (10,8), legend = False)
plt.suptitle('Degree Demand at Google', fontsize = 18)
plt.xlabel("")
plt.yticks(fontsize = 14)
plt.show


# <h1>How Many Years of Experience is Google Looking For?<a name="Step3"></a></h1>

# In[ ]:


years_exp = defaultdict(lambda: 0)

for w in re.findall(r'[0-9]+ year', minimum_qualifications_string):
    years_exp[w] += 1

#print(years_exp)


# In[ ]:


years_exp = sorted(years_exp.items(), key=lambda kv: kv[1], reverse=True)
years_exp_df = pd.DataFrame(years_exp, columns = ['Years', 'Demand'])
years_exp_df = years_exp_df[::-1]
#years_exp_df.head()


# In[ ]:


years_exp_df.plot.barh(x = 'Years', y = 'Demand', figsize = (10,5), legend = False, stacked = True)
plt.title('Years of Experience Needed To Work At Google', fontsize = 18)
plt.xlabel("")
plt.yticks(fontsize = 14)
plt.show


# ## What Experience Level is Most in Demand by Job Category <a name="Step4"></a>?

# In[ ]:


data['Experience'] = data['Minimum Qualifications'].str.extract(r'([0-9]+) year')


# In[ ]:


job_exp = data[['Experience', 'Category']]
job_exp = job_exp.dropna()


# In[ ]:


plt.figure(figsize=(10,15))
plt.title('Experience Needed by Job Type', fontsize=18)
sns.countplot(y='Category', hue='Experience', data=data, hue_order = data.Experience.value_counts().iloc[:4].index)
plt.yticks(fontsize=18)
plt.show()


# # Where is Google Hiring <a name="Step5"></a>?

# In[ ]:


threshold = 10
location_count = data.Location.value_counts()
exclude = location_count[:10]
data['Location'].replace(exclude, np.nan, inplace=True)
location_count = data.Location.value_counts()
location_count = location_count[::-1]


# In[ ]:


location_count.plot.barh(figsize=(15,15))
plt.title('Google Job Location Demand', fontsize = 18)
plt.xlabel('Demand', fontsize=14)
plt.ylabel('Location', fontsize=14)
plt.yticks(fontsize=12)
plt.show()


# # Which Job Categories Are Hiring The Most<a name="Step6"></a>?

# In[ ]:


category_count = data.Category.value_counts()
category_count = category_count[::-1]
category_count.plot.barh(figsize=(15,15))
plt.title('Most in Demand Job Category at Google', fontsize=18)
plt.xlabel('Demand', fontsize=14)
plt.ylabel('Category', fontsize=14)
plt.yticks(fontsize=12)
plt.show()


# ##  What Are The Most In Demand Locations for These Jobs <a name=Step7></a>?

# In[ ]:


plt.figure(figsize=(10,15))
plt.title('Location Demand for Categories with Most Need', fontsize=24)
g = sns.countplot(y='Location', hue= 'Category', data=data, hue_order=data.Category.value_counts().iloc[:3].index)
plt.yticks(fontsize=18)
plt.show()


# In[ ]:




