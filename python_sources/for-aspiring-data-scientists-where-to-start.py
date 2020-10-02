#!/usr/bin/env python
# coding: utf-8

# # Expert Consensus?
# 
# I'm currently a first year M.S. Applied Data Science student at Syracuse University. And like many of the people interested in data science, I was curious as to what the 'experts' in the field were saying.
# 
# Personally, I have a bias towards Python for analysis and data wrangling, as it's the first language I've learned. But, almost every University Professor I've worked with on research or projects have been using R. And I can see where R may have an advantage, especially for EDA and visualizations, but I tend to favor Python simply for the ML, which I think is much more robust across nearly all areas.
# 
# Anyways, doesn't matter what I think, let's see what the Kaggle community thinks. Where should aspiring data scientists focus their efforts?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

directory = '/kaggle/input/kaggle-survey-2019/'

df_mcr = pd.read_csv(directory+'multiple_choice_responses.csv', low_memory=False)
df_questions = pd.read_csv(directory+'questions_only.csv', low_memory=False)
cols_to_use = ['Time from Start to Finish (seconds)', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8',
              'Q10', 'Q11', 'Q14', 'Q15', 'Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5',
              'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12',
              'Q19', 'Q22', 'Q23']
df_mcr = df_mcr[cols_to_use]
df_mcr = df_mcr.drop([0], axis=0)


# # The broad opinion: Python
# 
# Python seems to be the recommendation across the Kaggle community for aspiring data scientists to learn first. R is the closest runner up, then SQL, then C++, then... MATLAB?...
# 
# Personally I'd find Bash more useful than the rest after SQL... Maybe if we looked at people experienced in machine learning implementation we'll get some different results. Besides, an expert would need experience, right?

# In[ ]:


# Overall opinion of what language to learn first
language_one = df_mcr.loc[(df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript')]['Q19'].value_counts(normalize=True)
fig, ax = plt.subplots()
ax.pie(language_one)
ax.legend(labels=language_one.index)
plt.show()

language_one


# # Still in the lead: Python
# 
# No surprise that Python continues to be the most recommended language to learn first for aspiring data scientists, though, as we've excluded people with less experience, the results seem to be less concrete. You'll also notice that MATLAB has actually moved up as a recommendation, which, makes sense at some level, as it may be easy and helpful for people to grasp some simple math and statistics before they start learning about more technical aspects.

# In[ ]:


# Share of 'expert' first language recommendation
experts = df_mcr.loc[((df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript'))&((df_mcr.Q23=='5-10 years')|(df_mcr.Q23=='10-15 years')|(df_mcr.Q23=='20+ years'))]['Q19'].value_counts(normalize=True)
fig, ax = plt.subplots()
ax.pie(experts)
ax.legend(labels=experts.index)
plt.show()

experts


# # Experience, experience, experience...
# 
# So, experience is always a good signifier of how knowledgeable a person may be about a craft or skillset. But, data science is a broad topic. So, maybe some of these recommendations aren't catered for what an aspiring data scientist would like to do.
# 
# Let's look at what Kagglers in specific job positions have to recommend.

# In[ ]:


for title in df_mcr.Q5.value_counts().index:
    title_vc = df_mcr.loc[((df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript'))&((df_mcr.Q23=='5-10 years')|(df_mcr.Q23=='10-15 years')|(df_mcr.Q23=='20+ years'))&((df_mcr.Q5==title))]['Q19'].value_counts(normalize=True)
    print('\n\n'+title + ' W/ 5 or more years ML experience')
    fig, ax = plt.subplots()
    ax.pie(title_vc)
    ax.legend(title_vc.index)
    plt.show()
    print('% share of languages recommended by: '+title)
    print(title_vc)


# # What about education?
# 
# Though experience may be nice, sometimes experience might not be enough. Rather than looking at job title, let's see the different suggestions per level of education.

# In[ ]:


learned = df_mcr.Q4.value_counts()
fig, ax = plt.subplots()
ax.barh(learned.index, learned)

plt.show()


# # Highly Educated
# 
# For the most part, the Kaggle community seems to be highly educated.
# 

# In[ ]:


for level in df_mcr.Q4.value_counts().index:
    ed_vc = df_mcr.loc[((df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript'))&((df_mcr.Q23=='5-10 years')|(df_mcr.Q23=='10-15 years')|(df_mcr.Q23=='20+ years'))&((df_mcr.Q4==level))]['Q19'].value_counts(normalize=True)
    print('\n\n'+level + ' W/ 5 or more years ML experience:')
    fig, ax = plt.subplots()
    ax.pie(ed_vc)
    ax.legend(ed_vc.index)
    plt.show()
    print('% share of languages recommended by: '+level)
    print(ed_vc)


# # Educated or not: Python
# 
# It doesn't seem as if education has an impact on what language is predominantly recommended to learn first for aspiring data scientists.
# 
# 
# # Kagglers Coding or Coding Kagglers?
# 
# So far, the simple statistical analyses performed in this notebook suggests that experienced Kagglers recommend that aspiring data scientists should learn Python as a first programming language. The only point of analysis in this notebook that challenges this notion is the analysis of recommended languages to learn first by job title. Statisticians and Data Analysts are the two job titles that have greater variations of which language an aspiring data scientist should learn first.
# 
# ## What I'd ask next year:
# 
# These responses from Kagglers about data science and the state of ML gives a nice overview. But, what would be nice to ask in another survey would be:
# 
# 1. What programming language do you use most regularly?
# 2. How important has traditional education been to your career?
# 3. Explain your reasoning for which language you'd recommend for an aspiring data scientist to learn first?
# 
# When thinking about what a data scientist will be doing on a daily basis, it may be hard to envision where you'll end up. And it's easy to think that the decisions you make early on in your career or education will have massive impacts on where you end up. But, for now, start with Python.
