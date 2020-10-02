#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import spearmanr


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.


# In[181]:


schemaFile, resultsFile = os.listdir("../input")
schema =  pd.read_csv("../input/" + schemaFile, dtype = str)
results =  pd.read_csv("../input/" + resultsFile, dtype = str)
results.describe()


# In[122]:


pd.options.display.max_colwidth = 300  ## to display the whole string instead of truncating at default 50 chars
schema


# Does job satisfaction correlate with the size of the company the respondents work for?

# In[123]:


results['CompanySize'].unique()
results['CompanySize'].value_counts()


# In[124]:


results['JobSatisfaction'].unique()


# In[158]:


# results['CompanySize'] = results['CompanySize'].astype('category')
# results['CompanySize'].head()
# results['CompanySizeR'] = results['CompanySize'].cat.reorder_categories(['Fewer than 10 employees','10 to 19 employees',
#                                                '20 to 99 employees', '100 to 499 employees',
#                                               '500 to 999 employees', '1,000 to 4,999 employees',
#                                               '5,000 to 9,999 employees', '10,000 or more employees'])
# results['CompanySizeR'].tail()


# In[137]:


# Frequency table CompanySize vs JobSatisfaction
JSvsCompSize = pd.crosstab(results.CompanySize,results.JobSatisfaction,) 
JSvsCompSize



# Why isnt't the order of the categories maintained? How can I fix this?

# In[138]:


JSvsCompSize.plot(kind="bar",stacked=True, # kind. kind, not type. moron.
                  figsize=(12,5),
                  color = ["#42f4eb","#175955","#8850ba","#d235f2","#f90ed6","#77574a","#14bab4"]) 
plt.xlabel("")
plt.ylabel("Number of respondents")
plt.title("Job satisfaction vs company size")
plt.show()


# Wow the responses in each bin seem so smilarly distributed. Maybe only zoom in on percentage of extremely satisfied respondents per company size category.
# 

# In[182]:


JSvsCompSizeNorm = pd.crosstab(results.CompanySize,results.JobSatisfaction,normalize="index") 
JSvsCompSizeNorm['Extremely satisfied']=JSvsCompSizeNorm['Extremely satisfied']*100
JSvsCompSizeNorm['Extremely satisfied'].sort_values().plot(kind="bar",
                                                           figsize=(9,4),
                                                           width=0.7,
                                                           color = ["#42f4eb","#175955","#8850ba","#d235f2","#f90ed6","#77574a","#14bab4","#e52b82"])
plt.title("Percentage of extremely satisfied employees vs company size")
plt.ylabel("[%]")
plt.xlabel("")
plt.show()


# There only seems to be a slightly higher percentage of super satisfied people in the smallest of companies, with fewer than 10 employees. 
# 
# What about the time from the last new job?

# In[142]:


# not really interested in those who've never had a job
JSvsLastNew = pd.crosstab(results.LastNewJob,results.JobSatisfaction) 
JSvsLastNew = JSvsLastNew.loc[['Less than a year ago',
                    'Between 1 and 2 years ago',
                    'Between 2 and 4 years ago',
                    'More than 4 years ago']]
        


# In[156]:


JSvsLastNew.plot(kind="bar",stacked=True,
                 figsize=(9,7),
                 color = ["#42f4eb","#175955","#8850ba","#d235f2","#f90ed6","#77574a","#d5e710"])
plt.title("Job satisfaction by the temporal distance from the last new job")
plt.show()


# In[183]:


# And the numbers say...

results['JobSatisfaction'] = results['JobSatisfaction'].astype(pd.api.types.CategoricalDtype(categories=['Extremely dissatisfied',
                                                                                                         'Moderately dissatisfied',
                                                                                                         'Slightly dissatisfied',
                                                                                                         'Neither satisfied nor dissatisfied',
                                                                                                         'Slightly satisfied',
                                                                                                         'Moderately satisfied',
                                                                                                         'Extremely satisfied'],
                                                                                             ordered=True)
    
                                                              )


results['JobSatisfaction'].head()


# In[184]:


results['JobSatisfactionCat'] = results['JobSatisfaction'].cat.codes
results['JobSatisfactionCat'].head()


# In[187]:


resultsNoSlackers = results.loc[results.LastNewJob.isin(['Less than a year ago',
                                         'Between 1 and 2 years ago',
                                         'Between 2 and 4 years ago',
                                         'More than 4 years ago'])]
resultsNoSlackers.LastNewJob = resultsNoSlackers.LastNewJob.astype(pd.api.types.CategoricalDtype(
                                                         categories=['Less than a year ago',
                                                                     'Between 1 and 2 years ago',
                                                                     'Between 2 and 4 years ago',
                                                                     'More than 4 years ago'],
                                                         ordered = True))
                                              
resultsNoSlackers['LastNewJobCat'] = resultsNoSlackers['LastNewJob'].cat.codes
spearmanr(resultsNoSlackers.LastNewJobCat,resultsNoSlackers['JobSatisfactionCat'])


# Well the numbers agree it's totally not correlated.  Heh.

# Aaaand... does one's job satisfaction have an effect on their contribution to open source code?

# In[188]:


bla =  sns.FacetGrid(resultsNoSlackers,col = "JobSatisfaction")
bla.map(sns.countplot,"OpenSource", order=["Yes","No"])
bla.set_xticklabels(rotation = 90)
plt.show()


# Mh, so and so. The extremely satisfied have equal numbers of contributors and non-contributors, whereas in other groups there is  a prevalence of non contributors.

# In[189]:


resultsNoSlackers.Gender[-resultsNoSlackers.Gender.isin(["Male","Female"])] = "Other"


# In[190]:


resultsNoSlackers.Gender = resultsNoSlackers.Gender.astype('category')

JSvsGender =  resultsNoSlackers.JobSatisfaction.groupby(resultsNoSlackers.Gender).value_counts()
a = pd.DataFrame(JSvsGender)
f = a.xs('Female')
m = a.xs('Male')
o = a.xs('Other')

plt.style.use('default')
plt.figure(figsize=(22,3))

plt.subplot(131)
f.JobSatisfaction.plot(kind='pie',
                       autopct='%.1f')
plt.ylabel('')
plt.axis('equal')
plt.title("Gender: Female")

plt.subplot(132)
m.JobSatisfaction.plot(kind='pie',autopct='%.1f')
plt.ylabel('')
plt.axis('equal')
plt.title("Gender: Male")

plt.subplot(133)
o.JobSatisfaction.plot(kind='pie',autopct='%.1f')
plt.ylabel('')
plt.axis('equal')
plt.title("Gender: Other")


plt.show()


# Females have a slightly higher percentage of extremely satisfied respondents (19.3%), whereas those who didn't identify with these two genders have the lowest score (16.4%), as well as the highest percentage of extremely dissatisfied respondents (4.4%). 
# 
# But let us observe the total satisfaction, not just the extremes. If we code the responses as a scale from 0 to 6, we can take some descriptive stats:

# In[194]:


resultsNoSlackers.JobSatisfaction.cat.codes.head()


# In[200]:


resultsNoSlackers.JobSatisfaction.cat.categories


# In[201]:


resultsNoSlackers.JobSatisfactionCat.unique()


# In[202]:


temp = resultsNoSlackers[resultsNoSlackers.JobSatisfactionCat!=-1] #don't take NaNs into consideration
JSvsGenderStats =  temp.JobSatisfactionCat.groupby(temp.Gender).describe()
JSvsGenderStats


# In[197]:


temp.JobSatisfactionCat.groupby(temp.Gender).median()


# Firstly, the genders are not equally represented, there are significantly less women taking the survey than non binary, and especially male respondents.
# 
# Secondly, the median satisfaction is the same for all 3 gender categories (5 - that's "Moderately Satisfied").
# 
# Males and females have roughly equal mean satisfaction (4,07 and 4.05 respectively), and non binary have somewhat lower mean score of 3.95.
# 
# Standard deviations are almost the same for all categories.

# In[203]:


temp.boxplot(column = 'JobSatisfactionCat',by="Gender")
plt.xlabel("")
plt.show()


# In[213]:


resultsNoSlackers.JobSatisfaction.value_counts().plot(kind='bar',figsize=(5,4))
plt.title("Job Satisfaction irrespective of everything else")
plt.show()


# It would be safe to say that generally developers are moderately satisfied with their jobs.
# 
