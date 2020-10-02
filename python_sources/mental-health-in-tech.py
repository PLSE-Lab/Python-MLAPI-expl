#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dataset = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')
# print(dataset.head())
print(dataset.columns)


# # Data Cleaning 
# For the gender column, printing the unique values revealed variations in capitalization and the use of abbreviations. These were replaced with either 'Male' or 'Female' using regex. The remaining values included uncommon misspellings and attempts at humour. Lumping transgendered/non-binary people into the same category ('other') as trolls didn't seem appropriate, so all of the data was simply omitted. 

# In[ ]:


print(dataset.Gender.unique())
dataset['Gender'] = dataset['Gender'].replace(to_replace = '^[mM]$|male', value='Male', regex=True)
dataset['Gender'] = dataset['Gender'].replace(to_replace = '^[Ff]$|[Ff]e[Mm]ale', value='Female', regex=True)
dataset = dataset[(dataset['Gender'] == 'Male')|(dataset['Gender'] == 'Female')]
print(dataset.Gender.unique())


# Next, I took a look at the age column. Since it's impossible to be over 300 or under -1700 years old, the values were restricted to between 12 (the youngest legal working age in a developed country as far as I know) and 100 (apologies to anyone above that age who are still actively employed). 

# In[ ]:


column = dataset['Age']
print(np.mean(column))
print(np.min(column))
print(np.max(column))
dataset = dataset[dataset['Age'].between(12, 100)]
new_column = dataset['Age']
print(np.mean(new_column))
print(np.min(new_column))
print(np.max(new_column))


# # Exploratory Data Analysis
# First, a look at the demographics of survey respondents. The vast majority were male, but the age distribution was similar for both genders.

# In[ ]:


gender = dataset.groupby('Gender')['Timestamp'].count().reset_index()
print(gender)
x = gender['Gender']
y = gender['Timestamp']
sns.barplot(x, y)
plt.ylabel('Count')
plt.show()

x = dataset['Gender']
y = dataset['Age']
sns.violinplot(x, y)
plt.show()


# At this point, while my original intention was to methodically go through the data and explore as much as I can, I have already honed in on an area of interest. Most of the survey questions centered around either benefits/treatment/care provided by employers or willingness to discuss mental health in the workplace. As a result, I'm curious about whether one or both of these factors correlate with if an employee seeks help, or the extent to which mental health interferes with work.
# 
# First, I converted columns 13-17 into dummy variables and summed up the yes values into an "availability" factor. Each survey respondent would therefore be assigned a value of 0 (employer provides no resources that they are aware of) to 5 (employer provides all of the resources that this survey covers). 

# In[ ]:


dataset.rename(columns={'care_options': 'care',
                       'wellness_program': 'wellness',
                       'seek_help': 'help'}, inplace=True)
available = pd.get_dummies(dataset[['benefits', 'care', 'wellness', 'help', 'anonymity']])
available = available.T.groupby([s.split('_')[1] for s in available.T.index.values]).sum().T
available['Uncertain'] = available["Don't know"] + available['Not sure']
available = available[['No', 'Yes', 'Uncertain']]
available.rename(columns={'No': 'avai_no',
                         'Yes': 'avai_yes',
                         'Uncertain': 'avai_uncertain'}, inplace=True)
print(available.head())
dataset = dataset.merge(available, left_index=True, right_index=True)


# From just looking at the bar graph for mean availability and treatment seeking, it's apparent that people who seek treatment have/know more about what resources are available to them. A two-sample t-test confirms that there is a significant difference in mean availability between people who seek help and people who do not.

# In[ ]:


from scipy.stats import ttest_ind

x = dataset['treatment']
y = dataset['avai_yes']
sns.barplot(x, y)
plt.xlabel('Have Sought Treatment')
plt.ylabel('Availability of Resources')
plt.show()

group_1 = dataset.loc[dataset['treatment'] == 'Yes', 'avai_yes']
group_2 = dataset.loc[dataset['treatment'] == 'No', 'avai_yes']
tstat, pval = ttest_ind(group_1, group_2)
print(pval)


# The relationship between mean availability and interference with work is less clear from the bar graph alone. After performing ANOVA and Tukey's Range Test, however, it is revealed that people who have mental health issues that **never** interfere with their work are significantly less aware of/able to access resources than people whose issues **rarely** or **sometimes** interfere with their work, but not people whose issues **often** interfere with their work.

# In[ ]:


from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

x = dataset['work_interfere']
y = dataset['avai_yes']
sns.barplot(x, y)
plt.xlabel('Interference with Work')
plt.ylabel('Availability of Resources')
plt.show()

o = dataset.loc[dataset['work_interfere'] == 'Often', 'avai_yes']
r = dataset.loc[dataset['work_interfere'] == 'Rarely', 'avai_yes']
n = dataset.loc[dataset['work_interfere'] == 'Never', 'avai_yes']
s = dataset.loc[dataset['work_interfere'] == 'Sometimes', 'avai_yes']
pval = f_oneway(o, r, n, s).pvalue
print(pval)

v = np.concatenate([n, r, s, o])
labels = ['n']*len(n) + ['r']*len(r) + ['s']*len(s) + ['o']*len(o)
tukey_results = pairwise_tukeyhsd(v, labels, 0.05)
print(tukey_results)


# To explore this further, let's take a look at not being able to access resources and not knowing about resources separately. 
