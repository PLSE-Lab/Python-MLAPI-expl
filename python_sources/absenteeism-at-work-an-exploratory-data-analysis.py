#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Seaborn for visualization
import seaborn as sns
import matplotlib.pyplot as plt

## initial viz settings ##
sns.set(font_scale=1.2)
col = '#3B1C8C'
cm = 'viridis'


# # 1. Data reading and little bit of wrangling

# In[ ]:


df = pd.read_csv('/kaggle/input/absenteeism-at-work-uci-ml-repositiory/Absenteeism_at_work.csv', sep=';')

nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df.head(10)


# In[ ]:


# Rename columns
df.rename(columns={'Reason for absence': 'Reason',
                  'Month of absence': 'Month',
                  'Day of the week': 'Weekday',
                   'Seasons': 'Season',
                   'Disciplinary failure': 'Failure',
                  'Transportation expense': 'Expense',
                  'Distance from Residence to Work': 'Distance',
                  'Service time': 'ServiceTime',
                  'Work load Average/day ': 'Workload',
                   'Hit target': 'HitTarget',
                   'Son': 'Child',
                  'Social drinker': 'Drinker',
                  'Social smoker': 'Smoker',
                  'Body mass index': 'BMI',
                  'Absenteeism time in hours': 'AbsH'}, 
          inplace = True)
df.head(10)


# In[ ]:


# add the string values for education
#high school (1), graduate (2), postgraduate (3), master and doctor (4)

df.loc[(df['Education'] == 1),'Edu_Text'] = 'High school'
df.loc[(df['Education'] == 2),'Edu_Text'] = 'Graduate'
df.loc[(df['Education'] == 3),'Edu_Text'] = 'Postgraduate'
df.loc[(df['Education'] == 4),'Edu_Text'] = 'Master and doctor'
df.head(10)


# In[ ]:


# Convert  attributes to categories

df['Month'] = df['Month'].astype('category')
df['Weekday'] = df['Weekday'].astype('category')
df['Reason'] = df['Reason'].astype('category')
df['Failure'] = df['Failure'].astype('category')
df['Season'] = df['Season'].astype('category')
df['Education'] = df['Education'].astype('category')
df['Edu_Text'] = df['Edu_Text'].astype('category')
df['Drinker'] = df['Drinker'].astype('category')
df['Smoker'] = df['Smoker'].astype('category')


# # 2. Data Exploration

# # 2.1 Explore Observations

# ### What one can see per observation (row)
# 
# * how many hours an employee was absent
# * for a certain reason
# * in a certain month
# * on a certain weekday
# * over 3 years
# * plus some work-related data like expense for traveling to work, distance, service time
# * plus some personal data for each ID like age, education, childs
# 
# 

# In[ ]:


df.describe(include = 'all')


# ### Histogram of Absence Hours

# In[ ]:


plt.figure(figsize = (15, 6))
sns.distplot(df['AbsH'], color = col, kde = False)
plt.show()


# ### Absence hours per reason

# In[ ]:


plt.figure(figsize = (15, 6))
sns.boxplot(x = df['Reason'], y = df['AbsH'])
plt.show()


# What one can see in the boxplot:
# 
# Reason 9 (Diseases of the circulatory system) causes longer absences, followed by Reason 12  	(Diseases of the skin and subcutaneous tissue).

# ### Top 5: Most absence hours per observation

# In[ ]:


df[['ID','Month', 'Weekday','Reason', 'AbsH']] .sort_values(['AbsH', 'ID'], ascending = [0, 1])[:5]


# What one can see: 
# 
# Employee 9 has been absent for 120h (=15d) on Tuesdays in July over 3 years for reason 6 (Diseases of the nervous system) and 112h (=14d) on Tuesdays in March for reason 12 (Diseases of the skin and subcutaneous tissue).
# 
# What is not clear: 
# 
# How the absence hours are distributed over the years. In the case of the 120h, the employee must have been absent almost every Tuesday in July over the three years. If there are e.g. 8h, one doesn't know when that happened and how the hours are distributed.
# 

# ### Top 5: Least absence hours per observation

# In[ ]:


df[['ID','Month', 'Weekday','Reason', 'AbsH']] .sort_values(['AbsH', 'ID'], ascending = [1, 1])[:5]


# What one can see: 
# 
# There are some absence hours = 0, reason = 0 and also month = 0.
# 
# That looks like the dataset contains all the employees, and some haven't been absent in certain months or have never absent (month = 0).

# ### Employees without any absence

# In[ ]:


df[df['Month'] == 0][['ID','Month', 'Weekday','Reason', 'AbsH']]


# What one can see:
# 
# There are 3 employees without any absence hours during the 3 years.
# 
# What is not clear: 
# 
# When these 3 employees have started to work for the company - maybe that was just a month before the data collection has finished.

# # 2.2 Explore monthly data

# ### Absence hours per month and per weekday
# Mondays and March have the highest sum of absence hours.

# In[ ]:


plt.figure(figsize = (15,4))
plt.subplot(1, 2, 1)
sns.barplot(df.groupby(['Weekday'])['AbsH'].sum().index, 
            df.groupby(['Weekday'])['AbsH'].sum(), 
            color = col)
plt.subplot(1, 2, 2)
sns.barplot(df.groupby(['Month'])['AbsH'].sum().index, 
            df.groupby(['Month'])['AbsH'].sum(), 
            color = col)
plt.show()


# But "month" and "day" in combination gives a different result:
# 
# The Tuesdays in July have the highest sum of absence hours, 2nd are the Tuesdays in December.

# In[ ]:


plt.figure(figsize = (15, 6))
sns.heatmap(df.groupby(['Month', 'Weekday'])['AbsH'].sum().unstack()[1:13], 
            annot = True, fmt = 'g', cmap = cm) 
plt.show()


# ### Number of employees per month and weekday
# 
# What one can see: 
# 
# - Most employees are absent on Wednesdays in May, 2nd are Mondays in July and Tuesdays in September.
# 
# Those analyses can help the company in two ways:
# 
# - Look for reasons: was there something special in the past?
# - Prepare for future: some more employees / helper can support in in case of bottlenecks.
# 
# 

# In[ ]:


plt.figure(figsize = (15, 6))
sns.heatmap(df.groupby(['Month', 'Weekday'])['ID'].nunique().unstack()[1:13], cmap = cm, annot = True, fmt = 'g')
plt.show()


# # 2.3 Explore the reasons

# ### Create a new df based on data per reason

# In[ ]:


#hours_per_reason = df.groupby('Reason')['AbsH'].sum()
#id_per_reason = df.groupby('Reason')['ID'].nunique()

reason_df = pd.concat([df.groupby('Reason')['AbsH'].sum(), df.groupby('Reason')['ID'].nunique()], axis=1, sort=False)
reason_df.columns = ['AbsH', 'AbsIDs']

reason_df.head(5)


# ### Absence hours and number of employees per peason
# 
# What one can see:
# 
# - Reason 13 (musculoskeletal system and connective tissue) causes the most absence hours, and 18 employees are affected.
# - Reason 19 (Injury, poisoning and certain other consequences of external causes) has the 2nd place, 16 employees are affected
# - Reason 23 (medical consultation) causes the 3rd most absence hours and most of the employees (24) are affected.
# - Reason 20 (external causes of morbidity and mortality) includes traffic accident: just an additional ICD-10 code and not included in that dataset
# 
# 

# In[ ]:


plt.figure(figsize = (12,6))
p1 = sns.scatterplot(x=reason_df.index[1:], y="AbsH", size="AbsIDs", hue ="AbsIDs", palette = cm,
                sizes=(20, 1000),data=reason_df[1:], markers = True)
plt.legend(loc='upper left')
plt.show()


# ### Cumulated absence hours
# 
# What one can see: only five reasons cause more than 50% of absence hours
# 
# - 13: Diseases of the musculoskeletal system and connective tissue
# - 19: Injury, poisoning and certain other consequences of external causes
# - 23: medical consultation
# - 28: dental consultation
# - 11: Diseases of the digestive system

# In[ ]:


reason_df = reason_df.sort_values('AbsH', ascending = 0)
reason_df['cum_sum'] = reason_df['AbsH'].cumsum()
reason_df['cum_perc'] = 100 * reason_df['cum_sum'] / reason_df['AbsH'].sum()

round(reason_df[:15],1).T


# ### Sum of absence hours per month and reason
# 
# The two top categories stick out:
# 
# - 13: Diseases of the musculoskeletal system and connective tissue - peak in April
# - 19: Injury, poisoning and certain other consequences of external causes - peak in March
# 
# 

# In[ ]:


plt.figure(figsize = (18, 6))
sns.heatmap(df[df['Reason'] != 0].groupby(['Month', 'Reason'])['AbsH'].sum().unstack(), 
            cmap = cm, annot = True, fmt = 'g')
plt.show() 


# ### Number of IDs (employees) per month and reason
# 
# Reasons with the highest number of absent employees:
# 
# - 23: medical consultation - less during Brazilian summer
# - 28: dental consultation - more or less evenly distributed
# 
# 

# In[ ]:


plt.figure(figsize = (18, 6))
ax = sns.heatmap(df[df['Reason'] != 0].groupby(['Month', 'Reason'])['ID'].nunique().unstack(), 
                 cmap = cm, annot = True, fmt = 'g')

plt.xlabel('Reason')
plt.show()


# # 2.4 Explore individual employee data

# ### Create a new df based on individual employee data

# In[ ]:


age = df['Age'].groupby(df['ID']).max()
edu = pd.to_numeric(df['Education']).groupby(df['ID']).max().astype('category')
son = df['Child'].groupby(df['ID']).max()
drink = pd.to_numeric(df['Drinker']).groupby(df['ID']).max().astype('category')
smoke = pd.to_numeric(df['Smoker']).groupby(df['ID']).max().astype('category')
fail = pd.to_numeric(df['Failure']).groupby(df['ID']).max().astype('category')
pet = df['Pet'].groupby(df['ID']).max()
service = df['ServiceTime'].groupby(df['ID']).max()
weight = df['Weight'].groupby(df['ID']).max()
height = df['Height'].groupby(df['ID']).max()
bmi = df['BMI'].groupby(df['ID']).max()
exp = df['Expense'].groupby(df['ID']).max()
dist = df['Distance'].groupby(df['ID']).max()
absh = df['AbsH'].groupby(df['ID']).sum()
reason = df[df['Reason'] != 0].groupby('ID')['Reason'].nunique()

#new features 
hitmax = df['HitTarget'].groupby(df['ID']).max()
hitmin = df['HitTarget'].groupby(df['ID']).min()
hitmean = round(df['HitTarget'].groupby(df['ID']).mean(),2)
wlmax = round(df['Workload'].groupby(df['ID']).max(),2)
wlmin = round(df['Workload'].groupby(df['ID']).min(),2)
wlmean = round(df['Workload'].groupby(df['ID']).mean(),2)


# In[ ]:


ind_df = pd.concat([age, edu, son, drink, smoke, fail, pet, hitmax, hitmin, hitmean, 
                    wlmax, wlmin, wlmean, service, weight, height, bmi, exp,dist,reason,absh], axis=1, sort=False)

ind_df.columns = ['Age', 'Education', 'Child', 'Drinker', 'Smoker', 'Failure', 'Pet',
       'HT_Max', 'HT_Min', 'HT_Mean', 'WL_Max', 'WL_Min',
       'WL_Mean', 'ServiceTime', 'Weight', 'Height', 'BMI', 'Expense',
       'Distance', 'R_Count', 'AH_Sum']


# In[ ]:


ind_df.loc[(ind_df['Education'] == 1),'Edu_Text'] = 'High school'
ind_df.loc[(ind_df['Education'] == 2),'Edu_Text'] = 'Graduate'
ind_df.loc[(ind_df['Education'] == 3),'Edu_Text'] = 'Postgraduate'
ind_df.loc[(ind_df['Education'] == 4),'Edu_Text'] = 'Master and doctor'


# In[ ]:


# employees without absence has reason_count = 0
ind_df['R_Count'] = ind_df['R_Count'].fillna(0)


# In[ ]:


ind_df.head()


# ### Workload per employee
# 
# The daily workload differs: for some is the average workload as high as the maximum for others.

# In[ ]:


plt.figure(figsize = (12,6))
sns.scatterplot(ind_df.index, ind_df['WL_Max'], color = '#450256', s = 80, label = 'Max')
plt.scatter(ind_df.index, ind_df['WL_Mean'], color = '#5AC865',  s = 80, label = 'Mean')
plt.scatter(ind_df.index, ind_df['WL_Min'], color = '#21908D',  s = 80, label = 'Min')
plt.ylabel('Workload')
plt.xlabel('Employee')
plt.legend(fontsize  = 'large', bbox_to_anchor=(1, 0.5))
plt.show()


# ### Are some attributes correlated?
# 
# Positive correlated are:
# 
# - Age & ServiceTime
# - HitTargetMaximum & Number of Reasons
# - HitTargetMaximum & Sum of Absence Hours
# - and also Min/Max/Mean of HitTarget, Workload in some cases
# 
# Negative correlated are:
# 
# - HitTargetMinimum & Sum of Absence Hours
# - WorkloadMinimum & Number of Reasons

# In[ ]:


plt.figure(figsize = (8,6))

corr = ind_df.drop(['Weight', 'Height'], axis=1).corr()

ax = sns.heatmap(
    corr[((corr <= -0.5) | (corr >= 0.5)) & (corr != 1)], 
    vmin = -1, vmax = 1, center = 0,
    cmap = cm,
    square = True,
    linewidths = .5
    #annot = True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 90,
    horizontalalignment = 'right'
)
plt.show() 


# In[ ]:


plt.figure(figsize = (8,6))
sns.regplot(ind_df['Age'], ind_df['ServiceTime'], color = col, scatter_kws = {'s':80})
plt.show()


# In[ ]:


plt.figure(figsize = (8,6))
sns.regplot(ind_df['HT_Min'], ind_df['AH_Sum'], color = col, scatter_kws = {'s':80})
plt.show()


# In[ ]:




