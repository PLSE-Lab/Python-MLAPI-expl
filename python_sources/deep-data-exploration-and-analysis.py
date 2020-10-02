#!/usr/bin/env python
# coding: utf-8

# #### In this notebook, I will do a profound data exploration since I am curious about all aspects of the data, especially this dataset has upto 35 columns.
# #### Data Visualization in this notebook is created using seaborn & bokeh.
# #### Now, let's get started!!

# # Import modules

# In[ ]:


# zipfile module to work on zip file
from zipfile import ZipFile

import pandas as pd
from pprint import pprint
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


# In[ ]:


#with ZipFile('human-resources-data-set.zip', 'r') as zipf:
    # print all contents of zip file
    #zipf.printdir()
    # extracting all the files
    #zipf.extractall()


# In[ ]:


df = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')
df.head()


# # Data Exploration

# #### I divide data exploration into 2 parts:
# 
# #### - Check null values
# #### - Address float values

# ## Check null values

# In[ ]:


df.info()


# #### This dataframe has 401 rows while almost all columns have upto 310 non-null values. Let's drop rows that contain null values

# In[ ]:


df.dropna(how = 'all', axis = 0, inplace=True)


# #### Perfect!
# #### As I stated earlier, almost all columns have 310 non-null values, except some such as: DateofTermination, TermReason, ManagerID, LastPerformanceReview_Date, DaysLateLast30. Let's explore them further to understand why?

# ### DateofTermination & TermReason

# #### I start with DateofTermination and TermReason. 

# In[ ]:


df['DateofTermination'].unique()


# #### DateofTermination has 2 kinds of values: first one with date and second one with NaN. A date indicates when the employee quit his job and when this date is not available, a NaN value is on. How about TermReason column?

# In[ ]:


print('Number of null values in TermReason column:')
print(df['TermReason'].isnull().sum())
    
df['TermReason'].value_counts()


# #### Although almost employees are still employed, we have several different reasons for a leave.

# In[ ]:


df[['DateofTermination', 'TermReason']].head(10)


# #### Since DateofTerminaison and TermReason are related, I need to check:
# 
# #### - Whenever DateofTerminaison has a date value, is TermReason fulfilled?
# 
# #### - Whenever DateofTerminaison is NaN, what is written in TermReason?
# 
# #### Let's dive into this.

# In[ ]:


df[df['DateofTermination'].notnull()]['TermReason'].unique().tolist()


# #### OK for some leaves, I don't have any reason, seeing the NaN value's presence. But I believe this is not much, let's check:

# In[ ]:


print('Number of job leaves without any reason:')
df[df['DateofTermination'].notnull()]['TermReason'].isnull().sum()


# #### How about when DateofTermination is NaN? What is written in TermReason?

# In[ ]:


print(df[df['DateofTermination'].isnull()]['TermReason'].value_counts())
print('\n Number of employees whose DateofTermination and TermReason are both unknown:')
df[df['DateofTermination'].isnull()]['TermReason'].isnull().sum()


# #### OK, so employees with no DateofTermination either are still employed or have not yet started.
# 
# #### Next, let's explore ManagerID

# ### ManagerID

# In[ ]:


df.ManagerID.unique()


# #### So this case is classical, some values are there and some are NaN. However, if I check ManagerName column, I don't have any null values. Let's check these two columns together.
# 

# In[ ]:


df.ManagerName.unique()


# In[ ]:


mng = df[['ManagerID', 'ManagerName']]


# In[ ]:


mng_id = list(df.ManagerID.unique())
def check_mng(dataframe):
    for i in mng_id:
        print('ManagerID: ' + str(i) + '...' + str(dataframe[dataframe['ManagerID'] == i]['ManagerName'].unique().tolist()))
    # since there is NaN value, let's write a code to this this case specifically
    print('ManagerID: ' + 'nan' + '...' + str(dataframe[dataframe['ManagerID'].isnull()]['ManagerName'].unique().tolist()))
check_mng(mng)


# #### All Manager IDs that are NaN are assigned to only one person: Webster Butler, but he already has another ManagerID which is 39. Let's address this issue by replacing NaN by 39.
# #### We get two ManagerID for Brandon R. LeBlanc: 1 and 3; and for Michael Albert: 22 and 30. Maybe each one has two ManagerID since they are Manager of two different departments? Let's check.

# In[ ]:


# replace NaN value in ManagerID by 39
df.ManagerID.fillna(39.0, inplace = True)


# In[ ]:


df[df.ManagerName == 'Brandon R. LeBlanc'][['ManagerID', 'Department']]


# #### Brandon R. LeBlanc is Admin Offices's manager but he is assigned with 2 ManagerID. 3 might be a mistake. I will correct this and assign only one ManagerID (1) to him.

# In[ ]:


df.ManagerID.replace(3.0, 1.0, inplace = True)


# #### Next, let's check Michael Albert's case:

# In[ ]:


df[df.ManagerName == 'Michael Albert'][['ManagerID', 'Department']]


# #### Same issue, I will replace 30 by 22:

# In[ ]:


df.ManagerID.replace(30.0, 22.0, inplace = True)


# #### Let's check whether ManagerID issue is addressed correctly:

# In[ ]:


df.ManagerID.unique()


# ### LastPerformanceReview_Date and DaysLateLast30

# 
# #### Finally, since LastPerformanceReview_Date and DaysLateLast30 have same number of NaN values, let's explore them together.

# In[ ]:


df[['LastPerformanceReview_Date', 'DaysLateLast30']].head(10)


# #### First impression: an employee who has no LastPerformanceReview_Date has no value in DaysLateLast30 column. Let's verify this.

# In[ ]:


df[df.LastPerformanceReview_Date.isnull()].DaysLateLast30.unique()


# #### Great! My hypothesis is confirmed. Whenever an employee has no LastPerformanceReview_Date, he also has no value in DaysLateLast30 column. This is logical, and data is consistent on this point.
# #### I want also to see what is DateofTermination of those employees.
# #### Let's get the number of employees with no LastPerformanceReview_Date first.

# In[ ]:


print('Number of employees with no LastPerformanceReview_Date:')
df.LastPerformanceReview_Date.isnull().sum()


# #### 103 employees has no LastPerformanceReview_Date. How about their DateofTermination?

# In[ ]:


df[df.LastPerformanceReview_Date.isnull()].DateofTermination.unique()


# #### Perfect! So all 103 employees with no LastPerformanceReview_Date had already quit the company. which is logical: Our data is consistent.

# ## Address float values

# #### Let's explore each column with numeric values one by one. But first, I need to convert float values to integer.
# 

# #### What are columns that contain float values?

# In[ ]:


df.select_dtypes('float').head(10)


# In[ ]:


df.select_dtypes('float').columns


# #### Here I can see that some columns must remain integer such as PayRate, EngagementSurvey and EmpSatisfaction. Let's convert the other:

# In[ ]:


# select the columns that I want to convert float values into integer values
cols = ['EmpID', 'MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID',
       'DeptID', 'PerfScoreID', 'FromDiversityJobFairID', 'Termd',
       'PositionID', 'Zip', 'ManagerID', 'SpecialProjectsCount', 'DaysLateLast30']


# In[ ]:


for col in cols:
    df[col] = df[col].astype('Int32')


# In[ ]:


df.head()


# # Data Analysis

# ## Payrate, performance, engagement survey, employee satisfaction (based on sex and department)

# #### I would like to check how payrate, performance, engagement survey and employee satisfaction varies upon sex and department. I first need to extract only employees that are still employed.

# In[ ]:


# get rid of space in Department and Sex values
df.Sex = df.Sex.apply(lambda x: x.strip())
df.Department = df.Department.apply(lambda x: x.strip())


# In[ ]:


emp = df[df.DateofTermination.isnull()]
emp.shape


# ### Payrate

# #### Among 207 employees that are still in the company, how their payrate varies?

# In[ ]:


print('PayRate count based on Sex and Department')
pd.pivot_table(emp[['Sex', 'Department', 'PayRate']], index=['Sex'],
                    columns=['Department'], values = ['PayRate'], aggfunc=lambda x: int(len(x)))


# In[ ]:


print('PayRate mean based on Sex and Department')
pr = pd.pivot_table(emp[['Sex', 'Department', 'PayRate']], index=['Sex'],
                    columns=['Department'], values = ['PayRate'], aggfunc=np.mean)
pr


# In[ ]:


# PLOT
plt.figure(figsize=(10,5))
bplot=sns.stripplot(y='PayRate', x='Department', data=emp, jitter=True, dodge=True, marker='o', alpha=0.8, hue='Sex')
bplot.legend(loc='upper left')

plt.xticks(rotation=60, horizontalalignment='right');


# #### Interesting! While men and women in Production and Sales are paid at same pay rate, men has slightly higher pay in IT and in Admin Offices than women. In contrast, women in software engineering get much better pay that men.
# #### There is a NaN value in Executive Office since the CEO is a woman.
# 
# #### Let's do the same with Performance.

# ### Performance

# In[ ]:


print('PerfScoreID mean based on Sex and Department')
pfm = pd.pivot_table(emp[['Sex', 'Department', 'PerfScoreID']], index=['Sex'],
                    columns=['Department'], values = ['PerfScoreID'], aggfunc=np.mean)
pfm


# In[ ]:


# PLOT
plt.figure(figsize=(10,4))
swarm=sns.swarmplot(y='PerfScoreID', x='Department', data=emp, dodge=True, marker='o', alpha=0.8, hue='Sex')
swarm.legend(loc='lower right', ncol = 2)

plt.xticks(rotation=60, horizontalalignment='right');


# #### Interesting again:
# 
# #### - In Admin offices, Performance Score are same for men and women, while pay rate has discrepancy as saw earlier.
# #### - In Software Engineering, while Men has better Performance Score than Women, they are being less paid. Note also that there are 2 men in Software Engineering against 5 women.
# 
# #### Let's check data consistency between PerfScoreID and PerformanceScore.

# In[ ]:


df.PerfScoreID.unique()


# In[ ]:


df.PerformanceScore.unique()


# #### I assume that 1 = 'PIP', 2 = 'Needs Improvement', 3 = 'Fully Meets', 4 = 'Exceeds' but let's verify this.

# In[ ]:


perf = df[['PerfScoreID', 'PerformanceScore']]
perf.head(10)


# #### Let's write a function to do this

# In[ ]:


def check_perf(dataframe):
    for i in range(1,5):
        print(dataframe[dataframe['PerfScoreID'] == i]['PerformanceScore'].unique().tolist())
check_perf(perf)


# #### One category of Performance that has all my attention is 'Exceeds'. I want to know who are these people and what is their payrate?

# In[ ]:


perf4 = emp[emp.PerfScoreID == 4].groupby('Department').count()['EmpID']
perf4


# #### Woah, production employees really stand out. But how about the percentage against each department's number of employees?

# In[ ]:


per_dep = emp.groupby('Department').count()['EmpID']


# In[ ]:


percent = pd.merge(perf4, per_dep, on='Department', how='left', suffixes=('_count_perf4', '_count'))
percent['Percentage'] = percent.apply(lambda row: row.EmpID_count_perf4/row.EmpID_count*100, axis = 1) 
percent


# #### OK, the percentages show that each department (IT/IS, Production and Software Engineering) has approximately same ratio of outstanding employees.

# ### Engagement Survey

# #### Next, let's check Engagement Survey

# In[ ]:


print('EngagementSurvey mean based on Sex and Department')
eng_s = pd.pivot_table(emp[['Sex', 'Department', 'EngagementSurvey']], index=['Sex'],
                    columns=['Department'], values = ['EngagementSurvey'], aggfunc=np.mean)
eng_s


# In[ ]:


# PLOT
plt.figure(figsize=(10,5))
sns.set(style="ticks", palette="pastel")

box = sns.boxplot(x="Department", y="EngagementSurvey",
            hue="Sex", palette=["m", "g"],
            data=emp)
sns.despine(offset=10, trim=True)
box.legend(loc='lower right', ncol = 2)

plt.xticks(rotation=60, horizontalalignment='right');


# #### The CEO has highest engagement survey, no surprise, right?
# #### In Software Engineering, men has higher Engagement Survey than women, but they get less pay.
# #### In contrast, in Admin Offices, women has higher Engagement Survey than Men, while they are less paid.
# #### In Sales, women also has higher Engagement Survey than Men, but they get same pay.

# ### Employee Satisfaction

# In[ ]:


print('EmpSatisfaction mean based on Sex and Department')
emp_s = pd.pivot_table(emp[['Sex', 'Department', 'EmpSatisfaction']], index=['Sex'],
                    columns=['Department'], values = ['EmpSatisfaction'], aggfunc=np.mean)
emp_s


# In[ ]:


df.EmpSatisfaction.unique()


# In[ ]:


# PLOT
plt.figure(figsize=(10,4))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
violin = sns.violinplot(x="Department", y="EmpSatisfaction", hue="Sex",
               split=True,
               palette={"F": "y", "M": "b"},
               data=emp, scale = 'count')
sns.despine(left=True)
violin.legend(loc='upper right', ncol = 2)
violin.set_xticklabels(violin.get_xticklabels(), rotation=45, horizontalalignment='right');


# #### Employee Satisfaction are measured from 1 to 5, I guess 5 is highest satisfaction level.
# #### The CEO has neutral opinion: she is not very satisfied, not unsatisfied.
# #### Men in Admin Offices has quite higher level of satisfaction than women.
# #### In other areas, Employee Satisfaction is quite same for men and for women.

# ### Correlation between payrate, performance, engagement survey, employee satisfaction

# #### After checking several metrics based on sex and department, I would like to see if these metrics are correlated to each other.

# In[ ]:


corr = df[['PayRate', 'PerfScoreID', 'EngagementSurvey', 'EmpSatisfaction']].corr()
corr


# #### Let's splot these correlations in a heatmap

# In[ ]:


f, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, cmap='Accent');


# #### I have here quite low correlations, except EmpSatisfaction & PerfScoreID with correlation score 0.31, this is logical.

# ## Payrate, performance, engagement survey, employee satisfaction (based on race and department)

# #### Apart of sex, race is another interesting category to investigate. Let's dive into this.

# #### Let's check first each race's proportion in this company.

# In[ ]:


emp.RaceDesc.value_counts(normalize = True)


# #### The majority are white. 1/5 are black or African American. Hispanic is minority.

# In[ ]:


print('PayRate count based on Race and Department')
pd.pivot_table(emp[['RaceDesc', 'Department', 'PayRate']], index=['RaceDesc'],
                    columns=['Department'], values = ['PayRate'], aggfunc=lambda x: int(len(x)))


# #### What I can see here is the CEO is a white woman.

# # to be continued

# In[ ]:




