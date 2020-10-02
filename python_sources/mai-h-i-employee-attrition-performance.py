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


# Libraries

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot

import scipy
from scipy import stats
from sklearn.preprocessing import LabelEncoder

sns.set_palette('RdBu')


# <center><h1> IBM HR analysis</h1></center>

# <center>![alkir/Thinkstock](https://compote.slate.com/images/75d251f2-6d54-4839-bfb1-96f40b237ef4.jpg) </center>

# ## Introduction
# 
# Hello everyone,  this notebook is an assignment in CBD Robotics internship, in order to exploit my basic acknowledge.
# 
# ***The assignment concludes 2 main parts.***  
# #### First, show off fundamental EDA skills, include:  
#  * Visulization with matplotlib.pyplot, seaborn, and plotly
#  * Missing value treatment with isnull()..
#  * pd.DataFrame's methods: info(), describe(), head(), tail()..  
# 
# #### Second, statistically questions:
#  * Which key factors influence attrition rates?
#  * Which key factors influence satisfaction rates?
# 
# Also, the statistics  must use ***following testing***:
#  * T-test  
#  * ANOVA and MANOVA
#  * Correlation calculation in Pearman method.

# # Table of Contents
# 
# 1. Take a look at the dataset
# 2. Missing data
# 3. Descriptive statistic
# 4. EDA
# 5. Statistic Tesing
#  * Which key factors to Attrition?
#  * Which key factors to Job Satisfaction?
# 6. Final verdict
# 7. My potential mistakes

# # 1. Take a look

# In[ ]:


df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


print('Observations                                   : ', df.shape[0])
print('Features -- exclude Attrition and Satisfication: ', df.shape[1] - 2)


# In[ ]:


df.head(10)


# In[ ]:


df.tail(10)


# In[ ]:


df.columns


# # 2. Missing data
# 
# I ran isnull(), read data desciption, manually look for any kind of missing data. There is no NaN nor any type of missing data in this set.  
# 
# *** This dataset is way so clean.***

# In[ ]:


print('Nan data points: ', df.isnull().sum().sum())


# # 3. Descriptive statistic

# Both Attrition and JobSafisfaction are categorical, so there is just small room for descriptive statistic here, which is: ***count*** and ***percentage***.

# In[ ]:


df.Attrition.describe()


# In[ ]:


df.Attrition.value_counts()


# In[ ]:


df.JobSatisfaction.describe(percentiles=[0.01, 0.45, 0.90])


# # 4. EDA

# In[ ]:


# The big picture
fig = make_subplots(rows=1, cols=2,
                   specs=[[{"type": "bar"}, {"type": "domain"}]])

# Sketch smaller details
trace0 = go.Histogram(x=df['Attrition'], name='In number', marker={'color':['red', 'blue']},
                     showlegend=False)
trace1 = go.Pie(values=df['Attrition'].value_counts(), name='Percentage', labels=['No', 'Yes'],
               textinfo='label+percent')

# Add traces
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

# Customize
fig.update(layout_title_text='<b> Attrition </b>')
fig.update_layout(showlegend=False)

# Done
fig.show()


# In[ ]:


# The big picture
fig = make_subplots(rows=3, cols=2,
                   specs=[[{'rowspan':3}, {"type": "domain"}],
                          [None,          {"type": "domain"}],
                          [None,          {"type": "domain"}]])

# Sketch smaller details

## The bar chart - with Yes = negative columns.
labels = ['R&D', 'Sales', 'HR']

yes = df['Department'][df.Attrition=='Yes'].value_counts()
trace_yes = go.Bar(x=labels, y=-yes, marker={'color':'red'}, showlegend=False) 

no  = df['Department'][df.Attrition=='No'].value_counts()
trace_no  = go.Bar(x=labels, y=no, marker={'color':'blue'}, showlegend=False )

## Pie 1 -- upper right
RD = df['Attrition'][df.Department=='Research & Development'].value_counts()
trace_3   = go.Pie(labels=['No', 'Yes'], values=RD, name='RD')

## Pie 2
Sales = df['Attrition'][df.Department=='Sales'].value_counts()
trace_4   = go.Pie(labels=['No', 'Yes'], values=Sales, name='Sales')

## Pie 3
HR = df['Attrition'][df.Department=='Human Resources'].value_counts()
trace_5   = go.Pie(labels=['No', 'Yes'], values=HR, name='HR')

# Add traces
fig.append_trace(trace_yes, 1, 1)
fig.append_trace(trace_no, 1, 1)

fig.append_trace(trace_3, 1, 2)
fig.append_trace(trace_4, 2, 2)
fig.append_trace(trace_5, 3, 2)

# Customize
fig.update(layout_title_text='<b> Attrition by Department </b>')

# Done
fig.show()


# In[ ]:


fig = px.box(df, y='MonthlyIncome', x='Gender', color='Gender', 
             points='all', 
             color_discrete_map={'Female':'red', 'Male':'Green'})

fig.update(layout_title_text='<b> Monthly Income by Gender </b>')
fig.update_layout(showlegend=False)

fig.show()


# In[ ]:


# The big picture
fig = make_subplots(rows=6, cols=2,
                   specs=[[{'rowspan':6}, {"type": "domain"}], # 1  --  1
                          [None,          {"type": "domain"}], # 0  --  2
                          [None,          {"type": "domain"}], # 0  --  3
                          [None,          {"type": "domain"}], # 0  --  4
                          [None,          {"type": "domain"}], # 0  --  5
                          [None,          {"type": "domain"}]])# 0  --  6

# Sketching
## Bar chart
labels=['Life Sciences', 'Medical','Marketing', 'Technical Degree', 'Other', 'Human Resources']

yes = df['EducationField'][df.Attrition=='Yes'].value_counts(ascending=False)
no = df['EducationField'][df.Attrition=='No'].value_counts(ascending=False)

fig.add_bar(y=-yes, x=labels, col=1, row=1, marker={'color':'red'},  showlegend=False)
fig.add_bar(y=no,   x=labels, col=1, row=1, marker={'color':'blue'}, showlegend=False)

## Pie chart
LS     = df['Attrition'][df.EducationField=='Life Sciences'].value_counts()
Med    = df['Attrition'][df.EducationField=='Medical'].value_counts()
Mar    = df['Attrition'][df.EducationField=='Marketing'].value_counts()
Tech   = df['Attrition'][df.EducationField=='Technical Degree'].value_counts()
Other  = df['Attrition'][df.EducationField=='Other'].value_counts()
HR     = df['Attrition'][df.EducationField=='Human Resources'].value_counts()

fig.add_pie(labels=['No', 'Yes'], values=LS,    name='LS',    col=2, row=1)
fig.add_pie(labels=['No', 'Yes'], values=Med,   name='Med',   col=2, row=2)
fig.add_pie(labels=['No', 'Yes'], values=Mar,   name='Mar',   col=2, row=3)
fig.add_pie(labels=['No', 'Yes'], values=Tech,  name='Tech',  col=2, row=4)
fig.add_pie(labels=['No', 'Yes'], values=Other, name='Other', col=2, row=5)
fig.add_pie(labels=['No', 'Yes'], values=HR,    name='HR',    col=2, row=6)

# Customize
fig.update(layout_title_text='<b> Attrition by Education Field </b>')
# Done
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=2)

trace0 = go.Histogram(x=df['Department'], y=df['JobSatisfaction'], histfunc='avg')
trace1 = go.Histogram(x=df['EducationField'], y=df['JobSatisfaction'], histfunc='avg')
trace2 = go.Histogram(x=df['OverTime'], y=df['JobSatisfaction'], histfunc='avg')
trace3 = go.Histogram(x=df['MaritalStatus'], y=df['JobSatisfaction'], histfunc='avg')

fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 1, 2)
fig.add_trace(trace2, 2, 1)
fig.add_trace(trace3, 2, 2)
#fig = px.histogram(df, x='Department', y='JobSatisfaction',  histfunc='avg')


fig.show()


# In[ ]:


g = sns.FacetGrid(data=df, row = 'Attrition', col = 'JobSatisfaction')
g.map(plt.hist, 'MonthlyIncome', bins=10)


# In[ ]:


sns.catplot(x='EducationField', y='MonthlyIncome',  data=df,
           kind='violin')
plt.xticks(rotation=45)


# In[ ]:


sns.countplot(x='DistanceFromHome', data=df)


# In[ ]:


# DistanceFromHome -- Attrition
sns.catplot(x='Attrition', y='DistanceFromHome', data=df,
           kind='box')


# In[ ]:


fig = px.histogram(df, x='JobSatisfaction', color='JobSatisfaction')

fig.update_layout(title='<b> JobSatisfaction </b>',
                  xaxis={'tickmode': 'array',
                         'tickvals': [1, 2, 3, 4]})

fig.show()


# # 5. Statistic Tesing
# 

# ## My thoughts
# 
# ### The questions:
#  * Which key factors influence attrition rates?
#  * Which key factors influence satisfaction rates?
# 
# ### I am thinking
# 
# 1. ***I have not known any how-influence measure.*** I have several basic tools of hypothesis testing: t-test, ANOVA and MANOVA, but none of them directly returns whether this influences that, just giving statistic testing of mean, variance, and so on.  
# 
#  So, I paraphrase questions which key factors influence become: In the population of feature A (ex: MonthlyIncome), *** whether Yes-Attrition mean is statiscal different from No-Attrition mean?  ***
# If the differences are significant, we conclusion this features **influences** the attrition rate.
# 
# 
# 2. ***T-test downside is only working with every pair of variable***. One variable must be our target, Attrition or Satisfaction, the underconsidering features would be the other one. Therefore, it is impossible to directly apply t-test across all of features.  
# 
#  So, I make a work around. ***For *feature* in *all features*, does this feature influence Attrition or Satisfication***. Then each pair will outputs a conclusion.
# 
# 
# 3. ***ANOVA and MANOVA have drawbacks, too.*** While put a set of features under the test, they can not tell specifically which one differs from which one. They give an alert that something wrong happen, that's all.  
# 
#  
# 4. ***The correlation, as this measure is tailored for examining the relationship between numerical variables,*** so I would save it for *** Job Satisfication*** and its relationships.
# 
# The mention of correlation measure and numerical variables reminds to a crucial issue: datatypes. ***Which tests are suitable to apply with different datatypes?*** I will propose a solution right later.
# 
# 
# 
# 
#     

# ## Determining statistic methods according to datatypes
# 
# To select the suitable methods, we need to answer two major questions:
# * What is **datatype** of the **target**?  
# * What is **datatype** of the **feature**?  
# 
# The below diagram will give us the answers.
# 
# ![Workflow.jpg](https://www.upsieutoc.com/images/2020/04/13/Workflow.jpg)
#  

# ## Which features should be analyzed?  
# 
# Having know how to choose statistic methods due to the filter, now I have to pick out a number of features put in it.  
# 
# ***Running tests throughout all features is not necessary.*** If this notebook were a business project, it would be a must absolutely. However, the assignment is not a business but aiming to help to  get familiar with hypothesis testing, so I do not.
# 
# 
# Besides, I want to know if my answers are corrent, so the analyzing features should consist:
#  * ***True positive*** - features really affect Attrition and Satisfaction.
#  * ***True negative*** - features do not. 
#  * ***the Unknown*** - clueless, for my exploration.
# 
# Also, ***None of T-test nor ANOVA*** can handle pairs of ***a categorical target and a non-binary categorical features***, so I intently will not choose this kind of pair.
# 

# ## Which features should be analyzed?  
# 
# ### For the Attrition
# *** True Positives *** are choosen according to [Attrition in an Organization || Why Workers Quit?](https://www.kaggle.com/janiobachmann/attrition-in-an-organization-why-workers-quit), there are:
# * Over Time
# * Monthly Income
# * Age. 
# 
# *** True Negatives *** should be:
# * Distance From Home
# * Total Working Years 
# * Martial Status.
# 
# *** The Unknown *** are:
# * Job Level, Num Companies Worked, Years Since Last Promotion,  
# * Years With Curr Manager, Training Times Last Year, Monthly Rate  
# * Education, Percent Salary Hike.
# 
# ### For Job Satisfication
# The features are ***exactly the same*** without any clue of which True Positive, True Negative, or the Unknown is.

# In[ ]:


features_to_analysis =      ['OverTime',         'MonthlyIncome',         'Age',
                             'DistanceFromHome', 'TotalWorkingYears',     'MaritalStatus',
                             'JobLevel',         'NumCompaniesWorked',    'YearsSinceLastPromotion',
                             'MonthlyRate',      'TrainingTimesLastYear', 'YearsWithCurrManager',
                             'Education',        'PercentSalaryHike']
features_to_analysis.sort()
print(features_to_analysis)


# ### 1. Datatypes of Features

# In[ ]:


# Create table of feature datatypes.
table_datatypes = pd.DataFrame(columns=['Features', 'Datatype'])

# 1st column: Features
table_datatypes['Features'] = features_to_analysis

# 2nd column: Datatypes
table_datatypes['Datatype'] = [df[feature].dtypes for feature in features_to_analysis]

print(table_datatypes)


# The above table classifies:
#  * *** Binary categories:*** OverTime -> needed to encode to 0 - 1 format.
#  * *** Trinary categories: *** MaritalStatus -> ANOVA could be appropriate.
#  * *** Nominal:*** Education and JobLevel -> should be considered numerical.
#  * *** Numerical:*** All the rest. 
#  

# ### 2. Preprocessing features

# In[ ]:


# Binary encoding: MaritalStatus and OverTime:
lb = LabelEncoder()

df['MaritalStatus_encoded'] = lb.fit_transform(df['MaritalStatus']).astype(int)
df['OverTime_encoded'] = lb.fit_transform(df['OverTime']).astype(int)

# Origins replaced by encoded
features_to_analysis = ['MaritalStatus_encoded' if x=='MaritalStatus' else x for x in features_to_analysis]
features_to_analysis = ['OverTime_encoded' if x=='OverTime' else x for x in features_to_analysis]


# ### 3. The Filter suggests methods based on Datatypes
# 
# Let's drop targets and features down the filter, we will find the way.
# 
# #### Attrition: a binary category, so:
#  * All the Features to analysis: using Hypothesis testing, includes MaritalStatus and Overtime which already encoded.
#  
# #### Job Satisfication is orinal, but let deem it numerical for now.
#  * MaritalStatus and OverTime: are partly categorical -> using Hypothesis testing.
#  * All the rest: Correlation.

# ### 4. Which key factors to Attrition?

# ### Hypothesis Testing

# 1. ***Populations***  
# 
#  * Population Yes: All employees who 'Yes' to Attrition.
#  * Population No : All employee who 'No' to Attrition.

# 2. ***Statements***  
# 
# For each feature in the list of undertest features.  
#  * H0: Mean of Population Yes == Mean of Population No  
#  * H1: Mean of Population Yes != Mean of Population No

# 3. ***Calculation***

# In[ ]:


# Split df to Yes-No Attrition
df_Attrition_yes = df[df.Attrition == 'Yes']
df_Attrition_no = df[df.Attrition == 'No']

# Run: One sample Two-sided T-test
t_statistic = []
p_value     = []

for feature in features_to_analysis:
    # t-test
    sample  = df_Attrition_yes[feature]
    popmean = df_Attrition_no[feature].mean() # mean of population
    t_stats, p = stats.ttest_1samp(sample, popmean)
           
    t_statistic.append(t_stats)
    p_value.append(p)    
    
    print('Feature: ', feature)
    print('t-statistic: %4.2f -- p-value: %4.4f \n' %(t_stats, p))


# 4. ***Conclusions***

# In[ ]:


# Create tabel
table = pd.DataFrame()
table['Features'] = features_to_analysis
table['t-statistic'] = t_statistic
table['p-value'] = p_value

# Conclusions
alpha = 0.05
table['Decisions'] = ['Rejected' if x<alpha else 'Failed to reject' for x in table['p-value']]
table['Key factors'] = ['Yes' if x=='Rejected' else 'No' for x in table['Decisions']]

# Drop not-needed
#table = table.drop(['t-statistic', 'p-value'], axis=1)

print(table[['Features', 'Decisions', 'Key factors']].sort_values(by='Key factors', ascending=False))

#print(table.sort_values(by='Decisions'))


# 5. Comments  
# 
#  * ***True Positives*** - Age, MonthlyIncome, and OverTime: all correct.
#  * ***True Negatives*** - DistanceFromHome, TotalWorkingYears, and MaritalStatus: the hypothesis testing considers them key factors.
#  * ***For-curious features*** - Age and Years WithCurr Manager are key factors.

# ### 4. Which key factors to Job Satisfaction?

# Look back to its distribution first.

# ## Hypothesis Testing for MaritalStatus and OverTime

# 1. ***Populations***  
# 
#  * Population: whole employees.
#  * Samples:
#      * Sample 1: employees who is Single   and  Yes-OverTime.
#      * Sample 2: employees who is Married  and  Yes-OverTime.
#      * Sample 3: employees who is Divorced and  Yes-OverTime.
#      * Sample 4: employees who is Single   and  No-OverTime.
#      * Sample 5: employees who is Married  and  No-OverTime.
#      * Sample 6: employees who is Divorced and  No-OverTime.

# In[ ]:


# Preparing samples for ANOVA
population = df[['MaritalStatus', 'OverTime', 'JobSatisfaction']]

anova_samples = {}
i = 1

# Create Samples by conditions
for MS in population['MaritalStatus'].unique():
    for OT in population['OverTime'].unique():
        sample = population['JobSatisfaction'][(df.MaritalStatus==MS) & (df.OverTime==OT)]
        sample.reset_index(drop=True, inplace=True)
        anova_samples[i] = sample
        
        i += 1


# 2. ***Statements***  
# 
#  * H0: Means of JobSatisfaction in 6 samples are equal.  
#  * H1: Existing at least one pair that breaks H0.
#  
#  ***Level of Significant***: 0.05

# 3. ***Calculation***
# 
# Test: One way ANOVA

# In[ ]:


f, p = stats.f_oneway(anova_samples[1],
                      anova_samples[2],
                      anova_samples[3],
                      anova_samples[4],
                      anova_samples[5],
                      anova_samples[6])

print('F-statistic: %4.2f' %(f))
print('p-value    : %4.2f' %(p))


# 4. ***Conclusions***
# 
#  * p-value 0.36 > 0.05 -> ***Failed to reject*** H<sub>0</sub>: all 6 samples mean of Job Satisfaction are equal.  
#  * MarituaStatus and OverTime: ***is not*** key values to JobSatisfaction

# ## Correlation determination for the rest of Features to Analysis

# In[ ]:


## Features to run correlation

# Those be analyzed already.
features_to_analysis.remove('MaritalStatus_encoded')
features_to_analysis.remove('OverTime_encoded')

# Put JobSatisfaction in to determine Correlation matrix latter
features_to_analysis.append('JobSatisfaction')

print(features_to_analysis)


# In[ ]:


corr_matrix = df[features_to_analysis].corr()

# The heatmap
figure = plt.figure(figsize=(16,12))

mask = np.triu(corr_matrix) # Hide the upper part.
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap="YlGnBu", mask=mask)

plt.show()


# ### Unbelievable!!!
# 
# None of Features analysising seems to have any dang correlation to JobSatisfaction!!

# # 6. Final Verdict
# 1. Key factors to Attrition.  
# 
#  * Age, DistanceFromHome, JobLevel,
#  * MaritalStatus, MonthlyIncome, OverTime,
#  * TotalWorkingYears, TrainingTimesLastYear, YearsWithCurrManager.
# 
# Though there is a drawback: we just know they influence the Attrition, but how much?? T-test can not give the answer.
# 
# 2. Key factors to Job Satisfaction.
# 
# My procedure is unable to point out key factors influence the Job Satisfaction, but it indicates those does not:
#  * Age, DistanceFromHome, Education,
#  * JobLevel, MaritalStatus, MonthlyIncome,
#  * MonthlyRate, NumCompaniesWorked, OverTime,
#  * PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear, 
#  * YearsSinceLastPromotion, YearsWithCurrManager

# # 7. My potential mistakes
# 1. ***Lacking of Assumptions checking***. T-test and ANOVA working based on concrete assumptions. If the data are not suitable for them, the testing could be incorrect.
# 
# 
# 2. ***Problems with ordinal data.*** I ran ANOVA with assumption that Job Satisfaction is simply a integer, but it is not. Job Satisfaction is an ordinal, which by far different from a ninteger, so this assumption badly effects on ANOVA at certain level.
