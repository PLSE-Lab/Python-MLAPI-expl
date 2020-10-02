#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


sns.set(style='whitegrid')


# My task:
# * 1) check basic info about dataset: len, missing data etc.
# * 2) calculate average_score each student
# * 3) see how students are distributed by race, parental education, test preparation course, lunch
# * 4) see how average score is distributed by race, parental education, test preparation course, lunch, gender
# * 5) check whether there are the relationship to the average score with race, parental education, test preparation course, lunch, gender.
# 

# In[ ]:


data.columns


# Check the columns name. we see that some columns have spaces in the names. So we will rename the columns

# In[ ]:


data.rename(columns={
                    'race/ethnicity':'race',
                    'parental level of education': 'parent_education',
                    'test preparation course': 'pretest'
                    },inplace=True)
data.columns


# In[ ]:


data.info()


# we see that number of rows is 1000 and number of columns 8. 3 columns have integer type of data, and other are object. and we haven't null value in each column and row

# In[ ]:


data.isna().sum()


# check missing data. Our data haven't missing data

# In[ ]:


data['avg_score'] = data.loc[:,['math score','reading score','writing score']].apply(np.mean, axis=1).round(4)

create column with average score . 
# In[ ]:


sns.distplot(data['avg_score'])


# In[ ]:


_ = stats.probplot(data['avg_score'], plot=sns.mpl.pyplot)


# in these two plots we see that our avg_score values distributed almost normally(has deviations on the tails on both sides)

# In[ ]:


p = sns.countplot(x='gender', data=data, palette='muted')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# see how students distributed by gender. we see not big difference in the distribution in favor of female

# In[ ]:


sns.boxplot(x='gender',y='avg_score',data=data)


# In[ ]:


sns.distplot(data[data['gender']=='female']['avg_score'])


# we can say that distribution is normally

# In[ ]:


sns.distplot(data[data['gender']=='male']['avg_score'])


# In[ ]:


def prepare_anova_data(column_name):
    list_names_factor_type = list(data[column_name].unique())
    n_sample = data[column_name].value_counts().min()
    groups = [data[data[column_name]==key].sample(n_sample) for key in list_names_factor_type]
    pre_data = pd.concat(groups)
    return pre_data


# create function that make the same size of samples

# In[ ]:


results = ols('avg_score ~ C(gender)', data=prepare_anova_data('gender')).fit()
results.summary()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table


# we can say that can exist some relationship a gender with an average score

# In[ ]:


p = sns.countplot(x='race', data=data, order=data['race'].value_counts().index, palette='muted')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# Students distributed by race. The smallest group in size is a group A. The biggest group in size is a group C.

# In[ ]:


p = sns.boxplot(x='race',y='avg_score', data=data, palette='muted')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# distribution average score group by race. 

# In[ ]:


sns.distplot(data[data['race']=='group A']['avg_score'],bins=20)


# In[ ]:


sns.distplot(data[data['race']=='group B']['avg_score'],bins=20)


# In[ ]:


sns.distplot(data[data['race']=='group C']['avg_score'],bins=20)


# In[ ]:


sns.distplot(data[data['race']=='group D']['avg_score'],bins=20)


# In[ ]:


sns.distplot(data[data['race']=='group E']['avg_score'],bins=15)


# distribution of average score by each race

# In[ ]:


results = ols('avg_score ~ C(race)', data=prepare_anova_data('race')).fit()
results.summary()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table


# the relationship between race and average score can exist

# In[ ]:


order=list(data['parent_education'].value_counts().index)
order


# In[ ]:


p = sns.countplot(x='parent_education', data=data, order=order, palette='muted')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# students ditribution by parental education. The smallest group in size is a "master's degree". The biggest group in size is a "some college".

# In[ ]:


sns.distplot(data[data['parent_education']==order[0]]['avg_score'])


# In[ ]:


sns.distplot(data[data['parent_education']==order[1]]['avg_score'])


# In[ ]:


sns.distplot(data[data['parent_education']==order[2]]['avg_score'])


# In[ ]:


sns.distplot(data[data['parent_education']==order[3]]['avg_score'])


# In[ ]:


sns.distplot(data[data['parent_education']==order[4]]['avg_score'],bins=15)


# In[ ]:


sns.distplot(data[data['parent_education']==order[5]]['avg_score'],bins=12)


# In[ ]:


p = sns.boxplot(data=data, x='parent_education', y='avg_score')
_ = plt.setp(p.get_xticklabels(), rotation=90)

distribution average score by parental education.
# In[ ]:


p = sns.countplot(x='parent_education',hue='race', data=data, order=data['parent_education'].value_counts().index, palette='muted')
plt.title('sorted by parent_education')
_ = plt.setp(p.get_xticklabels(), rotation=90)


# distribution race by parent_education

# In[ ]:


p = sns.countplot(x='lunch', data=data, palette='muted')


# we see that students who has a free/reduced lunch are much less than who has standard

# In[ ]:


sns.boxplot(x='lunch',y='avg_score',data=data)


# In[ ]:


sns.distplot(data[data['lunch']=='standard']['avg_score'])


# In[ ]:


sns.distplot(data[data['lunch']=='free/reduced']['avg_score'])


# we can say that average score distributed by lunch normally

# In[ ]:


results = ols('avg_score ~ lunch', data=prepare_anova_data('lunch')).fit()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table


# we can say that exist some relationship an average score with lunch

# In[ ]:


p = sns.countplot(x='pretest', data=data,palette='muted')
_ = plt.setp(p.get_xticklabels(),rotation=0)


# we see that students who completed a test are much less than who didn't complete

# In[ ]:


sns.boxplot(x=data['pretest'],y=data['avg_score'])


# In[ ]:


sns.boxplot(x=data['race'],y=data['avg_score'],hue=data['pretest'])


# we see how distibuted avg_score by pretest and race. We can say there is a tendency that students who wrote the test have a higher average score regardless of race

# In[ ]:


p = sns.lineplot(x=data['parent_education'],y=data['avg_score'],hue=data['pretest'])
_ = plt.setp(p.get_xticklabels(),rotation=90)


# we see how distibuted avg_score by pretest and parental education. We can say there is a tendency that students who wrote the test have a higher average score regardless of race

# In[ ]:


results = ols('avg_score ~ parent_education', data=prepare_anova_data('parent_education')).fit()
results.summary()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table.round(4)


# we can say that can exist some relationship a parental education with an average score

# In[ ]:


results = ols('avg_score ~ pretest', data=prepare_anova_data('pretest')).fit()
results.summary()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table


# we can say that can exist some relationship a pretest with an average score

# In[ ]:


model = ols('avg_score ~ C(race)*C(lunch)*C(gender)', data).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table.round(4)


# we can't say that exist influencs between average score and interection of race,lunch and gender
# 
