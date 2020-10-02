#!/usr/bin/env python
# coding: utf-8

# ### What skills move up our compensation? 

# Hi, in this notebook I will try to explore what skills make the difference in our compensation, using the kaggle survey of 2018. All of us have questions of what do we learn next for our carrer growth, and what frameworks, languages and skills make the difference against others,  and I wil try helping answering this questions on this notebook.

# In[ ]:


# importing packages

import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
pd.options.display.max_columns = 9999
from matplotlib import rc

import warnings
warnings.simplefilter("ignore")


get_ipython().run_line_magic('matplotlib', 'inline')

# opening data

df = pd.read_csv(r'../input/multipleChoiceResponses.csv',header=1)

cols_keep = ['Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
             'What is your age (# years)?',
             'What is your current yearly compensation (approximate $USD)?',
             'In which country do you currently reside?',
             'Does your current employer incorporate machine learning methods into their business?',
             'What is your gender? - Selected Choice',
             'How long have you been writing code to analyze data?',
             'How many years of experience do you have in your current role?']

cols_programming = [col for col in df.columns if 'What programming languages do you use on a regular basis' in col]
cols_programming.remove('What programming languages do you use on a regular basis? (Select all that apply) - Other - Text')

cols_keep.extend(cols_programming)

dict_change = {'What is your age (# years)?':'age',
               'What is your current yearly compensation (approximate $USD)?':'compensation',
               'In which country do you currently reside?':'country',
               'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'role',
               'Does your current employer incorporate machine learning methods into their business?':'use_ml',
               'What is your gender? - Selected Choice':'gender',
               'How long have you been writing code to analyze data?':'time_coding',
               'How many years of experience do you have in your current role?':'time_work'}

dict_programming = {}
for i in cols_programming:
    dict_programming[i] = i.split('-')[2]
    
    
dict_change.update(dict_programming)
                   
df2 = df[cols_keep]
                   
df2 = df2.rename(columns=dict_change)

for i in dict_programming.values():
    df2[i] = df2[i].fillna(0)
    df2[i] = np.where(df2[i] == 0, 0 ,1)
    df2[i] = df2[i].astype(int)
    
df2['Number_languages'] = df2[list(dict_programming.values())].sum(1)
                   
#df2['time_work'] = df2['time_work'].fillna('0-1')
df2['time_work'] = df2['time_work'].replace('30 +','25-30')
df2['time_work'] = df2['time_work'].replace('25-30','10-15')
df2['time_work'] = df2['time_work'].replace('20-25','10-15')
df2['time_work'] = df2['time_work'].replace('15-20','10-15')

df2['time_work'] = df2['time_work'].str.split('-', expand = True)
df2['time_work'] = df2['time_work'].astype(float)
                   
df2['age'] = df2['age'].replace('80+','70-79')
df2['age'] = df2['age'].str.split('-', expand = True)
df2['age'] = df2['age'].astype(int)


# For the analysis, I will only study the cases where the compensation is declared, since understanding the compensation is the main goal of this notebook

# In[ ]:


df2 = df2[df2['compensation'].notnull()]
df2 = df2[df2['compensation'] != 'I do not wish to disclose my approximate yearly compensation']


# In[ ]:


df2['compensation'] =  df2['compensation'].str.split('-').apply(pd.Series)[0]
df2['compensation'] = df2['compensation'].replace('500,000+',500)
df2['compensation'] = df2['compensation'].astype(int)


# The first thing I do is get a distribuition of the roles. Here, we see that Data Scientist and Student are the most popular ones. Right now, I will focus on the Data Scientist role, but later on I will come back to other roles.

# In[ ]:


ax, fig = plt.subplots(1,1, figsize= (16,6))
ax = sns.countplot(y="role",
                   data=df2,
                   order = df2.groupby(['role'])['country'].count().sort_values(ascending=False).index)


# In[ ]:


df3 = df2[df2['role'] == 'Data Scientist']


# First, let's see how the compensation changes with the time working, since it must be one of the most correlated features.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience')
ax[1].set_title('Distribution of Years Experience')

sns.countplot(y="time_work",
            orient='h',
            data=df3,
            edgecolor='black',
            linewidth=1.5,
            ax=ax[1])


# We see that we have a lot of kagglers with not much experience,  and that more years of experience, the compensation tends to grow. 
# Now, let's see if there a difference in  compensation between this groups, if we take in consideration the languages that the kagglers know.

# Following up is a plot of the languages know by Data Scientists. Not surprisingly, Python, SQL and R are the most popular ones. 

# In[ ]:


df4 = df3.melt(value_vars=dict_programming.values())
ax, fig = plt.subplots(1,1, figsize= (16,6))
ax = sns.barplot(x="value",y='variable',
                 order = df4.groupby(['variable'])['value'].sum().sort_values(ascending=False).index,
                 data=df4)#,


# Now, let's see the total distribution that Data Scientists know.

# In[ ]:


fig, ax = plt.subplots(1,1, figsize= (18,6))

ax.set_title('Number of languages know by Data Scientists')

sns.countplot(x="Number_languages",
              orient='h',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax)


# We see that most Data Scientist know between 1 and 4 languages. Now let's remove the outliers and cross this information with compensation, to see if makes a difference knowing a lot of languages.

# In[ ]:


df3['Number_languages'] = np.where(df3['Number_languages'] > 6, 6, df3['Number_languages'])


# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,8))
sns.boxplot(y="Number_languages",x='compensation',
            orient='h',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience')
ax[1].set_title('Distribution of Years Experience')

sns.countplot(y="Number_languages",
            orient='h',
            data=df3,
            edgecolor='black',
            linewidth=1.5,
            ax=ax[1])


# Here, it seens that programming in a big number of languages does not make that much of a difference. So, my guess is that is not quantity of languages that matter, but knowing the most important ones. Now, let's analyze how our Data Scientist top 3 (Python, SQL and R) change the compensation. In this part, I will cross the information with years of experience, to diminish our bias,  

# In[ ]:


df3['Python & R'] = np.where((df3[' R'] == 1) & (df3[' Python'] == 1), 1,0)
df3['Python & R & SQL'] = np.where((df3[' R'] == 1) & (df3[' Python'] == 1) & (df3[' SQL'] == 1), 1, 0)
df3['Python & SQL'] = np.where((df3[' SQL'] == 1) & (df3[' Python'] == 1), 1, 0)
df3['R & SQL'] = np.where((df3[' R'] == 1) & (df3[' Python'] == 1), 1, 0)

df3['Top_languages'] = np.where((df3[' R'] == 1) | (df3[' Python'] == 1) | (df3[' SQL'] == 1), 1, 0)


# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='Top_languages',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience')
ax[1].set_title('Distribution of Years Experience')

sns.countplot(y="time_work",
              orient='h',
              hue='Top_languages',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# With this analysys, we get a problem! Almost all of the Data Scientist who answer the survey know at least one of the 3 top languages!  So, let's try to analyze them separately.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue=' Python',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - Python')
ax[1].set_title('Distribution of Years Experience - Python')

sns.countplot(y="time_work",
              orient='h',
              hue=' Python',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue=' R',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - R')
ax[1].set_title('Distribution of Years Experience - R')

sns.countplot(y="time_work",
              orient='h',
              hue=' R',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue=' SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - SQL')
ax[1].set_title('Distribution of Years Experience - SQL')

sns.countplot(y="time_work",
              orient='h',
              hue=' SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# Well, the difference between groups that we're trying to find it's not been find yet. Maybe because we are analyzing these languages separated. What if we group them? Usualy, people say it's important to learn either R or Python, and SQL to help on extraction. Let's see if this combiantion move ups compensation.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='Python & SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - Python & SQL')
ax[1].set_title('Distribution of Years Experience - Python & SQL')

sns.countplot(y="time_work",
              orient='h',
              hue='Python & SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='R & SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - R & SQL')
ax[1].set_title('Distribution of Years Experience - R & SQL')

sns.countplot(y="time_work",
              orient='h',
              hue='R & SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='Python & R & SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - Python & R & SQL')
ax[1].set_title('Distribution of Years Experience - Python & R & SQL')

sns.countplot(y="time_work",
              orient='h',
              hue='Python & R & SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])


# Well, it seems that knowing SQL + Python + R  help in the early years our careers, making the compensation of these groups slighly higher. So, if you are starting, maybe taking the time to learn SQL (if you already know python or R) can make the difference. However, after 5 years of work experience this difference is almost not existent. Maybe after this time programming skills are not the main focus? Let's try to find out, and also see if there's another skills that could help new Data Scientist growing their salaries. 

# # Work in progress, more to come...

# In[ ]:




