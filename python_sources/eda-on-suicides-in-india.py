#!/usr/bin/env python
# coding: utf-8

# # Analysis of Suicide in India

# [](http://) If you're having suicidal thoughts, please call suicide helpline! 
# 
# Suicide helpline numbers - http://www.suicide.org/international-suicide-hotlines.html
# 

# In[ ]:


# import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read csv file

df = pd.read_csv('../input/Suicides in India 2001-2012.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.duplicated().any()


# In[ ]:


df.sample(10)


# In[ ]:


df.Type_code.value_counts()


# In[ ]:


len(df[df['Total'] == 0])


# In[ ]:


df.State.value_counts()


# In[ ]:


df.Age_group.value_counts()


# ### Data Wrangling

# <ul>
#     <li> Rename State and Type like  'A & N Islands',   'A & N Islands (Ut)' </li>     
#     <li> Drop rows whose total is zero</li>
#     <li> Drop rows where state is "Total %"</li>    
# </ul>    

# In[ ]:


# rename states

df.replace('A & N Islands (Ut)', 'A & N Islands', inplace=True)
df.replace('Chandigarh (Ut)', 'Chandigarh', inplace=True)
df.replace('D & N Haveli (Ut)', 'D & N Haveli', inplace=True)
df.replace('Daman & Diu (Ut)', 'Daman & Diu', inplace=True)
df.replace('Lakshadweep (Ut)', 'Lakshadweep', inplace=True)
df.replace('Delhi (Ut)', 'Delhi', inplace=True)


# In[ ]:


# rename Type

df.replace('Bankruptcy or Sudden change in Economic', 
           'Bankruptcy or Sudden change in Economic Status', inplace=True)
df.replace('By Other means (please specify)', 'By Other means', inplace=True)
df.replace('Not having Children(Barrenness/Impotency',
           'Not having Children (Barrenness/Impotency', inplace=True)


# In[ ]:


# Drop rows where total is zero.

df = df.drop(df[df.Total==0].index)


# In[ ]:


df[df['Total']==0]


# In[ ]:


df = df.drop(df[(df.State == 'Total (Uts)') | (df.State == 'Total (All India)') | 
               (df.State == 'Total (States)')].index)


# ### EDA

# In[ ]:


#Gender Suicide Frequency
gender = df.groupby('Gender').sum()['Total'].plot("bar", figsize=(5,4), title ="Gender wise suicides\n");
gender.set_xlabel('\nGender')
gender.set_ylabel('Count\n')
sns.set_style('whitegrid')
sns.cubehelix_palette(8);


# In[ ]:


# Age Suicide Frequency
df_Age = df[df['Age_group']!='0-100+']

age = df_Age.groupby('Age_group').sum()['Total'].plot("bar",figsize=(8,5), title ="Age wise suicides frequency");
age.set_xlabel('\nAge Group')
age.set_ylabel('Counts\n')
sns.set_style('whitegrid')
sns.set_palette('Set2');


# Middle Age group i.e between 15-44 have the highest number but, It' scary that even kids between age of 0 - 14 also commit suicide.

# In[ ]:


# Suicide rate every year

year = df.groupby('Year').sum()['Total'].plot('line', figsize=(6,6), title = 'Suicide rate per year');
year.set_xlabel('\nYear')
year.set_ylabel('Count\n')
sns.set_style('whitegrid');


# Suicide rate is sky rocketting from 2002 till almost 2010 but, it has droped from then.

# In[ ]:


# State wise Suicide count

state = df.groupby('State').sum()['Total']
sort_state = state.sort_values(ascending = False)
state_fig = sort_state.plot('bar', figsize = (13,7), title = 'Suicide count across state\n', width = 0.75)
state_fig.set_xlabel('\nState')
state_fig.set_ylabel('Count\n')
sns.set_style('whitegrid')
sns.set_palette('Set2');


# Maharashtra, West Bengal and Tamil Nadu have the highest rate of suicide. 

# In[ ]:


# split df ny it's type code

cause = df[df['Type_code'] == 'Causes']
edu_status = df[df['Type_code'] == 'Education_Status']
means_adpt = df[df['Type_code'] == 'Means_adopted']
prof = df[df['Type_code'] == 'Professional_Profile']
soc_status = df[df['Type_code'] == 'Social_Status']


# In[ ]:


# function to plot

def plot_type(df, Title, X_lab):
    p_type = df.groupby('Type').sum()['Total']
    sort_df = p_type.sort_values(ascending = False)

    fig = sort_df.plot('bar', figsize = (10,6), title = Title + '\n', width = 0.75)
    fig.set_xlabel('\n' + X_lab )
    fig.set_ylabel('Count\n')
    sns.set_style('whitegrid')
    sns.set_palette('Set2');   


# In[ ]:


# plot by cause
plot_type(cause, 'Suicide by cause', 'Cause')


# Most of the cause are unknown but Family problems and Prolonged illness is at the top.

# In[ ]:


# plot by education status
plot_type(edu_status, 'Suicide by Education Status', 'Education Status')


# People with primary or no education are high in number.

# In[ ]:


# plot by means adopted
plot_type(means_adpt, 'Suicide by Means Adopted', 'Means Adopted')


# Hanging, Consuming Insecticides is more common but, most of other means are unknown.

# In[ ]:


# suicide by professional profile
plot_type(prof, 'Suicide by Professional Profile', 'Professional Profile')


# Most of the professional profile is unknown but house wife comes in second.

# In[ ]:


# suicide by social Status
plot_type(soc_status, 'Suicide by Social Status', 'Social Status',)


# Family Problems, House Wife and now Married these are at the top when it comes to the cause, professional profile and social status respectively.

# In[ ]:


age_lower = df[df['Age_group']== '0-14']
age_lower_cause = age_lower[age_lower['Type_code'] == 'Causes']
plot_type(age_lower_cause, 'Reason For Kids Suicide', 'Cause')


# Most of cause is unknown but, Failure in Examination is at the top.

# In[ ]:


age_middle = df[df['Age_group']== '15-29']
age_middle_cause = age_middle[age_middle['Type_code'] == 'Causes']
plot_type(age_middle_cause, 'Reason For Kids Suicide', 'Cause')


# Family Problems, Illness and Love affairs are at the top while, most of the reasons are unknown yet.

# ## Conclusion
# *  Age group between 15-44 has the highest number of suicides.
# *  Rate of suicide sky rocketed from 2002-2010 but since 2011 it has been decreasing but, since we have the data till 2012 we can't confim the pattern.
# *  Maharashtra, West Bengal and Tamil Nadu have the highest suicide rate this might also be because of the high population in these states.
# *  Family Problems, Illness, Mental Illness are some of the main reason while most of the reasons are still unknown.
# *  Hanging, Consuming Insecticides is more common but, most of other means are unknown.
# * Most of the professional profile is unknown but house wife comes in second, while Farming comes in at third.
# * Married is at the top when it comes to social status while never married is at second.
# *  The reason for suicide at the age group between 0-14  is because of Failure in Examination.
# *  The reason for suicide at the age group between 15-29 is because of Family Problems.
# 
