#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read and Explore Data
df = pd.read_csv('../input/Suicides in India 2001-2012.csv', parse_dates=['Year'])
df.head()


# In[ ]:


df.tail()


# In[ ]:


print(df.columns)
print("********************************************")
print(f'DataFrame shape is :  {df.shape}')
print("********************************************")
print(df.info())


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


# Removing unwanted rows from dataset
df_state_total = df[df['State'].str.startswith('Total')].index
df.drop(df_state_total , inplace=True)


# # EDA Of Dataset

# # Which Gender having higest Sueicide number/rate

# In[ ]:


df_gender_Total = df.groupby(['Gender']).sum()
print(df_gender_Total.sort_values(by=['Total'], ascending=False))

print("**********************************************************")

df_gender_percentage = df.groupby(['Gender']).sum()/df['Total'].sum()*100
print(df_gender_percentage.sort_values(by=['Total'], ascending=False))

df_gender_Total.plot(kind='Bar', color ='lightblue', figsize=(9,9) )


# >**Answer:** Male (Almost 64% male commited suicide over the years)  

# # Suicide Data over the years and Find top 5 years in which suicide number is high

# In[ ]:


# Year wise Suicide count

over_year = df.groupby('Year').sum()['Total']
over_year.plot(kind='line', figsize=(15,9))
plt.xlabel("year")
plt.ylabel('Count')
plt.xticks(rotation=45,ha='right')
plt.show()


# > **Answer:** Over the year suicide number is increasing except in 2012 year

# In[ ]:


df_highest_year = df.groupby(['Year'], as_index=False).sum()                  .sort_values(by= 'Total', ascending=False)
print(df_highest_year.head(5))

print("*********************************************")

df_highest_year_percentage = df.groupby(['Year']).sum() / df['Total'].sum()*100
print(df_highest_year_percentage.sort_values(by=['Total'], ascending=False).head(5))


# > **Anser:** In year 2011 suecide number is highest (677159) followed by year 2010, 2012, 2009, 2008 and respective suecide nuber is 672926, 647288, 635429, 625014

# # which age group is having highest suicide number

# In[ ]:


df_age_Total = df.groupby(['Age_group']).sum()
print(df_age_Total.sort_values(by=['Total'], ascending=False))

print("***********************************************************")
df_age_percentage = df.groupby(['Age_group']).sum()/df['Total'].sum()*100
print(df_age_percentage.sort_values(by=['Total'], ascending=False))


# In[ ]:



df_age_Total.plot(kind='Bar', figsize=(9,9))


# > **Answer:** age between 15-29 and 30-44 having highest suicide number respectivly

# # State wise Suicide count

# In[ ]:


df_highest_state = df.groupby(['State'], as_index=False).sum()                   .sort_values(by=['Total'], ascending=[False])
print(df_highest_state.head(5))

print("*********************************************")

df_highest_state_percentage = df.groupby(['State']).sum() / df['Total'].sum()*100
print(df_highest_state_percentage.sort_values(by=['Total'], ascending=False).head(5))

print("**************************************************")

# State wise Suicide count

state = df.groupby('State').sum()['Total']
sort_state = state.sort_values(ascending = False)
state_fig = sort_state.plot('bar', figsize = (13,7), title = 'Suicide count across state\n')
state_fig.set_xlabel('\nState')
state_fig.set_ylabel('Count\n')
plt.xticks(rotation=45,ha='right')


# # Find reason behind the suicide in each age group

# In[ ]:


# function to plot

def plot_fig(df, Title, X_lab):
    p_type = df.groupby('Type').sum()['Total']
    sort_df = p_type.sort_values(ascending = False)

    fig = sort_df.plot('bar', figsize = (17,6), title = Title + '\n', width = 0.75)
    fig.set_xlabel('\n' + X_lab )
    fig.set_ylabel('Count\n')
    plt.xticks(rotation=45,ha='right')
    sns.set_style('whitegrid')


# # Suicide Reason For 0-14 age Suicide

# In[ ]:


# Reason For 0-14 age Suicide
age_till_14= df[df['Age_group']== '0-14']
age_till_14_cause = age_till_14[age_till_14['Type_code'] == 'Causes']
plot_fig(age_till_14_cause, 'Reason For 0-14 age Suicide', 'Cause')
plt.show()


# # Suicide Reason For 15-29 age Suicide

# In[ ]:


# Reason For 15-29 age Suicide
age_15_29 = df[df['Age_group']== '15-29']
age_15_29_cause = age_15_29[age_15_29['Type_code'] == 'Causes']
plot_fig(age_15_29_cause, 'Reason For 15-29 age Suicide', 'Cause')


# # Suicide Reason For 30-44 age

# In[ ]:


age_30_44 = df[df['Age_group']== '30-44']
age_30_44_cause = age_30_44[age_30_44['Type_code'] == 'Causes']
plot_fig(age_30_44_cause, 'Reason For 30-44 age Suicide', 'Cause')


# # Suicide Reason between age 45-59

# In[ ]:


age_45_59 = df[df['Age_group']== '45-59']
age_45_59_cause = age_45_59[age_45_59['Type_code'] == 'Causes']
plot_fig(age_45_59, 'Reason For 45-59 age Suicide', 'Cause')


# # Suicide Reason for age 60+

# In[ ]:


age_over_60 = df[df['Age_group']== '60+']
age_over_60_cause = age_over_60[age_over_60['Type_code'] == 'Causes']
plot_fig(age_over_60_cause, 'Reason For 60+ age Suicide', 'Cause')


# # State wise suicide count (Male/ Female)

# In[ ]:


plt.figure(figsize=(19,6))
df_state_g = df.groupby(['State', 'Gender'], as_index=False).sum().sort_values(by=['Gender'], ascending=False)
df_state_g = df_state_g.sort_values(by=['Total'], ascending=False)
sns.barplot(x='State', y='Total', hue='Gender', data=df_state_g)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()


# # Distribution of suicides according to Educational status and Gender

# In[ ]:


eduDf = df[df['Type_code']=='Education_Status']

plt.figure(figsize=(15,6))
eduDf = eduDf[['Type','Gender','Total']]
edSort = eduDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)
sns.barplot(x='Type',y='Total',hue='Gender',data=edSort)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()


# # Distribution of number of suicided on the basis of Cause

# In[ ]:


causesDf = df[df['Type_code']=='Causes']

plt.figure(figsize=(15,6))
causesDf = causesDf[['Type','Gender','Total']]
causesSort = causesDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)
sns.barplot(x='Type',y='Total',data=causesSort,hue='Gender',palette='viridis')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()


# # Distribution of number of suicides on the basis of Professional Profile

# In[ ]:


profDf = df[df['Type_code']=='Professional_Profile']

plt.figure(figsize=(12,6))
profDf = profDf[['Type','Gender','Total']]
profSort = profDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)
sns.barplot(x='Type',y='Total',data=profSort,hue='Gender',palette='viridis')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()


# # Distribution of number of suicides on the basis of social status

# In[ ]:


socialDf = df[df['Type_code']=='Social_Status']

plt.figure(figsize=(9,6))
socialDf = socialDf[['Type','Gender','Total']]
socialSort = socialDf.groupby(['Type','Gender'],as_index=False).sum().sort_values('Total',ascending=False)
sns.barplot(x='Type',y='Total',data=socialSort,hue='Gender',palette='viridis')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()


# # Top 3 State based on Gender(Female) over the years

# In[ ]:


df_gender_female = df[df['Gender'] =='Female'].groupby(['Year', 'State'], as_index=False).sum()                   .sort_values(by=['Year','Total'], ascending=[True,False])
df_gender_female.groupby('Year').head(3)


# In[ ]:


state_gender_f = df[df['Gender'] =='Female'].groupby('State').sum()['Total']
sort_state_gender_f = state_gender_f.sort_values(ascending = False)
state_gender_f_fig = sort_state_gender_f.plot('bar', figsize = (13,7), title = 'Suicide count across state for Female\n')
state_gender_f_fig.set_xlabel('\nState')
state_gender_f_fig.set_ylabel('Female Count\n')


# # Top 3 State based on Gender(Male) over the years

# In[ ]:


df_gender_male = df[df['Gender'] =='Male'].groupby(['Year', 'State'], as_index=False).sum()                   .sort_values(by=['Year','Total'], ascending=[True,False])
df_gender_male.groupby('Year').head(3)


# In[ ]:


state_gender_m = df[df['Gender'] =='Male'].groupby('State').sum()['Total']
sort_state_gender_m = state_gender_m.sort_values(ascending = False)
state_gender_m_fig = sort_state_gender_m.plot('bar', figsize = (13,7), title = 'Suicide count across state for Male\n')
state_gender_m_fig.set_xlabel('\nState')
state_gender_m_fig.set_ylabel('Male Count\n')


# ## END
