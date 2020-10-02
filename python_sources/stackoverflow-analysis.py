#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_cell_magic('html', '', '<img src = "https://cdn.sstatic.net/insights/Img/Survey/2018/TwitterCard.png?v=5279b4381c14" , width = 1400>')


# In[10]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style("darkgrid")


# In[16]:


df = pd.read_csv('../input/survey_results_public.csv')
df.head()


# `** Stackoverflow userbase **

# In[12]:


df_Country = df['Country'].value_counts(False , True , False)[:10]
df_Country.plot.bar()
plt.xlabel("COUNTRY")
plt.ylabel("USERS")
plt.title("COUNTRIES WITH MOST USERS")


# ** OPEN SOURCE CONTRIBUTION FROM  COUNTRIES **

# In[13]:


country_opensource = df[['OpenSource' , 'Country']]
country_Contribution = country_opensource[country_opensource['OpenSource'] == 'Yes' ].groupby(['Country']).count().sort_values(by = 'OpenSource' , ascending = False)[:10].reset_index()
plt.figure(figsize=(15,8)) 
sb.barplot(x= 'Country' , y='OpenSource' , data = country_Contribution , palette = 'cool' )
plt.xlabel(' COUNTRIES ')
plt.ylabel(' OPENSOURCE CONTRIBUTION COUNT')
plt.title(' TOP TEN COUNTRIES CONTRIBUTING TO OPENSOURCE')


# **GENDER WISE USER DISTRIBUTION**

# SOME NOISY CATEGORIES ARE DROPPED
# 

# In[74]:


import re
df1=df[['Gender']]
df1=df1[df1['Gender'].notna()]

df1=df1[df['Gender'].str.len()<12]
df1['Gender']=df1["Gender"].str.replace('Female;Male','Transgender')
df1['Gender'].value_counts().plot(kind='pie')


# **THE STUDENTS**

# In[78]:


df['Student'].value_counts().plot.bar()


# **PEOPLE WHO CONTRIBUTE AS HOBBY**

# In[80]:


df['Hobby'].value_counts().plot.bar()


# INTERESTING PLOT THAT IDENTIFIES THAT MOST OF THE USERS ARE ENGAGED AS HOBBY.

# **CAREER SATISFACTION OF THE USERS**
# 
# 

# In[82]:


df['CareerSatisfaction'].value_counts().plot.bar()


# A SKEWED DISTRIBUTION THAT TELLS THAT PEOPLE WHO ARE SATISFIED WITH CAREER ARE THE MAJORITY USERS

# **JOB SATISFACTION**

# In[84]:


df['JobSatisfaction'].value_counts().plot.bar()


# ALMOST IDENTICAL TO CAREER SATISFACTION

# **CODING EXPRIENCE BASED INSIHGTS**

# In[110]:


data=df[df['YearsCoding'].notna()]
data=data['YearsCoding']
data.value_counts().plot.bar()
#df['YearsCodingProf'].value_counts().plot.bar()
labels=['3-5','6-8','9-11','0-2','12-14','15-17','18-20','30+','21-23','24-26','27-29']
explode=[0.3,0.2,0.0,0.5,0.0,0.0,0.0,0.2,0.3,0.0,0.5]


# People with coding experience between 3-11 years tends to be more pobable to contribute in the cummunity .
# Where as people with experience above 21 years tend to be less probable in engaging .

# In[111]:


plt.pie(data.value_counts(),autopct='%1.1f%%',labels=labels,explode=explode)


# distibution for users by coding experience

# **INSIGHTS BASED ON USERS EMPLOYMENT STATUS**

# In[119]:


emp=df['Employment'].value_counts()
emp.plot.bar()
labels=['Full_time','self_employed','looking_for','part_time','unemployed_notLooking','Retired']
explode=[0.1,0.0,0.5,0.3,0.1,0.2]


# In[120]:


plt.pie(emp,autopct='%1.1f%%',labels=labels,explode=explode)


# Surprisingly users who are employed full time contribute more to community than people self-employed , people looking for jobs , retired professionals , part-time employes.
# This insight tells us that 74% of the users are full time employed and prior insights tells us that majority of the users are satisfied with their jobs as well as carrer.

# **HOW MANY HOURS DO USERS SPEND ON COMPUTER?**

# In[126]:


comp=df['HoursComputer'].value_counts()
comp.plot.bar()
labels=['9-12','5-8','12>','1-4','<1']
explode=[0.2,0.3,0.1,0.3,0.3]


# In[127]:


plt.pie(comp,autopct='%1.1f%%',labels=labels,explode=explode)


# 52.7% of the users spent 9-12 hours in front of computer which could be explained them being full-time employed professionals as around 70% users are full time employed.

# 

# **>THATS IT FOR NOW ! PLEASE UPVOTE IF YOU LIKE IT.**
