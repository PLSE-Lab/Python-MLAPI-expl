#!/usr/bin/env python
# coding: utf-8

# I am kinda new to Kaggle and data analysis in general.

# This kernel was built by learning from a lot of people. So if you see a way to do things better and shorter please drop me a comment.
# * if you like what you see, a thumb up will be very welcome

# The EDA done in this kernel focus more on gender distribution and how it different from one category another.  
# The graph in the kernel was done using Matplotlib and Plotly
# 
# Here is the work flow of the kernel
# 
# 1.	Gender Distribution of the Dataset
# 2.	Gender Distribution of the Purpose 
# 3.	Gender Distribution of Housing
# 4.	Gender Distribution of Age
# 5.	 Gender Distribution  of Job by Skill Set
# 6.	Female Risk Evaluation
# 7.	Male Risk Evaluation
# 

# ### import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
sns.set_style('ticks')
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
print(os.listdir("../input"))


# In[3]:


df = pd.read_csv('../input/german_credit_data.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
# Determine the numbers of NaN value in each column by percentage
(df.isnull().sum()/df.shape[0])*100
df_isnull = pd.DataFrame(round((df.isnull().sum().sort_values(ascending=False)/df.shape[0])*100,1)).reset_index()
df_isnull.columns = ['Columns', '% of Missing Data']
df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
cm = sns.light_palette("skyblue", as_cmap=True)
df_isnull = df_isnull.style.background_gradient(cmap=cm)
print('Only the Saving and Checking Accounts column have missing data')
df_isnull
# 18.3 % of rows in Saving Accounts columns has NaN values
# 39.4% of rows in Checking account columns has NaN values


# ##  1. Gender Distribution of the Dataset

# In[4]:


f, ax = plt.subplots()
ax = df.Sex.value_counts().plot(kind='barh',figsize = (10,5))
kwarg ={'fontsize':15,'color':'black'}
for i in ax.patches:
    ax.text(i.get_width()+10, i.get_y()+0.25,
           str((i.get_width()/len(df.Sex))*100)+'%',**kwarg)
ax.invert_yaxis()  
ax.set_yticklabels(['Male','Female'],**kwarg)
x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the x axis label
ax.set_title('Credit Distribution by Gender',**kwarg)
sns.despine(bottom=True)
plt.show()


# Comment: The graph above shows that Men received credit cards more than Women on a ratio of 2 to 1

# #### Divide the dataset by Gender

# In[5]:


male_data = df[df.Sex == 'male']
female_data=df[df.Sex =='female']


# ## 2. Gender Distribution by Purpose 

# ### 2.1 Male Reason for Credit

# In[10]:


count = male_data.Purpose.value_counts().values
label = male_data.Purpose.value_counts().index
#color= ['darkred','tan','c','lightcoral','skyblue','w','silver','silver']
norm = matplotlib.colors.Normalize(vmin=min(count), vmax=max(count))
colors = [matplotlib.cm.Blues(norm(value)) for value in count]
#Plot the Treemap
f, ax = plt.subplots(figsize=(15,6))
ax = squarify.plot(sizes = count,label=label,color=colors,alpha=0.7,value=count)
plt.axis('off')
ax.set_title('Male: Purpose of the Credit',color='black', size=15)
plt.show()


# ### 2.2 Female Reason for Credit

# In[11]:


count = female_data.Purpose.value_counts().values
label = female_data.Purpose.value_counts().index
#color= ['darkred','tan','c','lightcoral','skyblue','w','silver','silver']
norm = matplotlib.colors.Normalize(vmin=min(count), vmax=max(count))
colors = [matplotlib.cm.Oranges(norm(value)) for value in count]
#Plot the Treemap
f, ax = plt.subplots(figsize=(15,6))
ax = squarify.plot(sizes = count,label=label,color=colors,alpha=0.7,value=count)
plt.axis('off')
ax.set_title('Female: Purpose of the Credit',color='black', size=15)
plt.show()


# ## 3. Gender Distribution by Housing

# In[12]:


f, (ax1,ax2) = plt.subplots(1,2,figsize=(15,7))
ax2.bar(female_data.Housing.value_counts().index,female_data.Housing.value_counts().values,alpha=.8, 
        ec='black',color='#ff7f0e')
ax1.bar(male_data.Housing.value_counts().index,male_data.Housing.value_counts().values,alpha=0.7,
        ec='black', color='#1f77b4')
fsum = female_data.Housing.value_counts().values.sum()
msum = male_data.Housing.value_counts().values.sum()
kwargs= {'fontsize':13, 'color':'black'}
for i in ax1.patches:
    ax1.text(i.get_x()+0.2, i.get_height()+5, str(round(((i.get_height())/msum)*100,1))+'%', **kwarg)
for i in ax2.patches:
    ax2.text(i.get_x()+0.3, i.get_height()+3, str(round(((i.get_height())/fsum)*100,1))+'%', **kwarg)
    
ax1.set_xlabel('Housing',**kwargs)
ax2.set_xlabel('Housing',**kwargs)
ax1.tick_params(length=3, width=1, colors='black',labelsize='x-large')
ax2.tick_params(length=3, width=1, colors='black',labelsize='x-large')
y_axis = ax1.axes.get_yaxis().set_visible(False) # turn off the y axis label
y_axis = ax2.axes.get_yaxis().set_visible(False) # turn off the y axis label
ax1.set_title('Male Housing Distribution',color='black', size=15)
ax2.set_title('Female Housing Distribution',color='black', size=15)
sns.despine(left=True)
plt.show()


# **Comment**: The majority of both men and Women own their house but the percentage of female renting is high than the male 

#  ### 3.1 Female Housing Distribution by skillset

# In[13]:


df_r = df.groupby(['Sex','Job','Housing']).size().reset_index()
df_r.columns = ['Sex','Job','Housing','count']
df_r.Job=df_r.Job.map({0:'Unskilled & Non Resident',1:'Unskilled & Resident',2:'Skilled',3:'Highly Skilled'})
fsh = df_r[df_r['Sex']=='female']
msh = df_r[df_r['Sex']=='male']

trace1 = go.Bar(
    x=fsh[fsh['Housing']=='free']['Job'].values,
    y= fsh[fsh['Housing']=='free']['count'].values,
    opacity=0.8,
    name = 'Free'
)
trace2 = go.Bar(
    x=fsh[fsh['Housing']=='rent']['Job'].values,
    y= fsh[fsh['Housing']=='rent']['count'].values, 
    opacity=0.8,
    name = 'Rent'
)
trace3 = go.Bar(
    x=fsh[fsh['Housing']=='own']['Job'].values,
    y= fsh[fsh['Housing']=='own']['count'].values, 
    opacity=0.8,
    name = 'own'
)
data = [trace1, trace2,trace3]

layout = go.Layout (
    yaxis = dict(
    title = 'Frequency'),
    
    xaxis = dict (
    title = 'Housing'),
    
    title = 'Female: Housing Distribution per Skillset'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)


# ### 3.2 Male Housing distribution per Skillset

# In[14]:


trace1 = go.Bar(
    x=msh[msh['Housing']=='free']['Job'].values,
    y= msh[msh['Housing']=='free']['count'].values,
    opacity=0.8,
    name = 'Free'
)
trace2 = go.Bar(
    x=msh[msh['Housing']=='rent']['Job'].values,
    y= msh[msh['Housing']=='rent']['count'].values, 
    opacity=0.8,
    name = 'Rent'
)
trace3 = go.Bar(
    x=msh[msh['Housing']=='own']['Job'].values,
    y= msh[msh['Housing']=='own']['count'].values, 
    opacity=0.8,
    name = 'own'
)
data = [trace1, trace2,trace3]

layout = go.Layout (
    yaxis = dict(
    title = 'Frequency'),
    
    xaxis = dict (
    title = 'Housing'),
    
    title = 'Male: Housing Distribution per Skillset'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)


# #### Categozising the Age distribution by Age Generation

# In[15]:


range_list = (18,35,45,65,80)
cat = ['Millennials','Gen_X','Baby_Boomer','Silent_Gen']
df['Age_Gen'] = pd.cut(df.Age,range_list,labels=cat)

### Categoze the Duration by Years

range_list = (1,12,24,36,48,60,72)
cat = ['One','Two','Three','Four','Five','Six']
df['Duration_yrs']= pd.cut(df.Duration,range_list,labels=cat)

## 4. Gender Distribution by Age

f, ax = plt.subplots(figsize = (15,5))
ax = sns.kdeplot(male_data.Age,shade=True)
ax = sns.kdeplot(female_data.Age,shade=True)
ax.legend(['Male','Female'],fontsize=12)
ax.set_xlabel('Age',fontsize=14,color='black')

ax.set_title('Age Distribution by Gender',color='black',fontsize=14)
y_axis = ax.axes.get_yaxis().set_visible(False) # turn off the y axis label
sns.despine(left=True)


# The graph above shows that the female coustomer tend to be younger than the male. This graph also shows that for both male and female, the majority of the customers are Millennials (less than 35 years old)

# ## 5. Gender Distribution  of Job by Skill Set

# In[16]:


# DataFrame by Job percentage
female_job_pct = pd.DataFrame(round(female_data.Job.value_counts()/len(female_data.Job)*100,1)).reset_index()
female_job_pct.columns = ['Job','female_pct']
# Male DataFrame for the Job percentage
male_job_pct = pd.DataFrame(round(male_data.Job.value_counts()/len(male_data.Job)*100,1)).reset_index()
male_job_pct.columns = ['Job','male_pct']
# Joining the DataFrame together
job_pct = pd.merge(female_job_pct,male_job_pct,how='left')
job_pct.Job = job_pct['Job'].map({0:'Unskilled & Non Resident',1:'Unskilled & Resident',2:'Skilled',3:'Highly Skilled'})

ax = job_pct[['male_pct','female_pct']].plot(kind='bar',figsize=(15,5),fontsize=13,width=0.6)
ax.set_xticklabels(job_pct.Job.tolist(),rotation=0)
ax.set_title('Percentage Distribution of Job per Gender',fontsize=18,color='black')
for i in ax.patches:
    ax.text(i.get_x()+0.05, i.get_height()+0.8, str((i.get_height()))+'%', fontsize=13)
y_axis = ax.axes.get_yaxis().set_visible(False) # turn off the y axis label
plt.legend(loc=('upper right'),fontsize=13,title ='Gender Class',ncol=2)
sns.despine(left=True)


# ## 6.  Female Risk Evaluation

# In[17]:


# Create a dataframe of only Sex, Duration_yrs and risk
df_duration_yrs = df.groupby(['Sex','Duration_yrs','Risk']).size().reset_index()
df_duration_yrs.columns = ['Sex','Duration_yrs','Risk','count']
# Split the New dataset df_duration_yrs by Gender
ms = df_duration_yrs[df_duration_yrs['Sex']=='male']
fs = df_duration_yrs[df_duration_yrs['Sex']=='female']

### 6.1 By Skill Set

female_risk = female_data.groupby(['Risk','Job']).size().reset_index()
female_risk.columns = ['Risk','Job','count']
female_risk.Job=female_risk.Job.map({0:'Unskilled & Non Resident',1:'Unskilled & Resident',2:'Skilled',3:'Highly Skilled'})
f, ax=plt.subplots(figsize=(14,5))
ax = sns.barplot(x='Job',y= 'count',data=female_risk,hue='Risk',palette=['darkred', 'darkgreen'],alpha=0.7,
                edgecolor="black")
kwarg ={'fontsize':15, 'color':'black'}
ax.set_title('Female: Risk Distribution per Skillset',**kwarg)
ax.tick_params(length=3, width=1, colors='black',labelsize='x-large')
ax.set_xlabel('Job',**kwarg)
ax.set_ylabel('Frequency',**kwarg)
for i in ax.patches:
    ax.text(i.get_x()+0.1, i.get_height()+2.2, str((i.get_height())), fontsize=13,color='black')
plt.legend(loc=(0.8,0.8),fontsize=13,title ='female Risk Class',ncol=2)
sns.despine()


# ### 6.2 By Duration in Years

# In[18]:


trace1 = go.Bar(
    x=fs[fs['Risk']=='bad']['Duration_yrs'].values,
    y= fs[fs['Risk']=='bad']['count'].values,
    text =  fs[fs['Risk']=='bad']['count'].values,
    textposition='auto',
    textfont=dict(
        family='sans serif',
        size=18,
        color='#FFFAFA'),
    marker=dict(
    color='rgb(139,0,0)'),
    opacity=0.8,
    name = 'Bad Risk'
)
trace2 = go.Bar(
    x=fs[fs['Risk']=='good']['Duration_yrs'].values,
    y= fs[fs['Risk']=='good']['count'].values,
    text =  fs[fs['Risk']=='good']['count'].values,
    textposition='auto',
    textfont=dict(
        family='sans serif',
        size=18,
        color='#FFFAFA'),
    marker=dict(
    color='rgb(0,100,0)'),
    opacity=0.8,
    name = 'Good Risk'
)
data = [trace1, trace2]
layout = go.Layout (
    yaxis = dict(
    title = 'Frequency'),
    
    xaxis = dict (
    title = 'Duration in Years'),
    
    title = 'Female: Risk Distribution per Duration'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)


# ## 7.  Male Risk Evaluation

# ### 7-1 By Skill Set

# In[19]:


male_risk = male_data.groupby(['Risk','Job']).size().reset_index()
male_risk.columns = ['Risk','Job','count']
male_risk.Job= male_risk.Job.map({0:'Unskilled & Non Resident',1:'Unskilled & Resident',2:'Skilled',3:'Highly Skilled'})

f, ax=plt.subplots(figsize=(14,5))
ax = sns.barplot(x='Job',y= 'count',data= male_risk,hue='Risk',palette=['darkred', 'darkgreen'],alpha=0.7,
                edgecolor="black")
kwarg ={'fontsize':15, 'color':'black'}
ax.set_title('Male: Risk Distribution per Skillset',**kwarg)
ax.tick_params(length=3, width=1, colors='black',labelsize='x-large')
ax.set_xlabel('Job',**kwarg)
ax.set_ylabel('Frequency',**kwarg)
for i in ax.patches:
    ax.text(i.get_x()+0.1, i.get_height()+2.2, str((i.get_height())), fontsize=13,color='black')
#y_axis = ax.axes.get_yaxis().set_visible(False) # turn off the y axis label
plt.legend(loc=(0.8,0.8),fontsize=13,title ='Male Risk Class',ncol=2)
sns.despine()


#  ### 7-2 By Duration in Years

# In[20]:


trace1 = go.Bar(
    x=ms[ms['Risk']=='bad']['Duration_yrs'].values,
    y= ms[ms['Risk']=='bad']['count'].values,
    text =  ms[ms['Risk']=='bad']['count'].values,
    textposition='auto',
    textfont=dict(
        family='sans serif',
        size=18,
        color='#FFFAFA'),
    marker=dict(
    color='rgb(139,0,0)'),
    opacity=0.8,
    name = 'Bad Risk'
)
trace2 = go.Bar(
    x=ms[ms['Risk']=='good']['Duration_yrs'].values,
    y= ms[ms['Risk']=='good']['count'].values,
    text =  ms[ms['Risk']=='good']['count'].values,
    textposition='auto',
    textfont=dict(
        family='sans serif',
        size=18,
        color='#FFFAFA'),
    marker=dict(
    color='rgb(0,100,0)'),
    opacity=0.8,
    name = 'Good Risk'
)
data = [trace1, trace2]

layout = go.Layout (
    yaxis = dict(
    title = 'Frequency'),
    
    xaxis = dict (
    title = 'Duration in Years'),
    
    title = 'Male: Risk Distribution per Duration'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)


# Will continue with machine learning classification soon

# 

# In[ ]:





# In[ ]:




