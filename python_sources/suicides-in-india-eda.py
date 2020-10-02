#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter('ignore')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[2]:


suicides = pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')


# In[3]:


suicides.head()


# In[4]:


suicides = suicides[suicides['Year']==2012]


# In[5]:


for col in suicides.columns[:-1]:
    print(col,'-',suicides[col].nunique(),'-',suicides[col].unique())
    print('\n')


# In[6]:


suicides[suicides['Type_code']=='Means_adopted'].head()


# In[7]:


suicides[suicides['Type_code']=='Means_adopted'].Type.nunique()


# In[8]:


suicides[suicides['Type_code']=='Means_adopted'].groupby('Type').sum()


# In[9]:


means_adopted = suicides[suicides['Type_code']=='Means_adopted'].groupby('Type').sum().sort_values('Total', ascending=False)
plt.figure(figsize=(10,5))
ax = sns.barplot(x=means_adopted.index,y=means_adopted['Total'],data=means_adopted)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()


# I'll not be analysing further into the means of suicide as they hold less value. What is more important is to analyse the causes.

# In[10]:


suicides[suicides['Type_code']=='Education_Status'].head()


# In[11]:


educational_status = suicides[suicides['Type_code']=='Education_Status']


# In[12]:


for col in educational_status.columns[3:-1]:
    print(col,'-',educational_status[col].nunique(),'-',educational_status[col].unique())
    print('\n')


# In[13]:


educational_status_grouped = suicides[suicides['Type_code']=='Education_Status'].groupby('Type').sum().sort_values('Total', ascending=False)
plt.figure(figsize=(10,5))
ax = sns.barplot(x=educational_status_grouped.index,y=educational_status_grouped['Total'],data=educational_status_grouped)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()


# The thing that i can infer is that suicide rates are much less in students who had graduated and beyond. There is much more suicide rates among students who were still in school. Can we infer that in India the pressure a students faces is much more in school than in college. Maybe the hype around board exams and college entrance exams might be the reason for it!

# Lets break down the above chart gender wise - 

# In[14]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x='Type',y='Total',hue='Gender',data=educational_status.sort_values('Total',ascending='False'),ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()


# Among students, suicides are much higher among boys as compared to girls. Maybe its due to the patriarchal nature of the Indian society that more is expected from a boy because he is expected to head the family and is supposed to be the sole and main financial contributor.

# In[15]:


educational_status.head()


# In[16]:


educational_status['Type'].unique()


# In[17]:


educational_status_statewise = educational_status.groupby('State').sum().sort_values('Total')
educational_status_statewise = educational_status_statewise.drop(['Total (States)','Total (All India)','Total (Uts)'])


# In[18]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x=educational_status_statewise.index,y=educational_status_statewise['Total'],data=educational_status_statewise)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title('Absolute number of student suicides in Indian States')


# The absolute suicide numbers might not be the correct measure to measure a state's ranking as they vary quite a lot in terms of population. Hence use the population data and then try to rank states.

# In[19]:


#print(os.listdir("../input/all-census-data/"))
population_data = pd.read_csv('../input/all-census-data/elementary_2015_16.csv')


# In[20]:


population_data.head()


# In[21]:


population_data = population_data[['STATE NAME','TOTAL POULATION']]


# In[22]:


population_data = population_data.groupby('STATE NAME').sum()


# In[23]:


population_data.columns = ['population']


# In[24]:


#population data is of the year when the state andhra pradesh had been divided into 2 - one itself and the other named telangana
#education data is of the time when it was one whole
#hence the population data had to be manipulated for these 2 states
population_data.at['ANDHRA PRADESH', 'population'] = population_data.ix['ANDHRA PRADESH']['population'] + population_data.ix['TELANGANA']['population']


# In[25]:


population_data.drop('TELANGANA',axis=0,inplace=True)


# In[26]:


educational_status_statewise = educational_status_statewise.sort_values('State')
population_data.index = population_data.index.str.lower()
educational_status_statewise.index = educational_status_statewise.index.str.lower()


# In[27]:


educational_status_statewise.rename(index={'a & n islands': 'andaman and nicobar islands',
                                          'd & n haveli':'dadra and nagar haveli',
                                          'delhi (ut)':'delhi',
                                          'daman & diu':'daman and diu',
                                          'jammu & kashmir':'jammu and kashmir',
                                          'puducherry':'pondicherry'},inplace=True)


# In[28]:


new_data = pd.concat([population_data,educational_status_statewise],axis=1)


# In[29]:


new_data.head()


# In[30]:


new_data.columns


# In[31]:


new_data['%suicide'] = (new_data['Total']/new_data['population'])*100


# In[32]:


new_data = new_data.sort_values('%suicide',ascending=False)
new_data.head()


# In[33]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x=new_data.index,y=new_data['%suicide'],data=new_data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title('Rate of student suicides in Indian States')


# In[34]:


professional_profile = suicides[suicides['Type_code']=='Professional_Profile']


# In[35]:


professional_profile.head()


# In[36]:


for col in professional_profile.columns[:-1]:
    print(col,'-',professional_profile[col].nunique(),'-',professional_profile[col].unique())


# In[37]:


professional_profile = professional_profile[professional_profile['Type']!='Others (Please Specify)']
by_profession = professional_profile.groupby('Type').sum()
by_profession = by_profession.sort_values('Total',ascending=False)
plt.figure(figsize=(10,5))
ax = sns.barplot(x=by_profession.index,y=by_profession['Total'],data=by_profession)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title('Suicides categorised by different profession')


# Highest suicide rates among House Wives is a worrying sign. An article by BBC addresses this problem too - https://www.bbc.com/news/world-asia-india-35994601. In India, there is much more public debates around suicides by farmers than among housewives. The Agriculture sector in India is highly volatile because more than 60% of the land under cultivation is dependent on rainfall as its primary source of irrigation. Due to global warming the rainfall patterns are changing which are putting our farmers in deep distress leading to high suicide rates among them.

# Lets just analyse if rainfall patterns influence suicides among farmers in India - 

# In[38]:


#print((os.listdir("../input/rainfall-in-india/")))
rainfall = pd.read_csv('../input/rainfall-in-india/district wise rainfall normal.csv')
rainfall.head()


# In[39]:


rainfall = rainfall.groupby('STATE_UT_NAME').sum()
rainfall = rainfall[['ANNUAL']]
rainfall.columns = ['Annual Rainfall']
rainfall.head()


# In[40]:


rainfall.index = rainfall.index.str.lower()


# In[41]:


rainfall.rename(index={'andaman and nicobar islands': 'a & n islands',
                        'chatisgarh':'chhattisgarh',
                        'dadar nagar haveli':'d & n haveli',
                        'daman and dui':'daman & diu',
                        'delhi':'delhi (ut)',
                        'himachal':'himachal pradesh',
                        'jammu and kashmir':'jammu & kashmir',
                        'orissa':'odiasha',
                        'pondicherry':'puducherry',
                        'uttaranchal':'uttarakhand'},inplace=True)


# In[42]:


agricultural_suicides = professional_profile[professional_profile['Type']=='Farming/Agriculture Activity']
agricultural_suicides = agricultural_suicides.groupby('State').sum()
agricultural_suicides.index = agricultural_suicides.index.str.lower()
agricultural_suicides.head()


# In[43]:


agr_suicide_rainfall_data = pd.concat([rainfall,agricultural_suicides],axis=1)
agr_suicide_rainfall_data.head()


# In[44]:


sns.lmplot(x='Annual Rainfall',y='Total',data=agr_suicide_rainfall_data)
ax = plt.gca()
ax.set_title("Agricultural suicides vs rainfall received")
ax.set_ylabel('Suicides')


# So we were wrong to generalise that agricultural suicides are directly proportional to rainfall patterns. The picture is not that simple. Maybe other factors such as ease of access to financial services(loans etc), medical services etc also play a role.
# 
# getting back to suicides among professionals - 

# In[45]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x='Type',y='Total',hue='Gender',data=professional_profile.sort_values('Total',ascending=False))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title('Suicides categorised by different profession')


# As was the case with students, in professional front too, suicides are much more among men than women. But suicides among housewives(females) are almost double than among any other professional activity.

# In[46]:


social_status = suicides[suicides['Type_code']=='Social_Status']
social_status.head()


# In[47]:


for col in social_status.columns[:-1]:
    print(col,'-',social_status[col].nunique(),'-',social_status[col].unique())


# In[48]:


social_status.head()


# In[49]:


ax = sns.barplot(x='Type',y='Total',data=social_status.sort_values('Total'),hue='Gender',
                 ci=None,order=['Married','Never Married','Widowed/Widower','Seperated','Divorcee'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title('Suicides categorised by marital status')


# In[50]:


suicide_causes = suicides[suicides['Type_code']=='Causes']
suicide_causes.head()


# In[51]:


for col in suicide_causes.columns[:-1]:
    print(col,'-',suicide_causes[col].nunique(),'-',suicide_causes[col].unique())


# In[52]:


suicide_causes_bytype = suicide_causes.groupby('Type').sum().sort_values('Total',ascending=False)
suicide_causes_bytype.head()


# In[53]:


suicide_causes_bytype.drop(['Other Causes (Please Specity)','Causes Not known'],inplace=True)
suicide_causes_bytype.head()


# In[54]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x=suicide_causes_bytype.index,y='Total',data=suicide_causes_bytype,ci=None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title('Major causes of suicides in India')


# The major causes of suicides in India are Family Problems, Prolonged and Mental Illnesses, Drug Abuse and Love Affairs -  all of which could be attended to by proper counselling. The Indian State needs to evolve institutions which can spread awareness about these issues and make psychologists accessable to anyone who is dealing with such issues.

# In[ ]:




