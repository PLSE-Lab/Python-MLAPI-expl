#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 


# In[ ]:


data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data.head()


# In[ ]:


placed=pd.DataFrame(data[data['status']=='Placed'])


# In[ ]:


data.info()


# **What factors lead to a student getting placed??**

# In[ ]:


data['status'].value_counts()


# Out of 215 students 148 students got placed

# In[ ]:


data['salary']=data['salary'].fillna(0.0)


# Is there any relationship between mba percent and aptitude test percent?

# In[ ]:


sns.scatterplot(x=data['mba_p'],y=data['etest_p'])


# All over the place

# In[ ]:


sns.regplot(x=data['mba_p'],y=data['etest_p'])


# ### Very slight correlation between mba % and Etest%(Not reliable)

# In[ ]:


sns.lmplot(x='mba_p',y='etest_p',hue='status',data=data)


# #### Lets see the correlation between those being placed and both metrics individually

# In[ ]:


sns.swarmplot(x=data['status'],y=data['mba_p'])


# #### Seems to be no correlation

# In[ ]:


sns.swarmplot(x=data['status'],y=data['etest_p'])


# No correlation between grades and placement status at all

# In[ ]:


sns.swarmplot(x=data['status'],y=data['degree_p'])


# ### Placed students tend to have better degree percentage

# In[ ]:


m_data=placed[placed['gender']=='M']
f_data=placed[placed['gender']=='F']


# In[ ]:


sns.distplot(a=m_data['salary'],kde=False,label='Salary for Men')
sns.distplot(a=f_data['salary'],kde=False,label='Salary for Women')

plt.legend()


# In[ ]:


sns.barplot(y=m_data['salary'],x=data['gender'])
sns.barplot(y=f_data['salary'],x=data['gender'])


# In[ ]:


sns.barplot(x=placed['gender'],y=placed['mba_p'])


# #### Men tend get higher packages as compared to women eventhough women tend to have better mba percentage

# In[ ]:


data.head()


# In[ ]:


sns.barplot(x=data['specialisation'],y=data['salary'])


# In[ ]:


sns.barplot(x=data['specialisation'],y=data['mba_p'])


# ##### Finance students tend to get higher packages even though they on average score same as HR students

# In[ ]:


ndata=placed.groupby(['gender','specialisation'])[['salary']].mean().reset_index()


# In[ ]:


ndata


# In[ ]:


ndata['ss']=['female finance','female HR','male finance','male HR']


# In[ ]:


sns.barplot(x=ndata['ss'],y=ndata['salary'])


# Male finance students get the best packages and Female HR students tend to get worst packages

# In[ ]:


sns.countplot(x='workex',data=data)


# In[ ]:


sns.barplot(x=placed['workex'],y=placed['salary'])


# #### Students with work experience get higher packages

# In[ ]:


sns.barplot(x=placed['ssc_b'],y=placed['salary'])


# #### Average Package for both is same

# In[ ]:


ismale=(data['gender']=='M')


# In[ ]:


isfemale=(data['gender']=='F')


# In[ ]:


isplaced=(data['status']=='Placed')


# In[ ]:


ismale.value_counts()#There are 139 boys and 76 girls


# In[ ]:


isplaced.value_counts()#148 students are placed and 67 arent


# In[ ]:


(ismale & isplaced).value_counts()#Out of 148 boys 100 got placed


# In[ ]:


100/148 # Probability of a boy getting placed is 0.675


# In[ ]:


(isfemale&isplaced).value_counts()#Out of 76 girls 48 got placed


# In[ ]:


48/76 #Probability of a girl geting placed is 0.63
#Therefore boys have a higher chance of getting placed


# In[ ]:


sns.regplot(x=data['ssc_p'],y=data['hsc_p'])


# #### Students with high marks in 10th tend get high marks in 12th

# In[ ]:


placed


# In[ ]:


sns.regplot(x=placed['degree_p'],y=placed['salary'])


# #### No correlation between degree marks and salary

# In[ ]:


sns.countplot(x='status',data=data)


# In[ ]:


values = [(data['ssc_p'].mean()),(data['hsc_p'].mean()),(data['mba_p'].mean()),(data['degree_p'].mean()),(data['etest_p'].mean())]
names=['ssc_p','hsc_p','mba_p','degree_p','etest_p']


# In[ ]:


sns.barplot(x=names,y=values)


# In[ ]:


data['workex'].replace(to_replace='Yes',value=1,inplace=True)
data['workex'].replace(to_replace='No',value=0,inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




