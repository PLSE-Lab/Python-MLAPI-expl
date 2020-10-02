#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


df.head()


# In[ ]:


#This librarys is to work with matrices
import pandas as pd 
# This librarys is to work with vectors
import numpy as np
# This library is to create some graphics algorithmn
import seaborn as sns
# to render the graphs
import matplotlib.pyplot as plt
# import module to set some ploting parameters
from matplotlib import rcParams
# Library to work with Regular Expressions
import re

# This function makes the plot directly on browser
get_ipython().run_line_magic('matplotlib', 'inline')

# Seting a universal figure size 
rcParams['figure.figsize'] = 10,8


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


len(df.Location.value_counts())


# In[ ]:


df.Location.value_counts()


# In[ ]:


df['PLocation'] = df['Location'].str.split(' |,').str[0]


# In[ ]:


df['PLocation']


# In[ ]:





# ## Place-wise job crawled 

# In[ ]:


fig,ax = plt.subplots(figsize = (15,120))
sns.countplot(y='PLocation',data = df )
plt.show()


# In[ ]:


df['Job Experience Required'].value_counts()


# In[ ]:



df['Job Salary'].value_counts()


# In[ ]:


len(df['Job Salary'].value_counts())


# In[ ]:


df['Date crawl']=df['Crawl Timestamp'].str.split(' ').str[0]


# In[ ]:





#  ## Date-wise Number of job crawled 

# In[ ]:


fig,ax = plt.subplots(figsize = (15,120))
sns.countplot(x='Date crawl',data = df )
plt.show()


# In[ ]:


skills=df['Key Skills'].str.split('|')


# In[ ]:


skills=skills.to_frame()


# In[ ]:


skills=skills.dropna()


# In[ ]:


skill={}
for index,row in skills.iterrows():
        for i in row['Key Skills']:
            if i.strip() in skill:
                skill[i.strip()]+=1
            else:
                skill[i.strip()]=1
            print(i.strip())
   
    


# In[ ]:


skills=pd.DataFrame([skill])


# In[ ]:


#skills.shape
skills


# In[ ]:


temp = sorted ( skill)


# In[ ]:


for i in sorted (skill) : 
    print ((i, temp[i]), end =" ") 


# In[ ]:





# In[ ]:


temp


# In[ ]:


skills=skills.T


# In[ ]:


new_skills=skills.rename(columns={0:'count'})
new_skills=new_skills.sort_values('count', ascending=False)


# In[ ]:





# In[ ]:


top_skill=new_skills[:][:10]


# In[ ]:





# ## Top 10 skills required at Naukri.com

# In[ ]:


ax=top_skill.plot.bar(y='count',rot=0)


# In[ ]:


titles=df['Job Title'].str.lstrip().value_counts().to_frame()


# In[ ]:


len(df['Job Title'].value_counts())


# In[ ]:


top_title=titles.head(10)


# In[ ]:


titles.max(axis=1)


# In[ ]:





# ## Most wanted jobs

# In[ ]:



ax=top_title.plot.bar(y='Job Title',rot=0,figsize=(25, 10))


# In[ ]:


df['Job Title']=df['Job Title'].str.lstrip()


# In[ ]:


skills_req={}

for index,row in top_title.iterrows():
    skills_req[index]=df[df['Job Title']==index]["Key Skills"]
#     print(index,df.loc[row['Job Title'],['Key Skills']],sep=' : ')
#     print('=====================================')


# In[ ]:



letc=[]
for key,value in skills_req.items():
    for i in value:
        try:
            #print(i.split('|'))
            skills_req[key]=i.split('|')
        except:
            print('try angain:',i)
            letc.append(i)
    print('=============================================')


# In[ ]:


for key,value in skills_req.items():
    print(len(value))


# In[ ]:


#


#  ## Top skills required for the  Top 10 most wanted jobs at Naukri.com 

# In[ ]:


skills_req


# In[ ]:




