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
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
plt.show()


# In[ ]:


file_path_train='../input/osic-pulmonary-fibrosis-progression/train.csv'
raw_data =pd.read_csv(file_path_train)
raw_data.head()


# In[ ]:


raw_data.info()


# # Smoking Status

# In[ ]:


# reading the data from the first week of evry patient
df=raw_data.groupby(['Patient']).first()
df.head()


# In[ ]:


print('The Totl number of patients visited :',len(raw_data.Patient.unique()))


# In[ ]:


Smoke=df.groupby(['SmokingStatus']).count()['Sex'].to_frame()
Smoke


# In[ ]:


sns.barplot(x=Smoke.Sex.keys(),y=Smoke.Sex.values)


# In[ ]:


df.groupby(['Sex']).count()['SmokingStatus'].to_frame()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='SmokingStatus', hue='Sex')


# ## In-sights
# `1.There are more number of patients belongs to ex-smokers and currently smoking patients are very less.
# 2.From gender prespective many patients are men` 

# # Age distirbution

# In[ ]:


# Age dostribution plot 
mu=df.Age.std()
mean=df.Age.mean()
plt.figure(figsize=(10,6))
plt.title('Age distirbution [mu {:.2f} and mean {:.2f}]'.format(mu,mean),fontsize=15,color='black')
sns.distplot(df['Age'],kde=True)


# In[ ]:


# smoking staus versus Age distribution
smoker_dist=df.loc[df.SmokingStatus=='Currently smokes']['Age']
exsmoker_dist=df.loc[df.SmokingStatus=='Ex-smoker']['Age']
nonsmoker_dist=df.loc[df.SmokingStatus=='Never smoked']['Age']

plt.figure(figsize=(10,6))
sns.kdeplot(smoker_dist,shade=True,label='currenty smokes')
sns.kdeplot(exsmoker_dist,shade=True,label='Ex-smoker')
sns.kdeplot(nonsmoker_dist,shade=True,label='Never smoked')


# In[ ]:


# Gender and Age distribution
Male_dist=df.loc[df.Sex=='Male']['Age']
Female_dist=df.loc[df.Sex=='Female']['Age']

plt.figure(figsize=(10,6))
sns.kdeplot(Male_dist,shade=True,label='Male')
sns.kdeplot(Female_dist,shade=True,label='Female')


# In[ ]:


plt.figure(figsize=(10,6))
sns.swarmplot(x=df["Sex"],y=df['Age'],hue=df['SmokingStatus'])


# # FVC and Percentage

# In[ ]:


patient_ids=raw_data.Patient.unique()

patient_week=[]
patient_fvc=[]
patient_percentage=[]
for ids in patient_ids:
    week=raw_data.loc[raw_data['Patient']==ids]['Weeks'].values
    fvc=raw_data.loc[raw_data['Patient']==ids]['FVC'].values
    percent=raw_data.loc[raw_data['Patient']==ids]['Percent'].values
    patient_week.append(week)
    patient_fvc.append(fvc)
    patient_percentage.append(percent)


# In[ ]:


plt.figure(figsize=(10,10))
plt.title("Each patient's FVC decay over the weeks")
plt.xlabel('Weeks')
plt.ylabel('FVC deacy ')
for i in range(len(patient_ids)):
    sns.lineplot(x=patient_week[i],y=patient_fvc[i],label ='P'+str(i+1),lw=1,legend='full')


# In[ ]:


plt.figure(figsize=(10,10))
plt.title("Each patient's Percentage over the weeks")
plt.xlabel('Weeks')
plt.ylabel('Percentage')
for i in range(len(patient_ids)):
    sns.lineplot(x=patient_week[i],y=patient_percentage[i],label ='P'+str(i+1),lw=1)


# In[ ]:


plt.figure(figsize=(10,10))
plt.title("Each patient's Percentage Vs FVC")
plt.xlabel('FVC')
plt.ylabel('Percentage')
for i in range(len(patient_ids)):
    sns.lineplot(x=patient_fvc[i],y=patient_percentage[i],label ='P'+str(i+1),lw=1)


# In[ ]:




