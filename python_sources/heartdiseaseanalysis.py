#!/usr/bin/env python
# coding: utf-8

# This kernel is an attempt to perform some prelimnary analysis on the heart disease dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


heart= pd.read_csv('../input/heart.csv')
print(heart.shape)
heart.head()


# In[ ]:


#get info on all the null entries 
heart.info()


# In[ ]:


# get the unique entries in each column
print('unique entries in each column')
heart.nunique()


# In[ ]:


# distribution of the range of ages for heart attack plot
heart.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',
       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']


# In[ ]:


heart.head()


# In[ ]:


bg_color = (0.25, 0.25, 0.25)
sns.set(rc={"font.style":"normal",
            "axes.facecolor":bg_color,
            "figure.facecolor":bg_color,
            "text.color":"white",
            "xtick.color":"white",
            "ytick.color":"white",
            "axes.labelcolor":"white",
            "axes.grid":False,
            'axes.labelsize':25,
            'figure.figsize':(10.0,5.0),
            'xtick.labelsize':15,
            'ytick.labelsize':15})  


# In[ ]:


heart['Age'].hist(grid=False)


# Majority of the people get a heart attack around the age of 55-60

# In[ ]:


# subselect features of interest and then perform classification 
# divide dataset into training, test and validation 
sns.swarmplot(heart['Age'])


# In[ ]:


result=[]
for i in heart['ChestPain']:
    if i == 0:
        result.append('Typical Angina')
    if i ==1:
        result.append('Atypical Angina')
    if i ==2:
        result.append('Non-Anginal')
    if i==3:
        result.append('Asymptomatic')
        
heart['ChestPainType']=pd.Series(result)

sns.swarmplot(x='ChestPainType', y='Age', data=heart)


# In[ ]:


#check the relation of major vessels and chest pain type
ax=sns.countplot(hue=result,x='MajorVessels',data=heart,palette='husl')

    


# In[ ]:


# plot the pie chart indicating distribution of each chest pain type
ChestPain=(heart['ChestPainType']).value_counts()
percent_typAng= ChestPain[0] *100/ len(heart)
percent_AtypAng=ChestPain[1]*100/len(heart)
percent_nonAng=ChestPain[2]*100/len(heart)
percent_none=ChestPain[3]*100/len(heart)

values= [percent_typAng, percent_AtypAng, percent_nonAng, percent_none]
labels=['Typical Angina','Atypical Angina','Non-Anginal','Asymptomatic']
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.title("Chest Pain Type Percentage")    
plt.show()


# Most of the people above the age of 55 have typical angina chest pain

# In[ ]:


# do a gender comparison 
ax = sns.countplot(hue=result,x='Gender',data=heart,palette='husl')

plt.title("Chest Pain Type Vs Gender")    
plt.ylabel("")
plt.yticks([])
plt.xlabel("")
for p in ax.patches:
    ax.annotate(p.get_height(),(p.get_x()+0.05, p.get_height()+1))
ax.set_xticklabels(['Female','Male'])
print(ax.patches)


# It can be seen that 73% of the patients who have typical angina are males

# In[ ]:


#check if bp and cholestrol are correlated
ax = sns.regplot(x='RestingBloodPressure', y='Cholestrol',data=heart, color="g")


# In[ ]:


heart_health=[]

