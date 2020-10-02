#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Smat=pd.read_csv("../input/student-mat.csv")


# In[ ]:


dmat=pd.read_csv("../input/student-mat.csv")
dpor=pd.read_csv("../input/student-por.csv")

dtot=pd.merge(dmat,dpor,on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])


# In[ ]:


dmat.head()


# In[ ]:


dpor.head()


# In[ ]:


dtot.shape


# In[ ]:


dtot.columns


# In[ ]:


dtot.describe()


# We add column 'Gtot' that stores the average grade of math and portuguese subjects:

# In[ ]:


dtot['Gtot']=(dtot['G3_x']+dtot['G3_y'])/2


# In[ ]:


plt.figure(figsize=(14,14))
corr=dtot.corr()
sns.heatmap(corr, annot=True)


# We can see from the heatmap above that there are 8 repeted numerical variabels : traveltime,studytime, famrel, freetime, goout, Dalc, Walc, health. Thus we remove the redendency in the following step:

# In[ ]:


l=["traveltime_y","studytime_y", "famrel_y", "freetime_y", "goout_y","Dalc_y","Walc_y","health_y"]
for el in l:
    del dtot[el]


# Now we look at categorical variables and verify if there is a redendency :

# In[ ]:


l=['guardian_','schoolsup_','famsup_','paid_','activities_','higher_','romantic_','absences_']      
[sum(dtot[el+'x']==dtot[el+'y']) for el in l]


# We conclude that we can remove 6 categorical variabels : 'guardian','schoolsup','famsup','paid','activities','higher','romantic','absences'

# In[ ]:


l=['guardian_y','schoolsup_y','famsup_y','activities_y','higher_y','romantic_y']      
for el in l:
    del dtot[el]


# Rename columns:

# In[ ]:


cnew=['guardian','schoolsup','famsup','activities','higher','romantic',"traveltime","studytime", "famrel", "freetime","goout","Dalc","Walc","health"]
cold=[el+'_x' for el in cnew]
dtot.rename(dict(zip(cold,cnew)), axis=1, inplace=True)


# In[ ]:


cold=['failures_x','paid_x','absences_x', 'G1_x', 'G2_x', 'G3_x','failures_y','paid_y', 'absences_y','G1_y', 'G2_y', 'G3_y']
cnew=['failures_mat','paid_mat','absences_mat','G1_mat','G2_mat','G3_mat','failures_por','paid_por','absences_por','G1_por','G2_por','G3_por']
dtot.rename(dict(zip(cold,cnew)), axis=1, inplace=True)


# In[ ]:


dtot.columns


# **Final grade for math and portuguese subject (Gtot: numeric: from 0 to 20) VS workday alcohol consumption (Dalc_x : numeric from 1 - very low to 5 - very high) and  weekend alcohol consumption (Walc_x : numeric from 1 - very low to 5 - very high) :**

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(12,5))
sns.barplot(x='Dalc', y='Gtot', data=dtot,ax=ax[0])
sns.barplot(x='Walc', y='Gtot', data=dtot,ax=ax[1])
plt.text(-4, 14, 'Total grade VS alcohol consumption', fontsize = 16,color='r')


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(12,5))
sns.boxplot(x='Dalc', y='Gtot', data=dtot,ax=ax[0])
sns.boxplot(x='Walc', y='Gtot', data=dtot,ax=ax[1])
plt.text(-4, 22, 'Total grade VS alcohol consumption', fontsize = 16,color='r')


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(12,5))
sns.pointplot(x='Dalc', y='Gtot',hue='Pstatus', data=dtot,ax=ax[0])
sns.pointplot(x='Walc', y='Gtot',hue='Pstatus', data=dtot,ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(17,5))
sns.barplot(x='Dalc', y='Gtot',hue='health', data=dtot,ax=ax[0])
sns.barplot(x='Walc', y='Gtot',hue='health', data=dtot,ax=ax[1])


# **Final grade for math's students (G3: numeric: from 0 to 20) VS workday alcohol consumption (Dalc : numeric from 1 - very low to 5 - very high) and  weekend alcohol consumption (Walc : numeric from 1 - very low to 5 - very high) :**

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(12,5))
sns.barplot(x='Dalc', y='G3', data=dmat,ax=ax[0])
sns.barplot(x='Walc', y='G3', data=dmat,ax=ax[1])
plt.text(-4, 13, 'Math grade VS alcohol consumption', fontsize = 16,color='r')


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(12,5))
sns.boxplot(x='Dalc', y='G3', data=dmat,ax=ax[0])
sns.boxplot(x='Walc', y='G3', data=dmat,ax=ax[1])
plt.text(-4, 22, 'Math grade VS alcohol consumption', fontsize = 16,color='r')


# At this level, we observe no explicit relationship between alcohol consumption and student's results. 

# In[ ]:




