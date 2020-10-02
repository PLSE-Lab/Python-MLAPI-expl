#!/usr/bin/env python
# coding: utf-8

# **Alaa khalid-1675252, Renad Saklou, Badriah Alsaeedi ******

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
iowa_file_path = '../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv'
home_data = pd.read_csv(iowa_file_path)
home_data.describe(include = "all")


# In[ ]:


home_data.info()


# In[ ]:


home_data.head()


# In[ ]:


home_data.corr()


# In[ ]:


print(pd.isnull(home_data).sum())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=home_data)

#print("Percentage of females who survived:", home_data["Survived"][home_data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

#print("Percentage of males who survived:", home_data["Survived"][home_data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=home_data)


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=home_data)


# In[ ]:


sns.barplot(x="Age", y="Survived", data=home_data)
# the output contain alot of bins, so I have to devide the x into range


# In[ ]:


home_data["Age"] = home_data["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
home_data['AgeGroup'] = pd.cut(home_data["Age"], bins, labels = labels)
sns.barplot(x="AgeGroup", y="Survived", data=home_data)


# In[ ]:


home_data.corr()


# 
