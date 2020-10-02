#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# > ***The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer***

# In[ ]:


df=pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv')
df.head()


# In[ ]:


df=df.rename(columns={"30": "Age", "64": "Op_Year","1":"axil_nodes","1.1":"Surv_status"})


# In[ ]:


df.head()


# In[ ]:


df.info()
df.shape


# In[ ]:


#Check how many of them have survived
df['Surv_status'].value_counts()


# ***Attribute Information:***
# 
# * Age of patient at time of operation (numerical)
# * Patient's year of operation (year - 1900, numerical)
# * Number of positive axillary nodes detected (numerical)
# * Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 years.

# In[ ]:


#Check how many of them have survived
df['Surv_status'].value_counts().plot.pie(explode=[0.01,0.01],autopct="%.1f%%")


# In[ ]:


df['Age'].value_counts()


# In[ ]:


df['Age'].max()


# In[ ]:


df['Age'].min()


# In[ ]:


#Range
df['Age'].max()-df['Age'].min()


# **Columns**
# * 
# 30 -Age
# * 64-Op_Year
# * 1-axil_nodes
# * 1.1-Surv_status

# In[ ]:


#Frequency Distribution Tables.
df1 = pd.Series(df['Age']).value_counts().sort_index().reset_index().reset_index(drop=True)
df1.columns = ['Age','Surv_status']
print(df1)


# In[ ]:


#Bar Charts.

df.plot(kind="scatter",x="Age",y="axil_nodes")


# In[ ]:


#Bar Charts.

df.plot(kind="bar",y="Age",x="Surv_status",color="blue")


# In[ ]:


sns.set(style="whitegrid")
sns.barplot(y="Age",x="Surv_status",hue="Age",data=df)


# In[ ]:


sns.set(style="whitegrid")
sns.barplot(y="Age",x="Surv_status",data=df)


# Some ways you can describe patterns found in univariate data include central tendency (mean, mode and median) and dispersion: range, variance, maximum, minimum, quartiles (including the interquartile range), and standard deviation.
# 
# You have several options for describing data with univariate data. 
# * Frequency Distribution Tables.
# * Bar Charts.
# * Histograms.
# * Frequency Polygons.
# * Pie Charts.

# In[ ]:


sns.distplot(df['Age'],rug=True)
#RugPlot is used to display the distribution of the data


# In[ ]:


sns.kdeplot(df['Age'],shade="True")

