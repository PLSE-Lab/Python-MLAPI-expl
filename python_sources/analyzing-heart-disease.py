#!/usr/bin/env python
# coding: utf-8

# <ul>
#     <a href='#1'><li>INTRODUCTION</li></a>
#     <a href='#2'><li>DATASET COLUMNS FEATURE EXPLAIN</li></a>
# <!--     <a href='#3'><li>INVESTIGATING THE DATA and EXPLORATORY DATA ANALSIS</li></a>
#         <ul>
#             <a href='#4'><li>One Visualization to Rule Them All</li></a>
#             <a href='#5'><li>Age Analysis</li></a>
#             <a href='#6'><li>Sex (Gender) Analysis</li></a>
#             <a href='#7'><li>Chest Pain Type Analysis</li></a>
#             <a href='#8'><li>Age Range Analysis</li></a>
#             <a href='#9'><li>Thalach Analysis</li></a>
#             <a href='#10'><li>Thal Analysis</li></a>
#             <a href='#11'><li>Target Analysis</li></a>
#         </ul> -->
# <!--     <a href='#12'><li>MODEL, TRAINING and TESTING</li></a>
#         <ul>
#             <a href='#13'><li>Logistic Regression</li></a>
#             <a href='#14'><li>K-Nearest Neighbors</li></a>
#              <a href='#15'><li>Naive Bayes</li></a>
#              <a href='#16'><li>Decision Tree</li></a>
#              <a href='#17'><li>Random Forest</li></a>
#              <a href='#18'><li>Gradient Boosting Machine</li></a>
#              <a href='#19'><li>Kernelized SVM</li></a>
#         </ul>
#     <a href='#20'><li>CONCLUSION</li></a>
#     <a href='#21'><li>REFERENCES</li></a>  -->
# </ul>

# <p id='1'><h3>INTRODUCTION</h3></p>
# 
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

# <p id='2'><h3>DATASET COLUMNS FEATURE EXPLAIN</h3></p>
# 
# * age (age in years)
# * sex (1 = male; 0 = female)
# * cp (chest pain type)
# * trestbps (resting blood pressure (in mm Hg on admission to the hospital))
# * chol (serum cholestoral in mg/dl)
# * fbs (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# * restecg (resting electrocardiographic results)
# * thalach (maximum heart rate achieved)
# * exang (exercise induced angina (1 = yes; 0 = no)
# * oldpeak (ST depression induced by exercise relative to rest)
# * slope (the slope of the peak exercise ST segment)
# * ca (number of major vessels (0-3) colored by flourosopy)
# * thal (3 = normal; 6 = fixed defect; 7 = reversable defect)
# * target (1 or 0)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Now, we are reading our dataset with **read_csv** function that we import pandas library

# In[ ]:


kalp = pd.read_csv("../input/heart.csv")


# **Get the infos of dataset.**
# 
# prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

# In[ ]:


kalp.info()


# In[ ]:


# Return the first 5 rows
kalp.head()


# In[ ]:


# Return the last 5 rows
kalp.tail()


# Show some statistical information of the Heart Disease dataset.

# In[ ]:


kalp.describe()


# In[ ]:


kalp.columns


# Now, check the columns and rows

# In[ ]:


# shape of the dataset : ( 303 rows and 14 columns )
kalp.shape


# In[ ]:


# Check the columns only
len(kalp.columns)


# In[ ]:


# Check the rows only
len(kalp.index)


# In[ ]:


# Checking how many null datas in dataset
print('Data Sum of Null Values \n')
kalp.isnull().sum()


# In[ ]:


#all rows control for null values
kalp.isnull().values.any()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(kalp.corr(),annot=True,fmt='.1f')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(kalp.corr(),vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()


# In[ ]:


# What is the min and max ages
kucukYas = min(kalp.age)
buyukYas = max(kalp.age)
ortalama = kalp.age.mean()
print("Min Age: ", kucukYas)
print("Max Age: ", buyukYas)
print("Mean Age: ", ortalama)


# In[ ]:


genc = kalp[(kalp.age<=30) & (kalp.age<40)]
orta = kalp[(kalp.age<=40) & (kalp.age<55)]
yasli = kalp[(kalp.age>55)]
print('Young Ages: ', len(genc))
print('Middle Ages: ', len(orta))
print('Elderly Ages: ', len(yasli))


# Gender Analysis

# In[ ]:


# Check the Genders how we got
kalp["sex"].value_counts()


# In[ ]:


toplamcinsiyet = len(kalp.sex)
erkek = len(kalp[kalp['sex'] == 1])
bayan = len(kalp[kalp['sex'] == 0])

print('Total Genders: ', toplamcinsiyet)
print('Males: ', erkek)
print('Female: ', bayan)


# In[ ]:


#Sex (1 = male; 0 = female)
sns.countplot(kalp.sex)
plt.show()


# In[ ]:


kalp[['age', 'trestbps', 'sex']].max()


# In[ ]:


kalp[['age', 'trestbps', 'sex']].min()


# In[ ]:


kalp.trestbps.mean()


# In[ ]:


kalp.thalach.mean()


# In[ ]:


# Find Trestbps Averages by age
kalp.groupby("age").mean()[["trestbps"]].head()


# In[ ]:


# How many people are there in the age?
kalp["age"].value_counts()


# In[ ]:


kadin = kalp.loc[kalp.sex==0]
kadin.head()


# In[ ]:


# List the women
kadin.describe()


# In[ ]:


kadin = kalp[kalp.sex==0]
kadin.rename(columns={'sex':'Women'}).set_index('Women').head(20)


# In[ ]:


kadin['trestbps'].plot(kind='hist', figsize=(8,8))
plt.xlabel("trestbps of Women")
plt.title("Histogram of Women")
plt.show()


# In[ ]:


kalp.hist('cp', figsize=(7,7))


# In[ ]:


kadin.hist('cp', figsize=(7,7))


# In[ ]:


kadin.target.hist()


# In[ ]:


kalp.hist(bins=75, figsize=(15,15))

