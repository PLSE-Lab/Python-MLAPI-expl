#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import all libraries used in the code
import numpy as np
import pandas as pd
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt

#ignore warnings
warnings.filterwarnings('ignore')

# Open the data
df = pd.read_csv("../input/StudentsPerformance.csv")


# lets check the data to better understand it

# In[ ]:


print(df.head())
print(df.info())
print(df.describe())
print(df.columns)


# I can see that there is no missing value in data <br>
# but you can also use this method to see number of empty values in each column

# In[ ]:


print('Empty cells in data')
print(df.isnull().sum())


# lets plot number of different features on bar chart 

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(18, 10))

sns.countplot(x='gender',  palette='Blues', data=df, ax=axes[0, 0])
sns.countplot(x='test preparation course', palette='Blues', data=df, ax=axes[0, 1])
sns.countplot(x='race/ethnicity', palette='Blues', data=df, ax=axes[1, 0])
sns.countplot(x='parental level of education', palette='Blues', data=df, ax=axes[1, 1])

f.tight_layout()


# In[ ]:


# create a dist plot
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

sns.distplot(df['reading score'], color='skyblue', ax=axes[0])
sns.distplot(df['math score'], color='gold', ax=axes[1])
sns.distplot(df['writing score'], color='red', ax=axes[2])

f.tight_layout()


# Creating box plot would help us to understand how different features effect exam score <br>
# Couple insights that we see from the data <br>
# * **female** students are more successful on reading and write but **male** students are more successful on math 
# * **lunch** is affecting student notes - students with free/reduced lunch get worse marks
# * also students with parents have **master degree** get better marks
# * lastly **test preparation course **  is another important dimension - sudents who has completed this course get better marks 
# 

# In[ ]:


# create box plot
f, axes = plt.subplots(ncols=3, figsize=(24, 12), sharex=True)
sns.boxplot(x='gender', y='math score', data=df, ax=axes[0], palette="Set3")
sns.boxplot(x='gender', y='reading score', data=df, ax=axes[1], palette="Set3")
sns.boxplot(x='gender', y='writing score', data=df, ax=axes[2], palette="Set3")

f, axes = plt.subplots(ncols=3, figsize=(24, 12), sharex=True)
sns.boxplot(x='lunch', y='math score', data=df, ax=axes[0], hue='gender', palette="Set3")
sns.boxplot(x='lunch', y='reading score', data=df, ax=axes[1], hue='gender', palette="Set3")
sns.boxplot(x='lunch', y='writing score', data=df, ax=axes[2], hue='gender', palette="Set3")

f, axes = plt.subplots(ncols=3, figsize=(24, 12), sharex=True)
sns.boxplot(x='parental level of education', y='math score', data=df, ax=axes[0], hue='gender', palette="Set3")
sns.boxplot(x='parental level of education', y='reading score', data=df, ax=axes[1], hue='gender', palette="Set3")
sns.boxplot(x='parental level of education', y='writing score', data=df, ax=axes[2], hue='gender', palette="Set3")

f, axes = plt.subplots(ncols=3, figsize=(24, 12), sharex=True)
sns.boxplot(x='test preparation course', y='math score', data=df, ax=axes[0], hue='gender', palette="Set3")
sns.boxplot(x='test preparation course', y='reading score', data=df, ax=axes[1], hue='gender', palette="Set3")
sns.boxplot(x='test preparation course', y='writing score', data=df, ax=axes[2], hue='gender', palette="Set3")

f.tight_layout()

