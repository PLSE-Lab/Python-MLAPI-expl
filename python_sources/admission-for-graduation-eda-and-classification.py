#!/usr/bin/env python
# coding: utf-8

# # Graduate Admission
# This kernel is my attempt to analyze the factors that play a key role in deciding whether you will be admitted to a graduate program based on your GRE SCore, CGPA etc.

# Let's start by importing some libraries that we need.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


df.columns


# Let's drop the Serial No column as it is a unique number and will not play a plart in the final model

# In[ ]:


df.drop(['Serial No.'], inplace=True, axis=1)
df.head()


# Remove the spaces from some of the column. It is not mandatory but it will make your life a tad easier.

# In[ ]:


df.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)
df.columns


# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x='GRE Score', y='Chance of Admit', data=df)


# Higher your GRE Score, higher your chance of getting admiited.

# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x='TOEFL Score', y='Chance of Admit', data=df)


# Higher your TOFEL Score, higher your chance of getting admitted.

# In[ ]:


plt.figure(figsize=(15, 9))
sns.boxplot(x='CGPA', y='Chance of Admit', data=df)


# If your CGPA is higher, chances of getting admitted is highre which is expected

# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x='SOP', y='Chance of Admit', data=df)


# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x='LOR', y='Chance of Admit', data=df)


# In[ ]:


plt.figure(figsize=(15, 9))
sns.scatterplot(x='GRE Score', y='TOEFL Score', hue='Research', data=df)


# Above plot shows that students with higher GRE and TOFEL Score are more inclined to do research.

# Let's create an admit indicating that student has been adnitted to a college if the chance of getting an admit is greater than 0.7

# In[ ]:


def hasAdmitted(data):
    if data > 0.7:
        return 1
    else:
        return 0
df['Admit'] = df['Chance of Admit'].apply(hasAdmitted)
df.head()


# Drop the Chance of Admit as we no longer need it.

# In[ ]:


df.drop(['Chance of Admit'], inplace=True, axis=1)
df.head()


# In[ ]:


print(df['Admit'].value_counts())
plt.figure(figsize=(15, 9))
sns.countplot(x='Admit', data=df)


# Let's explore the relationship between the variable and see if they play a role in getting admitted to a college

# In[ ]:


#plt.figure(figsize=(15, 9))
sns.scatterplot(x='GRE Score', y='TOEFL Score', hue='Admit', data=df)


# In[ ]:


#plt.figure(figsize=(15, 9))
sns.scatterplot(x='SOP', y='LOR', hue='Admit', data=df)


# In[ ]:


sns.scatterplot(x='GRE Score', y='CGPA', hue='Admit', data=df)


# In[ ]:


sns.scatterplot(x='TOEFL Score', y='CGPA', hue='Admit', data=df)


# In[ ]:


sns.heatmap(df.corr(), annot=True)


# CGPA and GRE Score play a dominant role in deciding whether you will get admitted to a certain college.

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(df.drop(['Admit'], axis=1),df['Admit'],
                                               test_size=0.1,random_state=50)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[ ]:


rf.score(X_test, y_test)


# In[ ]:


feat_importances = pd.Series(rf.feature_importances_, index=df.drop('Admit', axis=1).columns)
feat_importances.sort_values(ascending=False).plot(kind='barh')


# In[ ]:




