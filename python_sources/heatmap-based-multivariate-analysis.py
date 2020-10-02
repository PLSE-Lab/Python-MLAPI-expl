#!/usr/bin/env python
# coding: utf-8

# Hello<br>
# Welcome to my first kaggle kernel. <br>
# Suggestions/corrections appreciated! <br>
# TO DO:
# * End-to-End ML pipeline

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import re
import os
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from __future__ import division
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

print(os.listdir("../input"))


# In[ ]:


sp_data = pd.read_csv('../input/StudentsPerformance.csv')
sp_data.head()


# In[ ]:


sp_data.info()


# In[ ]:


# print (sp_data[1])


# It seems we have only 1000 rows of data. Not a lot to work with.<br> 
# On the brigh side, none of them are null/NaN values.<br> 
# Let the analysis begin!

# In[ ]:


for column in (sp_data.columns.values):
    if (sp_data[column].dtype) != np.dtype('int'):
        print ("Unique values in '"+ column + "' column are ", end='')
        print (sp_data[column].unique())


# In[ ]:


print (sp_data['gender'].value_counts())
sns.countplot(x='gender', data=sp_data);


# In[ ]:


print (sp_data['race/ethnicity'].value_counts())
ax = sns.countplot(x='race/ethnicity', data=sp_data, order= ['group A', 'group B', 'group C', 'group D', 'group E'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()


# In[ ]:


print (sp_data['parental level of education'].value_counts())
ax = sns.countplot(x='parental level of education', data=sp_data, order=['some high school', 'high school', 'associate\'s degree', 'some college',
                                                                         "bachelor's degree","master's degree"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()


# In[ ]:


print (sp_data['lunch'].value_counts())
sns.countplot(x='lunch', data=sp_data);


# In[ ]:


sns.countplot(x='test preparation course', data=sp_data);


# In[ ]:


sns.barplot(x='gender',y='math score',data=sp_data);


# In[ ]:


sp_data[['math score','reading score','writing score']].describe()


# In[ ]:


sns.distplot(sp_data['math score'], bins=25, kde=False);


# In[ ]:


sns.distplot(sp_data['reading score'], bins=25, kde=False);


# In[ ]:


sns.distplot(sp_data['writing score'], bins=25, kde=False);


# Lets convert the string to numerical values that can be used for further analysis

# In[ ]:


sp_data['gender'].loc[sp_data['gender'] == 'male'] = 1
sp_data['gender'].loc[sp_data['gender'] == 'female'] = 0


# In[ ]:


sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group A'] = 1
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group B'] = 2
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group C'] = 3
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group D'] = 4
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group E'] = 5


# In[ ]:


# sp_data['lunch'].loc[sp_data['lunch'] == 'standard'] = 1
# sp_data['lunch'].loc[sp_data['lunch'] == 'free/reduced'] = 0


# MData["test preparation course"]=MData["test preparation course"].replace({"none":0,"completed":1})

sp_data['lunch'] = sp_data['lunch'].replace({'standard':1,'free/reduced':0})


# In[ ]:


sp_data['test preparation course'].loc[sp_data['test preparation course'] == 'none'] = 0
sp_data['test preparation course'].loc[sp_data['test preparation course'] == 'completed'] = 1


# In[ ]:


sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'some high school'] = 1
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'high school'] = 2
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'associate\'s degree'] = 3
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'some college'] = 4
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'bachelor\'s degree'] = 5
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'master\'s degree'] = 6


# It can be seen that the value given to masters degree is the greatest while compared to 'some high school'. The ascending order of values wrt to level of the degree obtained will help us with correlation analysis.

# This is how our dataframe looks now:

# In[ ]:


sp_data.head(10)


# Now lets see the major factors that contribute to test outcomes.

# In[ ]:


(sp_data[['race/ethnicity','math score', 'reading score', 'writing score']].corr())
ax = sns.heatmap(sp_data.corr(),cmap="Blues",annot=True,annot_kws={"size": 7.5},linewidths=.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
# plt.tight_layout()


# Inference from the correlaation matrix:
# * Theres a psotive correlation between gender male and math scores. There is a negetive correlation between gender male and writing & rerading scores. This menas that boys seem perform better than girls in math whereas the oppsoite relation holds for the language based tests.
# * We don't excatly know the constitutents of each of the race/ethnicity groups. Eg: We dont know if group A cossits of Asians or if group C is Latino. Its best we don't make any conslusions for the tag constituents. But we can see that as we approach higher group levels (group a -> group b -> ... -> group e) the test scores improve.
# * The correlation between students scores and having a 'standard' lunch seems to be quite high. Especially their math scores.
# *  The test preperation course seems to be quite effective.
# * The corrleation between the different test scores are high. The high positive correlation means that studnets who do well in a particular test, do well on the other two as well. 
# ** There is an extremely high correlation between the reading and writing scores of students.
# * There are only 2 values that are >0.3. Standard lunch and test prep course seems to have a lot of effect on the final scores.
# 

# In[ ]:


sp_score_data = sp_data[['gender','math score','reading score','writing score']].groupby('gender',as_index=True).mean()
print ('averages: \n'+str(sp_score_data.head()))
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="gender", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="gender", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="gender", y="writing score", data=sp_data, ax=axs[2],showmeans=True);


# In[ ]:


sns.boxplot(x="gender", y="math score", data=sp_data, showmeans=True);


# Our hypothesis of boys performing better than girls in math tests and girls perorming bettter than boys in reading and writing tests has been proved right.

# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="lunch", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="lunch", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="lunch", y="writing score", data=sp_data, ax=axs[2],showmeans=True);


# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="race/ethnicity", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="race/ethnicity", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="race/ethnicity", y="writing score", data=sp_data, ax=axs[2],showmeans=True);


# 1. It can be infered from the above plots that students who have standard lunch do better on all tests that the ones who are having 'free/reduced' lunch

# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="parental level of education", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="parental level of education", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="parental level of education", y="writing score", data=sp_data, ax=axs[2],showmeans=True);


# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="test preparation course", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="test preparation course", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="test preparation course", y="writing score", data=sp_data, ax=axs[2],showmeans=True);


# In[ ]:


reduced_lunch = sp_data.loc[sp_data['lunch'] == 0]
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="test preparation course", y="math score", data=reduced_lunch, ax=axs[0],showmeans=True);
sns.boxplot(x="test preparation course", y="reading score", data=reduced_lunch, ax=axs[1],showmeans=True);
sns.boxplot(x="test preparation course", y="writing score", data=reduced_lunch, ax=axs[2],showmeans=True);


# 
