#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# An experiment on the effects of anti-anxiety medicine on memory recall when being primed with happy or sad memories. The participants were done on novel Islanders whom mimic real-life humans in response to external factors.
# 
# Drugs of interest (known-as) [Dosage 1, 2, 3]:
# 
# A - Alprazolam (Xanax, Long-term) [1mg/3mg/5mg]
# 
# T - Triazolam (Halcion, Short-term) [0.25mg/0.5mg/0.75mg]
# 
# S - Sugar Tablet (Placebo) [1 tab/2tabs/3tabs]
# 
# *Dosages follow a 1:1 ratio to ensure validity
# *Happy or Sad memories were primed 10 minutes prior to testing
# *Participants tested every day for 1 week to mimic addiction
# 
# Building the Case:
# Obstructive effects of Benzodiazepines (Anti-Anxiety Medicine):
# 
# Long term adverse effects on Long Term Potentiation of synapses, metacognition and memory recall ability
# http://www.jstor.org/stable/43854146
# Happy Memories:
# 
# research shown positive memories to have a deeper and greater volume of striatum representation under an fMRI
# https://www.sciencedirect.com/science/article/pii/S0896627314008484
# Sad Memories:
# 
# research shown sad memories invokes better memory recall for evolutionary purpose whereas, happy memories are more susceptible to false memories
# http://www.jstor.org/stable/40064315
# Participants - all genders above 25+ years old to ensure a fully developed pre-frontal cortex, a region responsible for higher level cognition and memory recall.

# # QUESTION:
# 
# How does anti-anxiety medicine affect you differently by age?
# ---

# # Importing Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Creating Initial DF

# In[ ]:


df1 = pd.read_csv("../input/memory-test-on-drugged-islanders-data/Islander_data.csv")
df1.head()


# # Checking out what this data is looking like

# Data Types
# ---

# In[ ]:


df1.dtypes


# Nulls
# ---

# In[ ]:


df1.isnull().sum()


# Description
# ---

# In[ ]:


df1.describe()


# Dataframe
# ---

# In[ ]:


df1


# # Exploratory Analysis

# Breakdown of age
# ---

# In[ ]:


df1['age'].value_counts()


# Breakdown of the drugs in the dataset, pretty even distribution
# ---

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.countplot(x='Drug', data=df1, ax=ax)
plt.tight_layout()
plt.title('Breakdown of Observations by Drug')
plt.show()


# Average difference in memory score for each drug. Includes all ages and all doses.
# ---

# In[ ]:


df1.groupby('Drug')['Diff'].mean()


# Making a new column for a plot. Avg Diff in memory score for all ages and all doses for each drug

# In[ ]:


df1['avg_ovr_diff'] = df1.groupby('Drug')['Diff'].transform('mean')
df1['avg_ovr_diff']


# Plot

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=df1['Drug'], y=df1['avg_ovr_diff'], ax=ax)
plt.tight_layout()
plt.title('Overall Avg Difference in Memory Score by Drug \n (For all Ages)')
plt.show()


# Time to check out how the drug effects memory by age.
# ---

# Grouping the ages for every 5 years in a new column

# In[ ]:


df1['age_group'] = pd.cut(
    df1['age'],
    np.arange(start=df1['age'].min(), step=5, stop=df1['age'].max())
)
df1[['age', 'age_group']].head()


# In[ ]:


df1.head()


# Making a new column for the average memory score difference by age group

# In[ ]:


df1['avg_age_diff'] = df1.groupby('age_group')['Diff'].transform('mean')
df1[['age_group', 'avg_age_diff']].head()


# Checking out the breakdown of observations by age group.

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.countplot(x='age_group', data=df1, ax=ax)
plt.title('Number of Observations by Age Group')
plt.tight_layout()
plt.show()


# Note: This plot is not specific to any drug. This shows how each age group, on average, by the anxiety drugs (including the placebo)

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=df1['age_group'], y=df1['avg_age_diff'], ax=ax)
plt.title('Avg Change in Memory Score by Age Group \n (Includes Both Drugs and Placebo)')
plt.tight_layout()
plt.show()


# This is all great, but lets check out how each age group is effected by each drug.

# To analyze each drug seperately, I am going to make a seperate DF for each.
# ---

# Alprazolam Dataframe

# In[ ]:


alprazolam_df = df1.loc[df1['Drug'] == 'A']
alprazolam_df


# Triazolam Dataframe

# In[ ]:


triazolam_df = df1.loc[df1['Drug'] == 'T']
triazolam_df


# Sugar Tablet Dataframe

# In[ ]:


sugar_df = df1.loc[df1['Drug'] == 'S']
sugar_df


# Great, now let's see how each drug effects memory test scores by age group.
# ---

# Changing the "avg_age_diff" column for each seperate drug df to be specific to that drug rather than all of the drugs overall.

# In[ ]:


alprazolam_df['avg_age_diff'] = alprazolam_df.groupby('age_group')['Diff'].transform('mean')
alprazolam_df[['age_group', 'avg_age_diff']].head()


# Lets plot the average memory difference for each age group that took Alprazolam.

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=alprazolam_df['age_group'], y=alprazolam_df['avg_age_diff'], ax=ax)
plt.title('Average Memory Score Difference \n (Alprazolam)')
plt.tight_layout()
plt.show()


# Sweet, now let's do Triazolam.

# In[ ]:


triazolam_df['avg_age_diff'] = triazolam_df.groupby('age_group')['Diff'].transform('mean')
triazolam_df[['age_group', 'avg_age_diff']].head()


# And the plot...

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=triazolam_df['age_group'], y=triazolam_df['avg_age_diff'], ax=ax)
plt.plot('Average Memory Score Difference \n (Triazolam)')
plt.tight_layout()
plt.show()


# Interesting, Triazolam seems to not effect memory scores nearly as much as Alprazolam, except for older folks.

# Now time for the placebo

# In[ ]:


sugar_df['avg_age_diff'] = sugar_df.groupby('age_group')['Diff'].transform('mean')
sugar_df[['age_group', 'avg_age_diff']].head()


# And the plot...

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=sugar_df['age_group'], y=sugar_df['avg_age_diff'], ax=ax)
plt.title('Average Memory Score Difference \n (Placebo)')
plt.tight_layout()
plt.show()


# In[ ]:




