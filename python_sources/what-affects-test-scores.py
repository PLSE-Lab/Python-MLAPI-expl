#!/usr/bin/env python
# coding: utf-8

# # Student Performance EDA
# ## Marco Gancitano

# ### Import packages

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

pass_mark = 60


# Use pass_mark of 60 due to it being the typical failing grade for US High schools.

# ### Import Data and take a quick look

# Rename the dataframe columns to make it easier for referencing

# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')
df.columns = ['gender','race','parent_education','lunch','test_prep','math','reading','writing']


# Check to see if any variables have NA data

# In[ ]:


df.isna().sum()


# Quick look to see how the data is formatted

# In[ ]:


df.head()


# <hr>

# ### Distribution of variables

# #### Gender

# In[ ]:


print('Male:',len(df[df.gender == 'male']))
print('Female:',len(df[df.gender == 'female']))
y = np.array([len(df[df.gender == 'male']),len(df[df.gender == 'female'])])
x = ["Male","Female"]
plt.bar(x,y)
plt.title("Gender Representation")
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.show()


# Pretty even by gender

# #### Parent's Education

# In[ ]:


x = ['Some college','Associates','Highschool','Some highschool','Bachelors','Masters']
y = df.parent_education.value_counts().values
plt.bar(x,y)
plt.title("Parent's Education")
plt.xlabel("Education Level")
plt.ylabel("Frequency")
plt.show()


# Hard to see the exact difference between finished college and not so we'll dig in a little more

# In[ ]:


no_college_idx = np.where([x in ['some college','high school','some high school'] for x in df.parent_education])[0]

df['College'] = 'Degree'
df.College[no_college_idx] = 'No Degree'


# In[ ]:


x = df.College.value_counts().index
y = df.College.value_counts().values

plt.bar(x,y)
plt.title("Parent's Degree")
plt.xlabel("College")
plt.ylabel("Frequency")
plt.show()


# We see there's a majority of parents who haven't finished college

# #### Lunch Payment

# In[ ]:


x = df.lunch.value_counts().index
y = df.lunch.value_counts().values
plt.bar(x,y)
plt.title("Lunch Payment")
plt.xlabel("Payment Type")
plt.ylabel("Frequency")
plt.show()


# I understand free/reduced lunch payment as typically being associated with lower income so we may be able to use this as a proxy for parent's income

# #### Test Scores

# In[ ]:


sns.pairplot(df)


# We see that reading and writing have quite a tight corrleation while math and reading/writing is slightly looser

# In[ ]:


fig,ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# ### Demographic and Economical Test Differences

# #### Gender

# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(1, 3,1)
sns.boxplot(x="gender", y="math", data=df).set_title('Math')
plt.subplot(1, 3,2)
sns.boxplot(x="gender", y="reading", data=df).set_title('Reading')
plt.subplot(1, 3,3)
sns.boxplot(x="gender", y="writing", data=df).set_title('Writing')


# Men do better in math while women do better in reading and writing. An interesting insight is the downside variation in womens' test scores. Not sure what the cause of this could be.

# #### "Income" (Lunch Payment)

# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(1, 3,1)
sns.boxplot(x="lunch", y="math", data=df).set_title('Math')
plt.subplot(1, 3,2)
sns.boxplot(x="lunch", y="reading", data=df).set_title('Reading')
plt.subplot(1, 3,3)
sns.boxplot(x="lunch", y="writing", data=df).set_title('Writing')


# We see that on average students who recieve free/reduced lunch have lower test scores for all three subjects compared to students who pay standard price for lunch. This may stem from a connection of lower income household leading to lower test scores.

# #### Parent's Education

# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(1, 3,1)
sns.boxplot(x="College", y="math", data=df).set_title('Math')
plt.subplot(1, 3,2)
sns.boxplot(x="College", y="reading", data=df).set_title('Reading')
plt.subplot(1, 3,3)
sns.boxplot(x="College", y="writing", data=df).set_title('Writing')


# We see that, on average, students whose parents have a college degree do better on each test

# ### Failing Students

# Next we look at the subset of students who failed each test (failing grade of 60) and see if we have any interesting insights.

# In[ ]:


failing = df[(df.math < pass_mark) & (df.writing < pass_mark) & (df.reading < pass_mark)]


# #### Gender

# In[ ]:


y = np.array([len(failing[failing.gender == 'male']),len(failing[failing.gender == 'female'])])
x = ["Male","Female"]
plt.bar(x,y)
plt.title("Gender Representation")
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.show()


# We see here that although for the entire dataset it's a slight majority to female but students who failed all the tests has a decent majority to men.

# #### "Income" (Lunch Payment)
# 

# In[ ]:


x = failing.lunch.value_counts().index
y = failing.lunch.value_counts().values

plt.bar(x,y)
plt.title("Lunch Payment")
plt.xlabel("Payment Type")
plt.ylabel("Frequency")
plt.show()


# Again, we see that the distribution has swapped from the full data set, showing that students who fail are more likely to be on a free/reduce lunch payment.

# #### Parent's education

# In[ ]:


x = failing.College.value_counts().index
y = failing.College.value_counts().values

plt.bar(x,y)
plt.title("Parent's Degree")
plt.xlabel("College")
plt.ylabel("Frequency")
plt.show()


# Furthermore, we see that will parent's education the majority grows when looking at students who failed.

# #### Test Scores

# In[ ]:


sns.pairplot(failing)


# It seems the correlations break down a little bit within this subset

# In[ ]:


fig,ax = plt.subplots(figsize=(6, 4))
sns.heatmap(failing.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# Confirmed with correlation matrix

# ### Conclusion

# We see that certain demographic and economic variables like gender, parent's education, and a proxy for household income has an affect on test scores.

# ### Future Works

# Things to work on in the future:
# 1. Regression models to estimate test scores.
# 2. Classificaiton models to estimate if students pass.
# 3. Run Clustering algorithms to see variable relations.
