#!/usr/bin/env python
# coding: utf-8

# # Goal:
# To understand what factors drive not only high test scores in each subject, but also *high performance across all three subjects*

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style("darkgrid")


# # Importing and cleaning the data

# In[ ]:


data = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# # Quick and Dirty EDA

# In[ ]:


sns.pairplot(data)


# In[ ]:


sns.heatmap(data.corr(),annot=True,cmap="Blues")


# Not at all surprising that there are high correlations between each of the test subjects.

# But let's see if we can figure out what factors drive higher scores.

# ## Let's dig into scores a bit more

# ## Let's assume that passing in this case is a score of 75

# A score of 75 puts students in about the top third of scores for each subject. Let's see what factors might drive that outperformance.

# In[ ]:


passing = data.copy()


# In[ ]:


passing_score = 75
passing["math score"] = passing["math score"].apply(lambda x: x >= passing_score)
passing["reading score"] = passing["reading score"].apply(lambda x: x >= passing_score)
passing["writing score"] = passing["writing score"].apply(lambda x: x >= passing_score)
passing.head()


# Percent of students who scored a 75

# In[ ]:


passing[["math score","reading score","writing score"]].sum()/passing[["math score","reading score","writing score"]].count()


# In[ ]:


sns.heatmap(passing.corr(),annot=True,cmap="Blues")


# Not surprising that there are still high correlations, but high math scores are less highly correlated than reading/writing.

# #### Passing ( <= 75% score) by race/ethnicity

# In[ ]:


factor = "race/ethnicity"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.tight_layout()


# #### Passing ( <= 75% score) by parental level of education

# In[ ]:


factor = "parental level of education"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.xticks(rotation = 45)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.xticks(rotation = 45)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.xticks(rotation = 45)
plt.tight_layout()


# #### Passing ( = 75% score) by lunch

# In[ ]:


factor = "lunch"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.tight_layout()


# #### Passing ( <= 75% score) by test prep

# In[ ]:


factor = "test preparation course"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.tight_layout()


# # What about people who passed all three subjects?

# ## Breakdown of students who passed all three:

# In[ ]:


z = 75 # Chosen passing score
data["all pass"] = (data["math score"] >= z) & (data["reading score"] >= z) & (data["writing score"] >= z)


# In[ ]:


data[data["all pass"] == True].head()


# #### Just 21.1% of students passed with at least a 75 in each subject:

# In[ ]:


data[data["all pass"] == True]["all pass"].count()/data["all pass"].count()


# #### What percent of the total in each of the factors passed in all three subjects?

# In[ ]:


factor = "race/ethnicity"
plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score")
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()


# In[ ]:


factor = "race/ethnicity"

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass",order=["group A","group B","group C","group D","group E"])

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass",order=["group A","group B","group C","group D","group E"])

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass",order=["group A","group B","group C","group D","group E"])
plt.tight_layout()


# In[ ]:


factor = "parental level of education"

plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score",
            order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()


# In[ ]:


factor = "parental level of education"

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass",
           order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass",
           order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass",
           order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)
plt.tight_layout()


# In[ ]:


factor = "lunch"

plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score")
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()


# In[ ]:


factor = "lunch"

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass")

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass")

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass")
plt.tight_layout()


# In[ ]:


factor = "test preparation course"

plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score")
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()


# In[ ]:


factor = "test preparation course"

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass")

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass")

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass")
plt.tight_layout()


# # Takeaways:

# The test preparation course is really the only factor is not an inherent quality of the student. But at glancing at the data, we do see that it reduces the variance of outcomes.
# 
# Next step: to dive deeper into test preparation across each of the other variables. For example, how/if test preparation improves scores across race/ethnicity, parental education, etc.

# In[ ]:




