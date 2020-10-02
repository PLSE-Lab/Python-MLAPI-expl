#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


data.head()
data["overall score"] = round((data["math score"] + data["writing score"] + data["reading score"]) / 3, 2)
data.head()


# In[ ]:


data['overall score'].plot(kind='hist', bins=100, figsize=(7,7))
plt.xlabel("Overall Score")
plt.ylabel("Frequency")
plt.title("Overall Scores")
plt.show()


# In[ ]:


data.describe()


# In[ ]:


data[data.isnull()=='True'].count()


# In[ ]:


data.info()


# In[ ]:


data.groupby("parental level of education", as_index=True)[["math score", "reading score", "writing score"]].mean()


# In[ ]:


data.corr()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()


# In[ ]:


print("Gender")
print(data.gender.value_counts())


# In[ ]:


print("Race/Ethnicity")
print(data['race/ethnicity'].value_counts())


# In[ ]:


print("Parents Education ")
print(data['parental level of education'].value_counts())


# In[ ]:


sns.countplot(x=data['gender'])
plt.title('Number of males and females in the dataset')


# In[ ]:


sns.countplot(x=data['race/ethnicity'])
plt.title('Number of different races people')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='parental level of education', data = data, hue='gender')
plt.show()


# In[ ]:





# In[ ]:


race_val=[89,190,319,262,140]
race_labels=['Group A','Group B','Group C','Group D','Group E']
plt.pie(race_val,labels=race_labels,radius=1.5,autopct='%0.1f%%',shadow=True)


# In[ ]:


print("Lunch Type")
print(data.lunch.value_counts())


# In[ ]:


sns.countplot(x=data['lunch'])
plt.title('Lunch type')


# In[ ]:


print("Test Preperation course")
print(data['test preparation course'].value_counts())


# In[ ]:


sns.countplot(x=data['test preparation course'])
plt.title('Test Preperation')


# In[ ]:


print("Female")
print(data['race/ethnicity'][data.gender=='female'].value_counts())


# In[ ]:


print("Male")
print(data['race/ethnicity'][data.gender=='male'].value_counts())


# In[ ]:


print("Male")
print(data['lunch'][data.gender=='male'].value_counts(normalize=True))


# In[ ]:


print("Female")
print(data['lunch'][data.gender=='female'].value_counts(normalize=True))


# In[ ]:


sns.countplot(x='lunch', data =data, hue='gender', palette='bright')
plt.show()


# In[ ]:


female1=data[data.gender=='female']
male1=data[data.gender=='male']
print("MATH SCORE")
print("Female")
print("------")
print(female1.groupby(['lunch','test preparation course'])['math score'].max().unstack(level = 'lunch'))
print("Male")
print("------")
print(male1.groupby(['lunch','test preparation course'])['math score'].max().unstack(level = 'lunch'))


# In[ ]:


print("READING SCORE")
print("Female")
print("------")
print(female1.groupby(['lunch','test preparation course'])['reading score'].max().unstack(level = 'lunch'))
print("Male")
print("------")
print(male1.groupby(['lunch','test preparation course'])['reading score'].max().unstack(level = 'lunch'))


# In[ ]:


print("WRITING SCORE")
print("Female")
print("------")
print(female1.groupby(['lunch','test preparation course'])['writing score'].max().unstack(level = 'lunch'))
print("Male")
print("------")
print(male1.groupby(['lunch','test preparation course'])['writing score'].max().unstack(level = 'lunch'))


# In[ ]:


print("OVERALL SCORE")
print("Female")
print("------")
print(female1.groupby(['lunch','test preparation course'])['overall score'].max().unstack(level = 'lunch'))
print("Male")
print("------")
print(male1.groupby(['lunch','test preparation course'])['overall score'].max().unstack(level = 'lunch'))


# In[ ]:


sns.barplot(x='gender',y='math score',data=data)


# In[ ]:


sns.barplot(x='gender',y='reading score',data=data)


# In[ ]:


sns.barplot(x='gender',y='writing score',data=data)


# In[ ]:


sns.barplot(x='gender',y='overall score',data=data)


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x="overall score", data = data, palette="muted")
plt.show()


# In[ ]:


score_grouped = data.groupby("parental level of education", as_index=True)[["math score", "reading score", "writing score"]].mean().sort_values(by='writing score',ascending=False)
score_grouped.plot.bar(title = "Students Score Average per Parental Level of Education", figsize=(20,10))


# In[ ]:


plt.figure(figsize=(10, 8))
sns.violinplot(x='parental level of education',y='overall score',data=data,kind='bar',hue='gender',palette='Set1')


# In[ ]:


plt.figure(figsize=(10, 8))
sns.violinplot(x='test preparation course',y='overall score',data=data,kind='bar',hue='gender',palette='Set2')


# In[ ]:


data.plot(kind="scatter", x = "reading score",y = "overall score",alpha = 0.5,color = "orange",grid = True,figsize = (10,10))
plt.xlabel("Reading Score",color = "blue")
plt.ylabel("Overall score",color = "r")
plt.title("Overall vs Reading ")
plt.show()


# In[ ]:


data.plot(kind="scatter", x = "writing score",y = "overall score",alpha = 0.5,color = "orange",grid = True,figsize = (10,10))
plt.xlabel("Writing Score",color = "blue")
plt.ylabel("Overall score",color = "r")
plt.title("Overall vs Writing ")
plt.show()


# In[ ]:


data.plot(kind="scatter", x = "math score",y = "overall score",alpha = 0.5,color = "orange",grid = True,figsize = (10,10))
plt.xlabel("Math Score",color = "blue")
plt.ylabel("Overall score",color = "r")
plt.title("Overall vs Math ")
plt.show()


# In[ ]:


race = round(data.groupby(by = data['race/ethnicity']).mean(), 1)
race


# In[ ]:


x = list(race.index)
y = round(race.mean(axis=1),1)

fig, ax = plt.subplots(figsize=(12,6))

rects = ax.bar(x, y)

ax.set_ylabel('Average Scores')
ax.set_title('Average scores by race/ethnicity')
ax.set_xticklabels((x))



plt.show()


# In[ ]:




