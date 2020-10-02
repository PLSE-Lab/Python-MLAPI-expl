#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
import seaborn as sns


plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# In[ ]:


df = pd.read_csv("../input/multipleChoiceResponses.csv", encoding="ISO-8859-1")
df.head()


# > # Exploratory Analysis
# 
# 

# # 1. Gender

# ## Missing Values

# In[ ]:


len(df[pd.isnull(df.GenderSelect)])


# ## Distribution of Respondents by Gender across all countries
# 
# The proportion of males is markedly higher than that of females while the representation of other genders is very miniscule.

# In[ ]:


plot = df[df.GenderSelect.isnull() == False].groupby(df.GenderSelect).GenderSelect.count().plot.bar()
plot = plt.title("Number of Respondents by Gender")


# ## Which Country has the highest ratio of Female/Male Respondents

# In[ ]:


filtered_df = df[(df.GenderSelect.isnull() == False) & (df.Country.isnull() == False)]


# In[ ]:


def getFemaleMaleRatio(df):
    counts_by_gender = df.groupby('GenderSelect').GenderSelect.count()
    return counts_by_gender[0]/counts_by_gender[1]


# In[ ]:


group_by_country = filtered_df.groupby(df.Country)
ratios = group_by_country.apply(getFemaleMaleRatio)
print("Maximum Female/Male Ratio: ", ratios.idxmax(), ratios.max())
print("Minimum Female/Male Ratio: ", ratios.idxmin(), ratios.min())


# ##  Distribution of Ages of Males and Females
# The shape of the distributions of ages for both male and female are very similar although the size differs markedly. There also seem to be no women datscientists in the 60+ age bracket, while there are quite a few men in that bracket

# In[ ]:


fig, ax = plt.subplots()
df[df.GenderSelect == 'Male'].Age.plot.hist(bins=100, ax=ax, alpha=0.5)
df[df.GenderSelect == 'Female'].Age.plot.hist(bins=100, ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age distribution for Male and Female Data Scientists")


# ## Distribution of Ages of Men and Women above 60
# It might be interesting to see why there are so few women respondents above the age of 60

# In[ ]:


fig, ax = plt.subplots()
df[(df.GenderSelect == 'Male') & (df.Age > 60)].Age.plot.hist(ax=ax, alpha=0.5)
df[(df.GenderSelect == 'Female') & (df.Age > 60)].Age.plot.hist(ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age Distribution for Male and Female Data Scientists above 60 years of age")


# ## Distribution of Ages of Male & Female Students
# - Again, the shapes of the distributions are similar for males and females
# - However, there are no female students above the age of 38, although there are male students as old as 50 years of age 

# In[ ]:


fig, ax = plt.subplots()
df[(df.GenderSelect == 'Male') & (df.StudentStatus == 'Yes')].Age.plot.hist(bins=30, ax=ax, alpha=0.5)
df[(df.GenderSelect == 'Female') & (df.StudentStatus == 'Yes')].Age.plot.hist(bins=30, ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age Distribution for Male and Female Student Respondents")


# ## Distribution of Ages of Males and Females who are not Students but are still learning data science

# In[ ]:


fig, ax = plt.subplots()
isNotStudentAndLearning = ((df.StudentStatus == 'No') & ((df.LearningDataScience == "Yes, I'm focused on learning mostly data science skills") | 
                                                         (df.LearningDataScience == "Yes, but data science is a small part of what I'm focused on learning")))
df[(df.GenderSelect == 'Male') & isNotStudentAndLearning].Age.plot.hist(ax=ax, alpha=0.5)
df[(df.GenderSelect == 'Female') & isNotStudentAndLearning].Age.plot.hist(ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age Distribution for Male and Female Student Respondents")


# ## Distribution of the ages of the men and women who code
# - The fact that they code or not does not seem to have much an effect on the shape of the distribution as compared to the original distribution of ages of males and females

# In[ ]:


fig, ax = plt.subplots()
df[(df.GenderSelect == 'Male') & (df.CodeWriter == 'Yes')].Age.plot.hist(bins=50, ax=ax, alpha=0.5)
df[(df.GenderSelect == 'Female') & (df.CodeWriter == 'Yes')].Age.plot.hist(bins=50, ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age Distribution for Male and Female Coders")


# In[ ]:


fig, ax = plt.subplots()
df[(df.GenderSelect == 'Male') & (df.CodeWriter == 'No')].Age.plot.hist(bins=30, ax=ax, alpha=0.5)
df[(df.GenderSelect == 'Female') & (df.CodeWriter == 'No')].Age.plot.hist(bins=30, ax=ax, alpha=0.8)
legend = ax.legend(['Male', 'Female'])
plot = plt.title("Age Distribution for Male and Female Non-Coders")


# ## Relationship between Employment Status and Gender
# - It may not make sense to look at the absolute numbers of people here as we already know that the number of male respondents is much higher than the number of other gender respondents. 

# In[ ]:


counts_by_gender = df.groupby([df.GenderSelect, df.EmploymentStatus]).size().reset_index(name="Total")


# In[ ]:


n_male = len(df[df.GenderSelect == "Male"])
n_female = len(df[df.GenderSelect == "Female"])
n_diff_identity = len(df[df.GenderSelect == "A different identity"])
n_other = len(df[df.GenderSelect == "Non-binary, genderqueer, or gender non-conforming"])
print(n_male, n_female, n_diff_identity, n_other)


# In[ ]:


counts_by_gender_plot = counts_by_gender.pivot("GenderSelect", "EmploymentStatus", "Total")
ax = sns.heatmap(counts_by_gender_plot, linewidths=.5, cmap="Blues")
plot = plt.title("Heatmap of Absolute number of people across Gender & Employment Status")


# ## Relative number of People (within their gender group) across Different Employment Statuses
# - Note: The proportions here are calculated across each gender(each row)
# - It seems like the major proportion of people(looks like around 60%) across all genders are Employed full-time. 
# - Their is a *slightly* higher proportion of Women who are in the "Currently Employed and Looking for Work" bracket as compared to other Gender groups
# - Reference: https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby

# In[ ]:


relative_counts = df.groupby([df.GenderSelect, df.EmploymentStatus]).size().groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum())).reset_index(name="percentage")


# In[ ]:


relative_counts_by_gender_plot = relative_counts.pivot("GenderSelect", "EmploymentStatus", "percentage")
ax = sns.heatmap(relative_counts_by_gender_plot, linewidths=.5, cmap="Blues")
plot = plt.title("Heatmap of Relative number of people across Gender who are in each Employment Category")


# ## What Kind of Jobs are most common amongst men and women data scientists?

# In[ ]:


jobs_by_gender = df[["GenderSelect", "CurrentJobTitleSelect"]].groupby([df.CurrentJobTitleSelect, df.GenderSelect]).size().reset_index(name="number")


# In[ ]:


from matplotlib import pyplot
sns.plotting_context(None)

chart = sns.factorplot(x='CurrentJobTitleSelect', y='number', hue='GenderSelect', data=jobs_by_gender, kind='bar', size=15, aspect=2, legend=False)
for ax in plt.gcf().axes:
    ax.set_xlabel("Job Title", fontsize=35)
    ax.set_ylabel("Count", fontsize=35)

for ax in chart.axes.flatten(): 
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=25) 
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0, fontsize=25) 

plt.legend(loc='upper left')


# In[ ]:




