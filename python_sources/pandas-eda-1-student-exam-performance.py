#!/usr/bin/env python
# coding: utf-8

# # Pandas - EDA #1 - Student Exam Performance

# ## Hamzah Khouli

# ### This is my first attempt at writing a kernel. It is also my first attempt at using Pandas, Seaborn, and matplotlib.
#   
# ### In this kernel, I will be looking at a dataset that documents students' performance in an exam, an exam structured much like the SAT. I will be analyzing this dataset in attempt to find a driving factor behind a student's test score. 
#   
# ### My work may not be perfect, so please do not hesitate to reach out to me if you have any questions/suggestions/comments!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


stuperf = pd.read_csv('../input/StudentsPerformance.csv')
print(stuperf.shape)
print(stuperf.dtypes)


# Now that we have a basic understanding of our dataset, lets visualize it. 

# In[ ]:


stuperf.head(10)


# Looking at the second and fifth columns, it may be useful to change the column names. This way calling these columns will be easier later on. We will also replace the values in the test_preparation_course column to the following: 
# * Old Value      New Value
# *   'none'                  no
# *   'completed'         yes

# In[ ]:


stuperf = stuperf.rename(columns={'race/ethnicity':'race', 'test preparation course' : 'tpc'})
stuperf.tpc.replace('none', 'no', inplace=True)
stuperf.tpc.replace('completed', 'yes', inplace=True)

stuperf.head(5)


# Finally, let us check for any null values in the dataset.

# In[ ]:


stuperf.isnull().sum()


# This shows us that we are ready to go!

# Now that we have the data the way we'd like it, lets look at some basic descriptive statistics. 

# In[ ]:


stuperf.describe()


# How about some statistics on our sample? Lets look at the gender, and race distribution of our sample.

# In[ ]:


print("Gender Distribution")
print(stuperf.gender.value_counts(normalize=True))

print('Race Distribution')
print(stuperf.race.value_counts(normalize=True))


#  Lets visualize this a little better

# In[ ]:


stuperf.race.value_counts().plot.bar()


# In[ ]:


stuperf.groupby('gender').race.value_counts().plot.bar()


# Lets sum up each individual person's scores, to obtain an overall score.

# In[ ]:


stuperf['overall score'] = pd.Series(stuperf['math score'] + stuperf['reading score'] + stuperf['writing score'], index=stuperf.index)
stuperf.head()


# So, whats the distribution of our overall scores?

# In[ ]:


stuperf.boxplot(column='overall score', grid=False)


# Are there any statistically significant correlations between these scores?

# In[ ]:


corrs = stuperf.corr()
fig, ax = plt.subplots(figsize=(8,8))
dropself = np.zeros_like(corrs)
dropself[np.triu_indices_from(dropself)] = True
sns.heatmap(corrs, cmap='PuRd', annot=True, fmt='.2f', mask=dropself)
plt.xticks(range(len(corrs.columns)), corrs.columns)
plt.yticks(range(len(corrs.columns)), corrs.columns)
plt.show()


# As you would imagine, there is a stronger correlation between reading and writing scores, than when compared to math scores
#   
# Since we do have quite a strong correlation between reading and writing scores, lets see if we can fit a line to this data. Lets first look at these scores as a scatter plot.

# In[ ]:


plt.plot(stuperf['writing score'], stuperf['reading score'], 'b.')
plt.xlabel('Writing Score')
plt.ylabel('Reading Score')
plt.grid(False)
plt.show()


# Lets fit a line to this

# In[ ]:


wscores = stuperf['writing score']
rscores = stuperf['reading score']
xvals = np.linspace(0,100,100)
coeffs = np.polyfit(wscores, rscores, 1)
poly = np.poly1d(coeffs)
plt.plot(xvals, poly(xvals), 'r-')
plt.xlabel('Writing Score')
plt.ylabel('Reading Score')
plt.title('Fitted line of writing score and corresponding expected reading score')
plt.show()


# Suppose we had students who were only able to complete the writing segment of the exam, the above line and function would be useful to find an appropriate reading score for the student. 
#   
# Alternatively, this model could be used for predictive purposes.

# What if we break these students down into their grade bracket?
# i.e.
# *  A - 90%+
# *  B - 80%+
# *  C - 70%+
# *  D - 60%+
# *  F - Else

# In[ ]:


a_stu = stuperf.loc[stuperf['overall score']/300 >= 0.9]
b_stu = stuperf.loc[(stuperf['overall score']/300 >= 0.8) & (stuperf['overall score']/300 < 0.9) ]
c_stu = stuperf.loc[(stuperf['overall score']/300 >= 0.7) & (stuperf['overall score']/300 < 0.8) ]
d_stu = stuperf.loc[(stuperf['overall score']/300 >= 0.6) & (stuperf['overall score']/300 < 0.7) ]
f_stu = stuperf.loc[stuperf['overall score']/300 < 0.6 ]


# In[ ]:


fig, axarr = plt.subplots(3,2, figsize=(15,10))
fig.tight_layout()
a_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[0][0], fontsize=7, grid=False)
b_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[0][1], fontsize=7, grid=False)
c_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[1][0], fontsize=7, grid=False)
d_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[1][1], fontsize=7, grid=False)
f_stu.boxplot(by='parental level of education', column='overall score', ax=axarr[2][0], fontsize=7, grid=False)


# Looking at these boxplots, it preliminarily seems that parental level of education has no link to the success of a student.  

# In[ ]:


fig, axarr = plt.subplots(3,2, figsize=(20,10))
fig.tight_layout()
sns.violinplot(x='parental level of education', y='overall score',data=a_stu.sort_values(by='parental level of education'), ax=axarr[0][0])
sns.violinplot(x='parental level of education', y='overall score',data=b_stu.sort_values(by='parental level of education'), ax=axarr[0][1])
sns.violinplot(x='parental level of education', y='overall score',data=c_stu.sort_values(by='parental level of education'), ax=axarr[1][0])
sns.violinplot(x='parental level of education', y='overall score',data=d_stu.sort_values(by='parental level of education'), ax=axarr[1][1])
sns.violinplot(x='parental level of education', y='overall score',data=f_stu.sort_values(by='parental level of education'), ax=axarr[2][0])


# It seems as if parental levels of education have no determinant on how well students do when grouped into their grade brackets. How about overall?

# In[ ]:


sns.violinplot(x='parental level of education', y='overall score', data=stuperf.sort_values(by='parental level of education'), figsize=(15,10), rotation=90)


# There are not many inferences that can be made from these plots, but consider the following; is there a reason that the children of those with *masters degrees* had less variance in their scores than those of parents with only *some high school* education?
#   
# First, lets make sure that the above statement holds true.

# In[ ]:


stuperf.groupby('parental level of education')['overall score'].std()


# By definition, a lower standard deviation implies a lower variance. 
#   
# From this we can see that there is in fact a higher standard deviation in test scores between students with parents having *masters degrees* and those with parents only having *some high school* education. However, inferences may be easier to make if we also take a look at the average test scores by parental level of education.

# In[ ]:


pled_ind = stuperf.sort_values(by='parental level of education')
pled = pd.DataFrame({'Mean Score':list(stuperf.groupby('parental level of education')['overall score'].mean()),
                    'Standard Deviation': list(stuperf.groupby('parental level of education')['overall score'].std())},
                   index=pled_ind['parental level of education'].unique())
pled


# We can see that **_generally_**, higher levels of parental education leads to a higher average test score for their children. Except when looking at the difference in test scores between *some high school* parental education and *high* school parental education. 
#   
# This may be linked to nothing, or it may give us a better insight on what indicators are present for higher test scores.
#   
# Lets isolate the data of those whose parents had either *some high school* or *high school* education.

# In[ ]:


hs_ed = stuperf.loc[(stuperf['parental level of education'] == 'some high school') |
                    (stuperf['parental level of education'] == 'high school')]
hs_ed.head()


# My next "educated" guess would be that that the completion (or lack) of a test preparation course play some role in a student's test score.

# In[ ]:


hs_ed.groupby(['tpc'])['parental level of education'].value_counts().plot.bar(stacked=True)
hs_ed.groupby(['parental level of education']).tpc.value_counts()


# We may be onto something...
#   
# The data above shows that those students with parental education levels of only *some high school* were more likely to complete a test preparation course than those whose parental education levels were *high school*.
# 
# What if we looked at the test scores based on completion of a test preparation course or not? Below, we isolate the students who have completed and those who have not completed a test prep course.

# In[ ]:


prep = stuperf.loc[stuperf.tpc == 'yes']
noprep = stuperf.loc[stuperf.tpc == 'no']
print('Scores w/ completion of tpc')
print(prep['overall score'].describe())
print('\nScores w/out completion of tpc')
print(noprep['overall score'].describe())


# Clearly, those who completed a **tpc** got higher scores on their exams. Completion of a **tpc** seems to be the only effective way to increase a students grade. Additionally, we may conclude that **parental level of education** had so significant effect on how well a student does in his exam. 
