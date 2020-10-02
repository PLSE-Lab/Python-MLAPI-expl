#!/usr/bin/env python
# coding: utf-8

# # Introduction

# We all would have crossed our schooling days and most of us love it. But there is something which we hate, the exams! 
# Today, we have got an interesting dataset on students studying in a senior secondary school. 
# It has several attributes of students who are enrolled in mathematics and portuguese classes. 
# This dataset has details on students' family background, academic performance, personal preferences etc. 
# Let us dive deeper into this dataset to find some interesting insights on student behaivour.
# 
# I'm using the data of students who have enrolled in Portuguese course for this analysis. Let's start!

# # Importing Libraries and Dataset

# Let us start our coding by importing the required libraries and the dataset. The Python libraries which we are going to use are,
# * Numpy
# * Pandas
# * Scipy
# * Matplotlib
# * Seaborn

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
sns.set_context('notebook')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/student-alcohol-consumption/student-por.csv')


# # Brief Look at the Dataset

# Let's quickly glance the dataset to understand the nature of the feature it carries.

# In[ ]:


print('No. of students : ', df.shape[0])
print('No. of attributes : ', df.shape[1])


# In[ ]:


df.info()


# The dataset seems to be having no missing values, hence we need not bother about the completeness of the data.

# In[ ]:


df.describe()


# Looking at the descriptive statistics of this dataset, we could infer
# * Average age of students is 16.
# * Avg. weekly study time of students is 2 hours approx.
# * Avg. grade obtained in 1st and 2nd period is 12 approx.
# * Avg. final grade obtained is 12 approx.

# In[ ]:


df.head()


# In[ ]:


df.columns


# A preliminary analysis on the dataset shows that,
# * There are 649 observations and 33 features.
# * The features are of multiple datatypes.
# * The dataset is tidy.
# * Has no missing values.

# Okay, that's enough for now. Let's jump into the next section - Exploratory Data Analysis, which is the most favourite part of data scientits predominantly. It allows us to find interesting patterns in the dataset and also helps us derive intuitive inferences about the data.

# # Exploratory Data Analysis

# Let's kick off our EDA by checking out the number of students who have participated in the survey by the schools where they study.

# ## How many students in each school participated in the survey?

# In[ ]:


ax = sns.catplot(x = 'school', data = df, kind = 'count',hue = 'sex', palette = 'husl')
plt.title('Student Distribution in School')
plt.xlabel('School Name')
plt.ylabel('# Students')
ax.set(xticklabels = ["Gabriel Pereira", "Mousinho da Silveira"])
plt.show()


# * More no. of students who have participated in this survey are from Gabriel Pereira School.
# * Female students are higher in number than that of males.

# ## Are the students are of different age groups?

# In[ ]:


fig, ax = plt.subplots()
ax.hist(df.loc[(df['sex'] == 'F'), 'age'], color = 'k', histtype = 'step', label = 'Female')
ax.hist(df.loc[(df['sex'] == 'M'), 'age'], color = 'r', histtype = 'step', label = 'Male')
plt.title('Student Age by Gender')
plt.xlabel('Age')
plt.ylabel('# Students')
plt.legend()
plt.show()


# Most students are aged between 15 and 19.

# ## Do female students score higher than male students?

# In[ ]:


fig, ax = plt.subplots()
ax.hist(df.loc[(df['sex'] == 'F'), 'G3'], color = 'r', histtype = 'step', label = 'Female')
ax.hist(df.loc[(df['sex'] == 'M'), 'G3'], color = 'b', histtype = 'step', label = 'Male')
plt.title('Student Grade by Gender')
plt.xlabel('Grade')
plt.ylabel('# Students')
plt.legend()
plt.show()


# Most of the female students have secured a good grade comparing to the male students. Let's check whether the average grade secured by female students is higher than than of male students.

# In[ ]:


ax = df.groupby('sex')['G1', 'G2', 'G3'].mean().plot(kind = 'bar')
plt.title('Mean Score by Gender')
plt.xlabel('Gender')
plt.ylabel('Avg Grade')
plt.legend(loc = 'upper right')
ax.set_xticklabels(['Female', 'Male'], rotation = 360)
plt.show()


# As expected from the inference of previous graph, the mean score of female students are higher than that of male students.

# ## Does the education and job status of parents affect their child's grade?

# In[ ]:


ax = sns.FacetGrid(df,  col = 'Medu', hue = 'sex').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Student Grade Analysis by Mother's Education")
plt.subplots_adjust(top = 0.7)
plt.show()


# In[ ]:


df.groupby('Medu')['G3'].mean()


# * There is no direct relationship observed between the mother's education status and student's grade. 
# * As we could see from the graph, students of least educated mothers have also secured good grades whereas students of highly educated mothers have secured less grades.
# * Considering the average grade, students of highly educated mothers have scored high.

# In[ ]:


ax = sns.FacetGrid(df, col = 'Fedu', hue = 'sex').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Student Grade Analysis by Father's Education")
plt.subplots_adjust(top = 0.7)
plt.show()


# In[ ]:


df.groupby('Fedu')['G3'].mean()


# Here the trend is more or less same as that of the previous one.

# In[ ]:


ax = sns.FacetGrid(df, col = 'Mjob', hue = 'sex').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Student Grade Analysis by Mother's Occupation")
plt.subplots_adjust(top = 0.7)
plt.show()


# In[ ]:


df.groupby('Mjob')['G3'].mean()


# In[ ]:


ax = sns.FacetGrid(df, col = 'Fjob', hue = 'sex').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Student Grade Analysis by Father's Occupation")
plt.subplots_adjust(top = 0.7)
plt.show()


# In[ ]:


df.groupby('Fjob')['G3'].mean()


# * As we could see from the graph, students whose parents are teachers have secured a good grade at an average.

# ## Do students who travel more tend to study less?

# Let's check out whether the study time of students has an impact on their grades.

# In[ ]:


ax = sns.FacetGrid(df, col = 'studytime', row = 'traveltime', hue = 'studytime').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Student Grade Analysis by Study Time & Travel Time")
plt.subplots_adjust(top = 0.9)
plt.show()


# In[ ]:


ax = sns.lmplot(x = 'studytime',y = 'G3', hue = 'sex', data = df, palette = 'Set1')
ax.fig.suptitle('Correlation b/w Study Time and Grade')
plt.subplots_adjust(top = 0.9)


# This seems to be interesting. Let me jot down the inferences derived out of this graph.
# * Students who travel more than 2 hours is less in number.
# * Students who travel for more than 2 hours are spending less time to study.
# * Students who travel less than 2 hours have mixed preferences in their study pattern. There are few students who use this time to study for more hours, but most of them study for less than 2 hours.
# * As expected, study time is influencing the grade. Students whose study time is more were able to secure a good grade.

# ## Do students require additional educational support to secure good grade?

# In[ ]:


sns.catplot(x = 'schoolsup',y = 'G3', hue = 'sex', data = df, kind = 'bar', ci = None)
plt.title('Effect of Extra Educational Support on Grade')
plt.ylabel('Avg. Grade')
plt.show()


# In[ ]:


sns.catplot(x = 'famsup',y = 'G3', hue = 'sex', data = df, kind = 'bar', ci = None)
plt.title('Effect of Family Educational Support on Grade')
plt.ylabel('Avg. Grade')
plt.show()


# In[ ]:


sns.catplot(x = 'paid',y = 'G3', hue = 'sex', data = df, kind = 'bar', ci = None)
plt.title('Effect of Tuitions on Grade')
plt.ylabel('Avg. Grade')
plt.show()


# Looking at all the above trends, we could not see a pattern in any of those. 
# * Avg grade secured by students who have no extra educational support is higher than those who have.
# * Avg grade secured by students is almost the same between who have family educational support and those who have not.
# * Avg grade secured by students who attended tuition classes is lesser than those who have not.

# ## How the involvement in extra activities influence student's grade?

# In[ ]:


sns.catplot(x = 'activities', y = 'G3', kind = 'bar', data = df)
plt.title('Effect of Extra Activities on Grade')
plt.ylabel('Avg. Grade')
plt.show()


# Average grade secured by Students who do extra activities is higher than students who do not.

# ## Are students with internet connection scoring good grade?

# In[ ]:


sns.catplot(x = 'internet', y = 'G3', kind = 'bar', data = df)
plt.title('Effect of Internet on Grade')
plt.ylabel('Avg. Grade')
plt.show()


# As expected, average grade of students who have internet connection at their homes is higher than that of students who do not access internet.

# ## Does personal relationships influence student's grade?

# In[ ]:


sns.catplot(x = 'romantic', y = 'G3', kind = 'bar', data = df)
plt.title('Effect of Romantic Relationships on Grade')
plt.ylabel('Avg. Grade')
plt.show()


# Students who have personal relationship have average grade which is bit lesser than others.

# ## Do the students prefer hanging out during their freetime?

# In[ ]:


ax = sns.FacetGrid(df, col = 'goout', row = 'freetime', hue = 'sex').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Effects of Personal Preferences on Grades")
plt.subplots_adjust(top = 0.95)
plt.show()


# In[ ]:


ax = sns.lmplot(x = 'goout',y = 'freetime', hue = 'sex', data = df, palette = 'Set1')
ax.fig.suptitle('Correlation b/w Hanging out and Leisure Time')
plt.subplots_adjust(top = 0.9)


# * Students who have more free time tend to hang out for more hours with their companions.
# * But this does not seem to be influencing their grades.

# ## Is alcohol consumption influencing student's grade?

# In[ ]:


ax = sns.FacetGrid(df, col = 'Dalc', row = 'goout', hue = 'Dalc').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Effects of Personal Preferences on Grades")
plt.subplots_adjust(top = 0.95)
plt.show()


# In[ ]:


ax = sns.lmplot(x = 'goout',y = 'Dalc', hue = 'sex', data = df, palette = 'Set1')
ax.fig.suptitle('Correlation b/w Hanging out and Alcohol Consumption')
plt.subplots_adjust(top = 0.9)


# * Correlation between hanging out and consuming alcohol is different for each of the genders.
# * Male Students who hang out more frequently consume more alcohol whereas it is not the same for female students.
# * But this is not affecting their grades.

# In[ ]:


ax = sns.FacetGrid(df, col = 'Walc', row = 'goout', hue = 'Walc').map(plt.hist, 'G3').add_legend()
ax.fig.suptitle("Effects of Personal Preferences on Grades")
plt.subplots_adjust(top = 0.95)
plt.show()


# In[ ]:


ax = sns.lmplot(x = 'goout',y = 'Walc', hue = 'sex', data = df, palette = 'Set1')
ax.fig.suptitle('Correlation b/w Hanging out and Alcohol Consumption')
plt.subplots_adjust(top = 0.9)


# * The alcohol consumption pattern is quite different in weekends.
# * Students who go out more frequently during weekends tend to consume more alcohol.
# * However it is not affecting their grades at a larger scale.

# ## Do absentees fail more?

# In[ ]:


ax = sns.lmplot(x = 'failures',y = 'absences', hue = 'sex', data = df, palette = 'Set1')
ax.fig.suptitle('Correlation b/w Absences and Failures in Subjects')
plt.subplots_adjust(top = 0.9)


# There is no much linear correlation seen between the absenteeism and failure in exams.

# ## Quick Correlation Check

# In[ ]:


fig, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(df.corr(), annot = True, cmap = 'Blues', linewidths = .5)


# # Conclusion

# * Some interesting insights are derived from this dataset which are summarized as below.
# * As age of the students increases, the failure also raises. This may be because the adolescents tend to hang out more and involve in activities which could distract them from studies.
# * Avg grade of students who have highly educated parents is higher than others.
# * Students who study for more hours tend to score high.
# * Failure rate impacts the student's grade.
# * Students who have more free time tend to hang out much.
# * Students who hang out more frequently consume alcohol in weekends. However this is not influencing their grades.

# In[ ]:




