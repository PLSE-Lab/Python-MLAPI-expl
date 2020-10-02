#!/usr/bin/env python
# coding: utf-8

# ## EXPLORATORY DATA ANALYSIS
# by [Iqrar Agalosi Nureyza](https://www.kaggle.com/iqrar99)

# Hello Everyone! I'm a student and I try my best to do data analysis. This Student Performance Dataset is a very good dataset to sharper your analysis skill. I hope you can understand my analysis. I am very open in accepting criticism and suggestions for perfecting this kernel.
# 
# If you find this notebook useful, feel free to **Upvote**.

# **Table of Contents**
# 1. [Basic Analysis](#1)
#     * [Frequency](#2)
#     * [Male vs Female](#MF)
#     * [Top 30 Students](#T)
# 2. [Data Visualisation](#3)
#     * [Count Plot](#4)
#     * [Swarm Plot](#5)
#     * [Scatter Plot and Hexbin Plot](#6)
#     * [Pie Plot](#7)
#     * [Histogram](#8)
#     * [Heatmap](#9)

# In[ ]:


#importing all important packages
import numpy as np #linear algebra
import pandas as pd #data processing
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns #data visualisation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#input data
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data.head(10)


# In[ ]:


data.info() #checking data type for each column


# In[ ]:


data.shape


# Our data has 8 columns and 1000 rows. Let's do some basic analysis.

# <a id ="1"></a>
# ### Basic Analysis

# In[ ]:


data.describe() #starter code to find out some basic statistical insights


# Now, Let's check for any missing values

# In[ ]:


print(data.isna().sum())


# There are no missing values in our data. So, we don't need to drop any values.

# <a id = "2"></a>
# #### Frequency
# First, we look at the student's gender.

# In[ ]:


data['gender'].value_counts()


# more female than male in this data.

# In[ ]:


data.iloc[:,1].value_counts()


# As we can see, group C has the most members compared to the others

# In[ ]:


data.iloc[:,2].value_counts()


# In[ ]:


data.iloc[:,3].value_counts()


# Many students have standard lunch.

# In[ ]:


data.iloc[:,4].value_counts()


# It seems that the total number of people who did not complete the course was double that of those who completed the course

# <a id = "MF"></a>
# #### Male vs Female
# We will see who is better in math, reading, and writing.

# In[ ]:


male = data[data['gender'] == 'male']
female = data[data['gender'] != 'male']

print("Math Score")
print("Male    :",round(male['math score'].sum()/len(male),3))
print("Female  :",round(female['math score'].sum()/len(female),3),'\n')

print("Reading Score")
print("Male    :",round(male['reading score'].sum()/len(male),3))
print("Female  :",round(female['reading score'].sum()/len(female),3),'\n')

print("Writing Score")
print("Male    :",round(male['writing score'].sum()/len(male),3))
print("Female  :",round(female['writing score'].sum()/len(female),3))


# Male students are better in math. Female students are good in writing and reading.

# <a id = "T"></a>
# #### Top 30 Students
# let's look at all the top 30 students who get very high scores

# In[ ]:


scores = pd.DataFrame(data['math score'] + data['reading score'] + data['writing score'], columns = ["total score"])
scores = pd.merge(data,scores, left_index = True, right_index = True).sort_values(by=['total score'],ascending=False)
scores.head(30)


# It seems that there are 3 students who are geniuses here, they get perfect scores for all subjects. But, 2 of them didn't complete their test preparation course. Only 2 possibilities: **Genius** or **Cheating**.
# 
# From the data, we have seen that students with standard luch have better score than free/reduce lunch.

# ____________________________

# <a id = "3"></a>
# ### Data Visualisation

# And now we move to the important part where we will get informations from visualizing our data. First, we make count plots 

# <a id = "4"></a>
# #### Count Plot

# In[ ]:


sns.set(style="darkgrid")
f, axs = plt.subplots(1,2, figsize = (20,8))

sns.countplot(x = 'race/ethnicity', data = data, ax = axs[0]) #race / ethnicity
sns.countplot(x = 'parental level of education', data = data, ax = axs[1]) #parental level of education

plt.xticks(rotation=90)

plt.show()


# with count plot, we know the number of values for each element in our data. Let's see the data above by dividing it by gender.

# In[ ]:


f, ax = plt.subplots(1,1, figsize = (15,5))

sns.countplot(x = 'race/ethnicity', data = data, hue = 'gender', palette = 'Set1') #race / ethnicity

plt.show()


# What can we get from the information above? 3 out of 5 groups, have the most male. And we get some insights here:
# * Group C has more differences in the number of male and female than the others
# * The group with the least number of male is group A, and so are the female 
# * The group with the most number of female is group C, and so are the male
# <br>
# <br>
# <br>
# How about their lunch?

# In[ ]:


f, axs = plt.subplots(2,1, figsize = (15,10))

sns.countplot(x = 'parental level of education', data = data, hue = 'lunch', ax = axs[0], palette = 'spring')
sns.countplot(x = 'race/ethnicity', data = data, hue = 'lunch', ax = axs[1] , palette = 'spring')

plt.show()


# standard lunches are the majority for each level of education, and so are groups. 

# <a id = "5"></a>
# #### Swarm Plot
# We'll use Swarm plot in our ratio data: Math score, writing score, and reading score. It draw a categorical scatterplot with non-overlapping points.. 

# In[ ]:


sns.set_style('whitegrid')

f, ax = plt.subplots(1,1,figsize= (15,5))
ax = sns.swarmplot(x = 'math score', y='gender', data = data, palette = 'Set1') #math score
plt.show()

f, ax = plt.subplots(1,1,figsize= (15,5))
ax = sns.swarmplot(x = 'reading score', y='gender', data = data, palette = 'Set1') #reading score
plt.show()

f, ax = plt.subplots(1,1,figsize= (15,5))
ax = sns.swarmplot(x = 'writing score', y='gender', data = data, palette = 'Set1') #writing score
plt.show()


# As we can see, we get some information:
# * The majority of students have grades in the range 60-80
# * In all subjects, the lowest value is held by female
# * Female who get perfect score in writing and reading more than male

# <a id = "6"></a>
# #### Scatter Plot and Hexbin Plot
# The most familiar way to visualize a bivariate distribution is a scatterplot, where each observation is shown with point at the x and y values. We can draw a scatterplot with the matplotlib *plt.scatter* function, and it is also the default kind of plot shown by the **jointplot()** function.
# 
# Hexbin plot shows the counts of observations that fall within hexagonal bins. This plot works best with relatively large datasets.
# 
# First, we enter math score as x-axis and writing score as y-axis.

# In[ ]:


sns.jointplot(x ='math score', y = 'writing score', data = data, color = 'green', height = 8, kind = 'reg')
sns.jointplot(x ='math score', y = 'writing score', data = data, color = 'green', height = 8, kind = 'hex')


# In[ ]:


sns.jointplot(x ='math score', y = 'reading score', data = data, color = 'blue', height = 8, kind = 'reg')
sns.jointplot(x ='math score', y = 'reading score', data = data, color = 'blue', height = 8, kind = 'hex')


# In[ ]:


sns.jointplot(x ='reading score', y = 'writing score', data = data, color = 'red', height = 8, kind = 'reg')
sns.jointplot(x ='reading score', y = 'writing score', data = data, color = 'red', height = 8, kind = 'hex')


# It seems like all students have a balanced skill between writing and reading.

# <a id = "7"></a>
# #### Pie Plot
# Let's assume if there are students who score < 60, then we determine they did not pass the exam.

# In[ ]:


#math score
passed = len(data[data['math score'] >= 60])
not_passed = 1000 - passed

percentage1 = [passed, not_passed]

#reading score
passed = len(data[data['reading score'] >= 60])
not_passed = 1000 - passed

percentage2 = [passed, not_passed]

#writing score
passed = len(data[data['writing score'] >= 60])
not_passed = 1000 - passed

percentage3 = [passed, not_passed]


# In[ ]:


labels = "Passed", "Not Passed"

f, axs = plt.subplots(2,2, figsize=(15,10))

#Math Score
axs[0,0].pie(percentage1, labels = labels, explode=(0.05,0.05), autopct = '%1.1f%%', startangle = 200, colors = ["#e056fd", "#badc58"])
axs[0,0].set_title("Math Score", size = 20)
axs[0,0].axis('equal')

#Reading Score
axs[0,1].pie(percentage2, labels = labels, explode=(0.05,0.05), autopct = '%1.1f%%', startangle = 200, colors = ["#e056fd", "#badc58"])
axs[0,1].set_title("Reading Score", size = 20)
axs[0,1].axis('equal')

#Writing Score
axs[1,0].pie(percentage3, labels = labels, explode=(0.05,0.05), autopct = '%1.1f%%', startangle = 200, colors = ["#e056fd", "#badc58"])
axs[1,0].set_title("Writing Score", size = 20)
axs[1,0].axis('equal')

f.delaxes(axs[1,1]) #deleting axs[1,1] so it will be white blank

plt.show()


# We get some information:
# * 323 students didn't pass Math exam
# * 254 students didn't pass Reading exam
# * 281 students didn't pass Writing exam 

# <a id = "8"></a>
# #### Histogram
# A histogram represents the distribution of data by forming bins along the range of the data and then drawing bars to show the number of observations that fall in each bin.

# In[ ]:


f, ax = plt.subplots(1,1,figsize=(10,5))

plt.hist(data['math score'], 30, color = 'skyblue')
ax.set(title = "Math Score")

plt.show()


# In[ ]:


f, ax = plt.subplots(1,1,figsize=(10,5))

plt.hist(data['reading score'], 30, color = 'salmon')
ax.set(title = "Reading Score")

plt.show()


# In[ ]:


f, ax = plt.subplots(1,1,figsize=(10,5))

plt.hist(data['writing score'], 30, color = 'goldenrod') #visit https://xkcd.com/color/rgb/ if you want to see more colors
ax.set(title = "Writing Score")

plt.show()


# From these plots, It seems students performance in exam is around 60-80. 

# <a id = "9"></a>
# #### Heat Map
# Heat map is useful if you want to see any correlations between attributes. We'll see correlations between math, reading, and writing score.

# In[ ]:


sns.heatmap(data.corr(), annot= True, cmap = 'autumn') #data.corr() used to make correlation matrix

plt.show()


# As you can see, writing score and reading score have a strong correlation. This means that if students have good reading skill, it'll affect their writing skill. The lowest correlation is writing score and math score. This pretty makes sense because if you're good at math, it doesn't mean you're good at writing.

# __________________

# Ok, that's all my analysis. If you think I missed something, feel free to comment. Thank you for reading this notebook!
