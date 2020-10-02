#!/usr/bin/env python
# coding: utf-8

# **EDA on Student scores**

# Attempt to doing better than "Hello World"
# This is my second attempt at analysing data using Python. Apart from data visualisation, I hope to get my hands on linear regression. I shall follow the basic steps of EDA:
# 1. Getting a feel of the raw data
# 2. Cleaning and preparing data for further analysis
# 3. Action with analysis
# 4. Inferences
# 

# In[ ]:


#importing required libraries to read and clean raw data
import pandas as pd
import numpy as np
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()


# In[ ]:


#getting basic info about the data frame
data.info()


# Lets find out if there are any missing values in any of the columns.

# In[ ]:


data.isna().sum() # returns total no. of rows which contain null or missing values


# Our data frame as no null/missing values. This makes my work easier, but missing values would have added some more fun into cleaning/replacing missing values! 
# I shall rename some columns and remove word "group" from the column "Race/Ethinicity" so as to get only one categorical variable (A/B/C)
# 

# In[ ]:


data.columns = ["gender", "race","parent_edu", "lunch", "test_prep", "score_math", "score_reading", "score_writing"] #passing a list of string to rename columns
data.columns #re-checking work!


# In[ ]:


data["race"] = data["race"].str.replace("group", "")
data.head()


# In the above snapshot of our data, "none" value appears in the "test_prep" column. Please note that this is not a missing value. It simply indicates whether the student prepared for the exam or not. 
# Now that we have completed steps 1 & 2, lets dive into the action! Get ready for some beautiful visualisations using seaborn! 
# 
# We need to set a passing grade so as to draw some useful analysis from the dataset. 
# **Pass grade = 40 assuming that maximum marks = 100 for each section i.e math, reading and writing***

# In[ ]:


#introducing categorical variables "P" and "F" to indicate pass/fail status
data["math_status"] = np.where(data["score_math"] >=40, "P", "F")
data["reading_status"]= np.where(data["score_reading"] >=40, "P", "F")
data["writing_status"] = np.where(data["score_writing"] >=40, "P", "F")
data.head(3)


# In[ ]:


#importing required libraries for data visualisation

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# **Categorical Variables**
# The following columns have categorical variables:
# 1. Gender 
# 2. Race 
# 3. Parent_edu
# 4. Lunch
# 5. test_prep
# 

# In[ ]:


plt.rcParams["figure.figsize"] = 8,6
sns.countplot(x = data["gender"]).set_title("Sex ratio of students")
plt.show()


# In[ ]:


total = data["gender"].shape[0]
female = pd.value_counts(data["gender"] == "female")
female_percent = (female/total)*100
male_percent = 1 - female_percent
print(female_percent)

# 51.7% students are female and 48.2% students are male


# 

# In[ ]:


plt.rcParams["figure.figsize"] = 8,6
sns.countplot(x = data["race"]).set_title("Race/ethnicity distribution of students")
plt.show()


# In[ ]:


plt.rcParams["figure.figsize"] = 12,7
sns.countplot(x = data["parent_edu"]).set_title("Parent's education level")
plt.show()


# In[ ]:


plt.rcParams["figure.figsize"] = 8,8
sns.countplot(x = data["gender"], hue = data["test_prep"]).set_title("Preparation of students")
plt.show()


# In[ ]:


plt.rcParams["figure.figsize"] = 8,8
sns.countplot(x = data["gender"], hue = data["lunch"]).set_title("Lunch types")
plt.show()


# **Numerical Variables**
# The following columns have categorical variables:
# 1. Math section
# 2. Reading section
# 3. Writing section
# 
# Point plots represents an estimator of central tendency (mean value in this case) for a numerical variable. 
# More here: https://seaborn.pydata.org/generated/seaborn.pointplot.html

# In[ ]:


fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(15)
fig.set_figwidth(25)

sns.pointplot(y=data['score_math'], x=data["race"], ax=ax1)
sns.pointplot(y=data['score_math'], x=data['gender'], ax=ax2)
sns.pointplot(y=data['score_math'], x=data["parent_edu"],ax=ax3)
sns.pointplot(y=data['score_math'], x=data['gender'], hue= data["test_prep"],ax=ax4)
plt.show()


# **Inferences of Math Section:**
# 1.Students of Race E performed the best!
# 2. Male students performed better than females
# 3. Parent's higher level of education makes a positive impact on students' scores. However, there is **not** a marginal difference in scores of students whose parents have a master's degree over bachelor's degree
# 4. As expected, completing test preparation course has a positive impact on students' performance
# 

# In[ ]:


fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(15)
fig.set_figwidth(25)

sns.pointplot(y=data['score_reading'], x=data["race"], ax=ax1)
sns.pointplot(y=data['score_reading'], x=data['gender'], ax=ax2)
sns.pointplot(y=data['score_reading'], x=data["parent_edu"],ax=ax3)
sns.pointplot(y=data['score_reading'], x=data['gender'], hue= data["test_prep"],ax=ax4)
plt.show()


# **Inferences of Reading Section:**
# 1.Students of Race E performed the best!
# 2. Female students performed better than males
# 3. Parent's higher level of education makes a positive impact on students' scores. 
# 4. As expected, completing test preparation course has a positive impact on students' performance

# In[ ]:


fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(15)
fig.set_figwidth(25)

sns.pointplot(y=data['score_writing'], x=data["race"], ax=ax1)
sns.pointplot(y=data['score_writing'], x=data['gender'], ax=ax2)
sns.pointplot(y=data['score_writing'], x=data["parent_edu"],ax=ax3)
sns.pointplot(y=data['score_writing'], x=data['gender'], hue= data["test_prep"],ax=ax4)
plt.show()


# **Inferences of Writing Section:**
# 1.Students of Race E performed the best!
# 2. Female students performed better than males
# 3. Parent's higher level of education makes a positive impact on students' scores. 
# 4. As expected, completing test preparation course has a positive impact on students' performance

# Let's see how students have qualified for "free/reduced" lunch. Does it depend on their parent's low education leading to low finanical status thus, the children qualifying for a waiver on lunch prices

# In[ ]:


a = pd.value_counts(data["lunch"])
print(a)

#35.5% students qualified for the lunch fees waiver


# In[ ]:


plt.rcParams["figure.figsize"] = 15,10
sns.countplot(x = data["parent_edu"], hue = data["lunch"]).set_title("Parents education levels qualifying for lunch wiaver")
plt.show()


# There is no strong causation effect of low education levels of parents on the lunch waiver. Some information on how students' qualify for lunch waiver would be helpful for analysis. 

# In[ ]:




