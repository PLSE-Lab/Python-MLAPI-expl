#!/usr/bin/env python
# coding: utf-8

# # Students performance in exams
# #### Marks secured by the students in college
# 
# ## Aim
# #### To understand the influence of various factors like economic, personal and social on the students performance 
# 
# ## Inferences would be : 
# #### 1. How to imporve the students performance in each test ?
# #### 2. What are the major factors influencing the test scores ?
# #### 3. Effectiveness of test preparation course?
# #### 4. Other inferences 

# 

# #### Import the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Let us initialize the required values ( we will use them later in the program )
# #### we will set the minimum marks to 35 to pass in a exam

# In[ ]:


passmark = 35


# #### Let us read the data from the csv file

# In[ ]:


dataset = pd.read_csv("../input/StudentsPerformance.csv")


# #### We will print top few rows to understand about the various data columns

# In[ ]:


dataset.head()


# #### Size of data frame

# In[ ]:


print (dataset.shape, dataset.size, len(dataset))


# #### Let us understand about the basic information of the data, like min, max, mean and standard deviation etc.

# In[ ]:


dataset.describe()


# #### Let us check for any missing values

# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.info()


# #####  All above values are showna s not-null 

# ####  Let us explore the Math Score first

# In[ ]:


dir(sns)


# In[ ]:


plt.figure(figsize=(25,16))
p = sns.countplot(x="math score", data = dataset, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# #### How many students passed in Math exam ?

# In[ ]:


dataset['Math_PassStatus'] = np.where(dataset['math score']<passmark, 'Fail', 'Pass')
dataset.Math_PassStatus.value_counts()


# In[ ]:


plt.figure(figsize=(20,10))
p = sns.countplot(x='parental level of education', data = dataset, hue='Math_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# #### Let us explore the Reading score

# In[ ]:


plt.figure(figsize=(25,16))
sns.countplot(x="reading score", data = dataset, palette="muted")
plt.show()


# #### How many studends passed in reading ?

# In[ ]:


dataset['Reading_PassStatus'] = np.where(dataset['reading score']<passmark, 'Fail', 'Pass')
dataset.Reading_PassStatus.value_counts()


# In[ ]:


plt.figure(figsize=(20,10))
p = sns.countplot(x='parental level of education', data = dataset, hue='Reading_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# #### Let us explore writing score

# In[ ]:


plt.figure(figsize=(25,16))
p = sns.countplot(x="writing score", data = dataset, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# #### How many students passed writing ?

# In[ ]:


dataset['Writing_PassStatus'] = np.where(dataset['writing score']<passmark, 'Fail', 'Pass')
dataset.Writing_PassStatus.value_counts()


# In[ ]:


plt.figure(figsize=(20,10))
p = sns.countplot(x='parental level of education', data = dataset, hue='Writing_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# #### Iet us check "How many students passed in all the subjects ?"

# In[ ]:


dataset['OverAll_PassStatus'] = dataset.apply(lambda x : 'Fail' if x['Math_PassStatus'] == 'Fail' or
                                    x['Reading_PassStatus'] == 'Fail' or x['Writing_PassStatus'] == 'Fail' else 'Pass', axis =1)

dataset.OverAll_PassStatus.value_counts()


# In[ ]:


plt.figure(figsize=(20,10))
p = sns.countplot(x='parental level of education', data = dataset, hue='OverAll_PassStatus', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# #### Find the percentage of marks

# In[ ]:


dataset['Total_Marks'] = dataset['math score']+dataset['reading score']+dataset['writing score']
dataset['Percentage'] = dataset['Total_Marks']/3


# In[ ]:


plt.figure(figsize=(25,16))
p = sns.countplot(x="Percentage", data = dataset, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=0) 


# #### Let us assign the grades
# 
# ### Grading 
# ####    above 80 = A Grade
# ####      70 to 80 = B Grade
# ####      60 to 70 = C Grade
# ####      50 to 60 = D Grade
# ####      40 to 50 = E Grade
# ####    below 40 = F Grade  ( means Fail )
# 

# In[ ]:


def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'Fail'):
        return 'Fail'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'Fail'

dataset['Grade'] = dataset.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

dataset.Grade.value_counts()


# #### we will plot the grades obtained in a order

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x="Grade", data = dataset, order=['A','B','C','D','E','Fail'],  palette="muted")
plt.show()


# In[ ]:


plt.figure(figsize=(25,16))
p = sns.countplot(x='parental level of education', data = dataset, hue='Grade', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 

