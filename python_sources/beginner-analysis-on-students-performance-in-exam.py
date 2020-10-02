#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This data set shows students's grades from 3 exams, students' race, their parent's education level, their type of lucnh and their test preparation status.
# <font color="red">
# 
# Content:
# 1. [Load and Check Data](#1)
#    
# 2. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Varibale Analysis](#4)
#         * [Numerical Variable Analysis](#5)
#  
# 3. [Basic Data Analysis](#6)
# 4. [Outlier Detection](#7)
# 5. [Missing Value](#8)
#     * [Find Missing Value](#9)
#     * [Fill Missing Value](#10)
# 
#    

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns
from collections import Counter
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a><br>
# # Load and Check Data

# In[ ]:


data=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.describe()


# <a id="2"></a><br>
# # Variable Description
# 1. gender: gender of students
# 2. race/ethnicity: ethnicity of students
# 3. parental level of education: students' parents level of education
# 4. lunch: Is lunch's price standard or reduced/free
# 5. test preparation course: status in test preparation course (completed or not)
# 6. match score: students' grades in math
# 7. reading score: students' grades in reading
# 8. writing score: students' grades in writing

# In[ ]:


data.info()


# * int64(3): Match score, reading score and writing score 
# * object(5): gender, race/ethnicity, parental level of education, lunch, test preparation course

# <a id="3"></a><br>
# # Univariate Variable Analysis
# * Categorical Variable: gender, race/ethnicity, parental level of education, lunch, test preparation course
# * Numerical Variable: math score, reading score, writing score

# <a id="4"></a><br>
# ## Categorical Variable

# In[ ]:


def bar_plot(variable):
    """
    
    input: variable ex:"race"
    output: bar plot & value count
    """
    #get feature
    var=data[variable]
    #count number of categorical variable(value/sample)
    varValue=var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[ ]:


category1=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
for c in category1:
    bar_plot(c)


# <a id="5"></a><br>
# ## Numerical Variable

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(data[variable])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericvar=["math score","reading score","writing score"]
for n in  numericvar:
    plot_hist(n)


# <a id="6"></a><br>
# # Basic Data Analysis
# * **test preparation course-test scores**
#     * test preparation course vs math score
#     * test preparation course vs reading score
#     * test preparation course vs writing  score
#     * test preparation vs average score
#     
# * **race/ethnicity-test scores**
#    * race/ethnicity vs math score
#    * race/ethnicity vs reading score
#    * race/ethnicity vs writing  score
#    * race/ethnicity vs average score
#    
# * **gender-test scores**
#    * gender vs math score
#    * gender vs reading score
#    * gender vs writing  score
#    * gender vs average  score
#    
# * **parental level of education-test scores**
#    * parental level of education-test scores vs math score
#    * parental level of education-test scores vs reading score
#    * parental level of education-test scores vs writing  score
#    * parental level of education-test scores vs average  score
#    
# * **lunch-test scores**
#    * lunch vs math score
#    * lunch vs reading score
#    * lunch vs writing  score
#    * lunch vs average  score
# 
#  

# ## Test Preparation Course vs Students' Test Scores

# ### Test Preparation Course vs Math Score

# In[ ]:


data[["test preparation course","math score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="math score",ascending=False)


# Students who completed test preparation course are slighlty more successfull in math exam than student who did not complete

# ### Test Preparation Course vs Reading Score

# In[ ]:


data[["test preparation course","reading score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="reading score",ascending=False)


# Students who completed test preparation course are slighlty more successfull in reading exam than student who did not complete

# ### Test Preparation Course vs Writing Score

# In[ ]:


data[["test preparation course","writing score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="writing score",ascending=False)


# Students who completed test preparation course are slighlty more successfull in writing exam than student who did not complete. Max difference is in this dataset as about 10

# ### Test Preparation vs Average Score

# In[ ]:


# Creating average column by using math, writing and reading scores
data["Average"]=[(data["math score"][a]+data["writing score"][a]+data["reading score"][a])/3 for a in range(len(data)) ]


# In[ ]:


data.head()


# In[ ]:


data[["test preparation course","Average"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="Average",ascending=False)


# We can see that students who complete test preparation course are more successful than students who does not complete the course

# ## Race/Ethnicity vs Students' Test Scores

# ### Race/Ethnicity vs Math score

# In[ ]:


data[["race/ethnicity","math score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="math score",ascending=False)


# ### Race/Ethnicity vs Reading Score

# In[ ]:


data[["race/ethnicity","reading score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="reading score",ascending=False)


# ### Race/Ethnicity vs Writing Score

# In[ ]:


data[["race/ethnicity","writing score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="writing score",ascending=False)


# ### Race/Ethnicity vs Average Score

# In[ ]:


data[["race/ethnicity","Average"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="Average",ascending=False)


# There is a correlation between race of students and their scores. In math, writing, reading and average, the placement stays same as E-D-C-B-A from most successful to less successful

# ## Gender vs Students' Test Scores

# ### Gender vs Math Score

# In[ ]:


data[["gender","math score"]].groupby(["gender"],as_index=False).mean().sort_values(by="math score",ascending=False)


# ### Gender vs Reading Score

# In[ ]:


data[["gender","reading score"]].groupby(["gender"],as_index=False).mean().sort_values(by="reading score",ascending=False)


# ### Gender vs Writing score 

# In[ ]:


data[["gender","writing score"]].groupby(["gender"],as_index=False).mean().sort_values(by="writing score",ascending=False)


# ### Gender vs Average Score

# In[ ]:


data[["gender","Average"]].groupby(["gender"],as_index=False).mean().sort_values(by="Average",ascending=False)


# Female students are better in writing and reading while male students are better in math. In average, females are more successful than males.

# ## Parental Level of Education vs Students' Test Scores

# 
# ### Parental Level of Education-Test Scores vs Students' Math Scores

# In[ ]:


data[["parental level of education","math score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="math score",ascending=False)


# ### Parental Level of Education-Test Scores vs Students' Reading Scores

# In[ ]:


data[["parental level of education","reading score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="reading score",ascending=False)


# ### Parental Level of Education-Test Scores vs Students' Writing Scores

# In[ ]:


data[["parental level of education","writing score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="writing score",ascending=False)


# ### Parental Level of Education-Test Scores vs Students' Average Scores

# In[ ]:


data[["parental level of education","Average"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="Average",ascending=False)


# When we compare parental level of education with students' scores, we can see that if parents's education level is master's degree, these parents' child gets higher grades. On the other hand, if parents's education level is high school, these parents' child gets lower grades

# ## Lunch vs Students' Test Scores

# ### Lunch vs Students' Math Scores

# In[ ]:


data[["lunch","math score"]].groupby(["lunch"],as_index=False).mean().sort_values(by="math score",ascending=False)


# ### Lunch vs Students' Reading Scores

# In[ ]:


data[["lunch","reading score"]].groupby(["lunch"],as_index=False).mean().sort_values(by="reading score",ascending=False)


# ### Lunch vs Students' Writing Scores

# In[ ]:


data[["lunch","reading score"]].groupby(["lunch"],as_index=False).mean().sort_values(by="reading score",ascending=False)


# ### Lunch vs Students' Average Scores

# In[ ]:


data[["lunch","Average"]].groupby(["lunch"],as_index=False).mean().sort_values(by="Average",ascending=False)


# In our comparasion, we can conclude that students who have free or reduced lunch are less successful than students who have standart lunch

# <a id="7"></a><br>
# # Outlier Detection

# In[ ]:


def detect_outliers(df,features):
    outlier_indices=[]
    for c in features:
        #1st quartile
        Q1=np.percentile(df[c],25)
        
        #3rd quartile
        Q3=np.percentile(df[c],75)
        
        #IQR
        IQR=Q3-Q1
        
        #Outliers Step
        outlier_step=IQR*1.5
        
        #Detect outlier and their indeces
        outlier_list_col=df[(df[c]<Q1-outlier_step)|(df[c]>Q3+outlier_step)].index
        
        #store indeces
        outlier_indices.extend(outlier_list_col)
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)
    return multiple_outliers
        
                            
            


# In[ ]:


data.boxplot()


# In[ ]:


clean_data=data.drop(detect_outliers(data,["math score","writing score","reading score","Average"]),axis=0).reset_index(drop=True)


# In[ ]:


clean_data.boxplot() # We dropped some of the outliers.


# <a id="8"></a><br>
# # Missing Values
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


clean_data.columns[clean_data.isnull().any()] #We do not have any column that contains missing values


# We do not need to fill missing values because we do not have any in this data set

# In[ ]:




