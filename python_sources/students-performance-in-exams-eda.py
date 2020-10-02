#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# <font color = "blue">
# Content :
# 1. [Load And Check Data](#1)
# 1. [Variable Description](#2)
#     * [Univariate Variable Analysis](#3)
#         * [Categorical Variables](#4)
#         * [Numerical Variables](#5)
# 1. [Basic Data Analysis](#6)
#     * [Results of Basic Data Analysis](#7)
# 1. [Outlier Detection](#8)
# 1. [Missing Value](#9)
# 
# ---> If you like it please upvote my notebook <---

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "1"></a><br>
# # Load And Check Data

# In[ ]:


# This method takes csv file and translate a dataframe
data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


# This method takes first 5 row
data.head()


# In[ ]:


# This method takes randomly 5 row
data.sample(5)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# <a id = "2"></a><br>
# # Variable Description 
# 1. gender: student's gender
# 1. race/ethnicity: race/ethnicity of students
# 1. parental level of education: parental education level of students
# 1. lunch: student's lunch type
# 1. test preparation course:courses which before the exams
# 1. math score: results of math exam
# 1. reading score: results of reading exam
# 1. writing score: results of writing exam

# <a id = "3"></a><br>
# # Univariate Variable Analysis
# * categorical Variables: Gender, Race/Ethnicity, Parental Level of Education, Lunch, Test Preparation Course
# * Numerical Variables: Math Score, Reading Score, Writing Score

# <a id = "4"></a><br>
# ## Categorical Variables

# In[ ]:


def bar_plot(variable):
    # get features
    var = data[variable]
    # get value counts
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (10,10))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} : \n{}".format(variable,varValue))


# In[ ]:


category1 = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']
for c in category1:
    bar_plot(c)


# <a id = "5"></a><br>
# ## Numerical Variables

# In[ ]:


def hist_plot(variable):
    plt.figure(figsize = (10,10))
    plt.hist(data[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} Distrubition with Histogram".format(variable))
    plt.show()


# In[ ]:


numericVariable = ['math score', 'reading score', 'writing score']

for n in numericVariable:
    hist_plot(n)


# <a id = "6"></a><br>
# # Basic Data Analysis 
# * Gender - Math Score
# * Gender - Reading Score
# * Gender - Writing Score
# * Parental Level Of Education - Math Score
# * Parental Level Of Education - Reading Score
# * Parental Level Of Education - Writing Score
# * Race/Ethnicity - Math Score
# * Race/Ethnicity - Reading Score
# * Race/Ethnicity - Writing Score
# * Test Preparation Course - Math Score
# * Test Preparation Course - Reading Score
# * Test Preparation Course - Writing Score

# In[ ]:


data.columns


# In[ ]:


data[["gender","math score"]].groupby(["gender"],as_index = True).mean().sort_values(by = "math score",ascending = False)


# In[ ]:


data[["gender","reading score"]].groupby(["gender"],as_index = True).mean().sort_values(by = "reading score",ascending = False)


# In[ ]:


data[["gender","writing score"]].groupby(["gender"],as_index = True).mean().sort_values(by = "writing score",ascending = False)


# In[ ]:


data[["parental level of education","math score"]].groupby(["parental level of education"],as_index = True).mean().sort_values(by = "math score",ascending = False)


# In[ ]:


data[["parental level of education","reading score"]].groupby(["parental level of education"],as_index = True).mean().sort_values(by = "reading score",ascending = False)


# In[ ]:


data[["parental level of education","writing score"]].groupby(["parental level of education"],as_index = True).mean().sort_values(by = "writing score",ascending = False)


# In[ ]:


data[["race/ethnicity","math score"]].groupby(["race/ethnicity"],as_index = True).mean().sort_values(by = "math score",ascending = False)


# In[ ]:


data[["race/ethnicity","reading score"]].groupby(["race/ethnicity"],as_index = True).mean().sort_values(by = "reading score",ascending = False)


# In[ ]:


data[["race/ethnicity","writing score"]].groupby(["race/ethnicity"],as_index = True).mean().sort_values(by = "writing score",ascending = False)


# In[ ]:


data.columns


# In[ ]:


data[["test preparation course","math score"]].groupby(["test preparation course"],as_index = True).mean().sort_values(by = "math score",ascending = False)


# In[ ]:


data[["test preparation course","reading score"]].groupby(["test preparation course"],as_index = True).mean().sort_values(by = "reading score",ascending = False)


# In[ ]:


data[["test preparation course","writing score"]].groupby(["test preparation course"],as_index = True).mean().sort_values(by = "writing score",ascending = False)


# <a id = "8"></a><br>
# ## Results of Basic Data Analysis
# We can draw the following conclusion from here: 
# 1. While the male group is more successful than the female group in numerical terms, the female group is also more successful than the male group in reading and writing.
# 1. In addition, the education level of parents directly affects students' exam results.
# 1. Also, race / ethnicity affects students' examination results.
# 1. And finally, whether students attend pre-exam courses also greatly affects exam results.

# <a id = "8"></a><br>
# # Outlier Detection
# ### Why we must detect outliers ? 
# * Because outliers are corrupting our datas

# In[ ]:


def detect_outliers(df,features):
    outlier_indeces = []
    for c in features:
        Q1 = np.percentile(df[c],25)
        Q3 = np.percentile(df[c],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indeces.extend(outlier_list_col)
    outlier_indeces = Counter(outlier_indeces)
    multipler_outliers = list(k for k,v in outlier_indeces.items() if v > 2)
    return multipler_outliers


# In[ ]:


data.loc[detect_outliers(data,["math score","reading score","writing score"])]


# In[ ]:


# This method drops our outliers
data = data.drop(detect_outliers(data,['math score', 'reading score','writing score']),axis = 0).reset_index(drop = True)


# In[ ]:


data.loc[detect_outliers(data,["math score","reading score","writing score"])]


# <a id = "9"></a><br>
# # Missing Value
# * This data set hasn't got any missing values.

# In[ ]:


data.columns[data.isnull().any()]


# In[ ]:


data.info()


# In[ ]:


data.isnull().any()


# In[ ]:


data.isnull().sum()

