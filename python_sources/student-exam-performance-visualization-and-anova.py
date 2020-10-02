#!/usr/bin/env python
# coding: utf-8

# **Students Performance in Exams**
# 
# **Introduction**: This dataset consists of student test score data for subjects including math, reading, and writing. The goal of this analysis is to determine correlation between the catergorical variables('gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course') and the test scores in math, reading, and writing.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# 

# In[ ]:


#create function for summary data
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
def overview(dataframe):
    #docstring
    '''
    Read a csv file into a DataFrame.
    Print first 5 rows of data.
    Print datatype for each column.
    Print number of NULL/NaN values for each column.
    Print summary data.
    
    Return:
    data, rtype: DataFrame
    '''
    print("The first 5 rows of data are:\n", df.head())
    print("\n")
    print("The (Row,Column) is:\n", df.shape)
    print("\n")
    print("Data type of each column:\n", df.dtypes)
    print("\n")
    print("The number of null values in each column are:\n", df.isnull().sum())
    print("\n")
    print("Summary of data:\n", df.describe())
    return

overview(df)


# From our data summary, we can see that there are no null values indicating we are working with a clean dataset. Additionally, the scores in math, reading, and writing contain very similar averages.

# In[ ]:


#Create function to display distribution pairplot
def distribution(dataset, variable):
    '''
    Args:
        dataset: Include the DataFrame here
        variable: Include the column from dataframe used for color encoding
    Returns:
        sns pairplot with color encoding
    '''
    g = sns.pairplot(data = dataset, hue = variable)
    g.fig.suptitle('Graph showing distribution between scores and {}'.format(variable), fontsize=20)
    g.fig.subplots_adjust(top=0.9)
    return g


# In[ ]:


df.columns


# In[ ]:


#Score and gender
distribution(df, 'gender')


# <font color=red>Females</font> perform higher in <font color=red>reading</font> and <font color=red>writing</font> while <font color=blue>males</font> perform higher on <font color=blue>math</font>.

# In[ ]:


#score and race
distribution(df, 'race/ethnicity')


# This data does tell us much because the dataset does not describe what each group is in reference to.

# In[ ]:


#score and parental education level
distribution(df, 'parental level of education')


# There appears to be a trend in parental education level and student's score. The variance between the different catergorical data indicates this is not a major factor.

# In[ ]:


#Score and lunch
distribution(df, 'lunch')


# Students who ate the standard lunch on average tested higher in all three subjects.

# In[ ]:


#Score and test preparation course
distribution(df, 'test preparation course')


# Students who completed a test preparation course on average tested higher in all three subjects.

# **Finding correlation between categorical variables and test scores using 1-Way ANOVA**      
# 1-Way ANOVA hypothesis:
# 1. Null hypthoesis (H0): There is no difference between groups and equality between means
# 2. Alternative hypothesis (H1): There is a difference between the means and groups.
# 
# 1-Way Anova assumptions:
# 1. Normality: Each sample is taken from a normally distributed population
# 2. Sample independence: Each sample has been drawn independently of the other samples
# 3. Variance equality: The variance of data in the different groups should be the same
# 4. Dependent variable: Should be continuous
# 
# Hypothesis:
# Using a 95% confidence internal
# 1. Null hypothesis is that they are independent.
# 2. Alternate hypothesis is that categorical data is correlated in some way.

# In[ ]:


#clean up column names for StatsModels
df.columns = ['gender', 'race', 'parental_edu', 'lunch', 'test_prep_course', 'math_score', 'reading_score', 'writing_score']


# In[ ]:


#Create anova test function
def anova_test(data, variable):
    '''
    Args: 
        data = (DataFrame)
        variable = Categorical column used for 1-way ANOVA test
    Returns: Nothing
    '''
    x = ['math_score', 'reading_score', 'writing_score']
    for i,k in enumerate(x):
        lm = ols('{} ~ {}'.format(x[i],variable), data=data).fit()
        table = sm.stats.anova_lm(lm)
        print("P-value for 1-way ANOVA test between {} and {} is ".format(x[i],variable),table.loc[variable,'PR(>F)'])


# In[ ]:


#Gender ANOVA
anova_test(df, 'gender')


# The p-values are below 0.05 indicating we can reject the null hypothesis. This confirmation shows us there is statistical correlation between test scores and gender.

# In[ ]:


#Parental education ANOVA
anova_test(df, 'parental_edu')


# The p-values are below 0.05 indicating we can reject the null hypothesis. This confirmation shows us there is statistical correlation between test scores and parental education.

# In[ ]:


#Lunch ANOVA
anova_test(df, 'lunch')


# The p-values are below 0.05 indicating we can reject the null hypothesis. This confirmation shows us there is statistical correlation between test scores and what the student ate for lunch.

# In[ ]:


#Test Prep ANOVA
anova_test(df, 'test_prep_course')


# The p-values are below 0.05 indicating we can reject the null hypothesis. This confirmation shows us there is statistical correlation between test scores and what if the student completed a test preparation course.

# Although we saw statistical significance on parent level of education and student's scores, our pairplot showed us this difference was almost negligble. We will use a counplot below to take a further look at this data.

# In[ ]:


#Create countplot for parental education and student scores
plt.figure(figsize=(12,5))
sns.countplot(data=df, x='parental_edu', hue='gender')


# Our dataset included a very low number of parents with a master's degree or bachelor's degree. Due to the low sample size we can not confidently say that students with highly educated parents will score better.

# **Summary**
# Based on our analysis of student test scores we can conclude the following:
# 1. Females perform higher in reading and writing subjects.
# 2. Males perform higher in math.
# 3. Parental education level has a negligble difference in student's test performance.
# 4. Students who ate the standard lunch tested higher than those who ate a free/reduced meal.
# 5. Students who completed a test preparation course scored higher than those who did not.
# 
# **Discussion**
# All categorical data was statistically tested against the exam scores using a 1-Way ANOVA test. This test allows us to accurately confirm whether a category of data is correlated to the numerical outcome. Using a 95% confidence internal we acheived p-values < 0.05 for each catergory of data. This allows us to reject our null hypothesis and summize that the catergorical data in this dataset is correlated to the reading, writing, and math scores.
