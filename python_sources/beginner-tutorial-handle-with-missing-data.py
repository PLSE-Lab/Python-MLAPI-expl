#!/usr/bin/env python
# coding: utf-8

# # Preprocessing of Missing Data
# 1. Loading Dataset
# 2. Exploratory Analysis - EDA
# 3. Apporaches on Evaluation of the Missing Values
# 4. Delete or Impute
# 5. Evalutaion of Imputation Techniques

# # Loading Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#import data

import pandas as pd

df=pd.read_csv('/kaggle/input/pima-indians_diabetes.csv')

df.info()


# # Exploratory Analysis

# I prefer to use pandas-profiling for small dataset, because it provides us with fundamental descriptive statistics and visualisation by one single code line. If you wonder about detail of usage and development history of pandas-profiling, you can look at https://github.com/pandas-profiling/pandas-profiling or https://pypi.org/project/pandas-profiling/.  

# In[ ]:


# EDA : describe data quickly by pandas-profiling

from pandas_profiling import ProfileReport

profile=ProfileReport(df,title='Descriptive Analysis of Diabetes',html={'style':{'full_width':True}})


# In[ ]:


profile.to_widgets()


# ****Interpretation about Profile-Report****
# 
# In this kernel, we will focus on how to deal with missing data.On the Missing widget, you can deeply investigate missing's count, correlations, and some visualisation such as Heatmap, dendrogram. In the dataset, while all number of observation is 768, missing cell ratio is 9.4%. When we look at details of missing data, 'Serum_Insulin' (49%) and 'Skin_Fold'(30%) have higher missing proportion among variables. The variable of 'BMI', 'Glucose','Diastolic_BP' have relatively small proportion of missingness within observation's volume.

# # Apporaches on Evaluation of the Missing Values

# 
# First of all, we should understand why data goes missing. In the terminology, the reason which triggers missing condition is composed of three concepts;
# 
# 1. MAR - Missing at Random
# 
# It means that propensity of missingness is not related to the missing data itself,rather it is related some of the observed data. In this situation, we cannot say major problem for removing missing values from dataset.
# 
# 2. MCAR - Missing Completely at Random
# 
# It means that propensity of missingness is not related to neither the missing data itself nor the observed data. In this situation, we can remove missing values from dataset.
# 
# 3. MNAR - Missing not at Random 
# 
# It means that propensity of missingness is correlated with hypothetical value or some other variables' value. In this situation, imputation has high probability of giving better prediction result than removing them. Actually, we can say that imputation is necessity for better results.
# 

# ****Imputation Techniques****
# 
# Imputation techniques are differantiated from fundamental feature of the dataset. For example,while we implement KNN imputation at regression or classification problem, we implement linear interpolatation at time series analysis. In the diabetes dataset, we are striking to solve a classification problem. Therefore, I will mention about first-case imputation techniques. First of all, we seperately behave on variables according to their types, including categorical and numeric-float. In the  diabetes dataset, there is no categorical variable, so we will investigate how to deal with categorical variables at another kernel.
# 
# **Basic Imputation**
# 
# * Mode        
# * Mean       
# * Median
# * Constant
# 
# Basic imputation technique reduce variance in the dataset as main disadvantage.
# 
# **Advance Imputation**
# 
# It means actually, create ML models to predict composition of missing value. In this kernel, we will mention about;
# 
# * KNN 
# * MICE
# 
# Actaully there ara many sources to explain all of these methods, but i attach an short one. https://www.paultwin.com/wp-content/uploads/Lodder_1140873_Paper_Imputation.pdf
# 
# 

# # Delete or Impute
# 
# ****Lets Practice on Dataset****

# In[ ]:


# TYPE OF MISSINGNESS

"""We can use pandas profiling report for that, but i imlement 'missingno' package as being different tool"""

import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df)

plt.show()


# The above graphic gives us missingness pattern of each variable. We can say that 'BMI' and 'Glucose' are MCAR type; correlation is high but number of missing value is low, while 'Skin_Fold' and 'Serum_Insulin' are MNAR, plus Diastolic is MAR. We will use different techniques to evaluate them, but firstly we will look at correation of missingess. I will demonstrate relationship of missingness above as being between missing and non-missing values. Beacuse, we also should look at relationship missing value and non-missing observed values to determine missingness type.

# In[ ]:


"""Analyzing the missingness of a variable against another variable helps 
you determine any relationships between missing and non-missing values. """

import numpy as np
from numpy.random import rand

def dummy(df, scaling_factor=0.075):
    df_dummy = df.copy(deep=True)
    for col_name in df_dummy:
        
        col = df_dummy[col_name]
        col_null = col.isnull()    
    # Calculate number of missing values in column 
        num_nulls = col_null.sum()
    # Calculate column range
        col_range = col.max() - col.min()
    # Scale the random values to scaling_factor times col_range
        dummy_values = (rand(num_nulls) - 2) * scaling_factor * col_range + col.min()
        col[col_null] = dummy_values
    return df_dummy


# Fill dummy values in diabetes_dummy
diabetes_dummy = dummy(df)

# Sum the nullity of Skin_Fold and BMI
nullity0 = df['Skin_Fold'].isnull()+df['BMI'].isnull()

# Create a scatter plot of Skin Fold and BMI 
diabetes_dummy.plot(x='Skin_Fold', y='BMI', kind='scatter', alpha=0.5,
                    
                    # Set color to nullity of BMI and Skin_Fold
                    c=nullity0, 
                    cmap='rainbow')


plt.show()


# In[ ]:


# Sum the nullity of Skin_Fold and Serum_Insulin
nullity1 = df['Skin_Fold'].isnull()+df['Serum_Insulin'].isnull()

# Create a scatter plot of Skin Fold and BMI 
diabetes_dummy.plot(x='Skin_Fold', y='Serum_Insulin', kind='scatter', alpha=0.5,
                    
                    # Set color to nullity of BMI and Skin_Fold
                    c=nullity1, 
                    cmap='rainbow')


# Red point reflects missing and blue point reflects non-missing values. As you can see, the 'BMI' and the 'Skin_Fold' variables are not correlated. The missing values of the 'Skin_Fold' variable spreadout throughout y axis. However, the missing values of the 'Serum_Insulin' spreadout throughout x_axis, so the correlated relationship can be seen at this graphic also.

# In[ ]:


# Correlations among Missingness

# We can use heatmap or dendrogram which you have already seen at the profile-report. But in this code-line, i will missingno package.


# Plot missingness heatmap of diabetes
msno.heatmap(df)

# Plot missingness dendrogram of diabetes
msno.dendrogram(df)

# Show plot
plt.show()


# **Implementation of Deletion**

# In[ ]:


# I will delete MAR type of missingness. I implement both 'all' and 'any' strategies to give example.Furthermore: 
"""https://www.w3resource.com/pandas/dataframe/dataframe-dropna.php"""

# Print the number of missing values in MAR types
print(df['Glucose'].isnull().sum())
print(df['BMI'].isnull().sum())

df_2 = df.copy(deep=True)

# Drop rows where 'Glucose' has a missing value
df_2.dropna(subset=['Glucose'], how='any', inplace=True)

# Drop rows where 'BMI' has a missing value
df_2.dropna(subset=['BMI'], how='all', inplace=True)

df_2.info()

df_drop=df.copy(deep=True)
df_drop.dropna(how='any',inplace=True)


#  **Implementation Imputation Techniques**

# 1. Mean-Median-Mode-Constant Imputations

# In[ ]:


# I will create dummy dataframe for all techniques which i implement



from sklearn.impute import SimpleImputer

## Mean Imputation

# Make a copy of diabetes
diabetes_mean = df_2.copy(deep=True)

# Create mean imputer object
mean_imputer = SimpleImputer(strategy='mean')

# Impute mean values in the DataFrame diabetes_mean
diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)

## Median Imputation

# Make a copy of diabetes
diabetes_median = df_2.copy(deep=True)

# Create median imputer object
median_imputer = SimpleImputer(strategy='median')

# Impute median values in the DataFrame diabetes_median
diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)

## Mode Imputation

# Make a copy of diabetes
diabetes_mode = df_2.copy(deep=True)

# Create mode imputer object
mode_imputer = SimpleImputer(strategy='most_frequent')

# Impute using most frequent value in the DataFrame mode_imputer
diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)

## Constant Imputation

# Make a copy of diabetes
diabetes_constant = df_2.copy(deep=True)

# Create median imputer object
constant_imputer = SimpleImputer(strategy='constant', fill_value=0)

# Impute missing values to 0 in diabetes_constant
diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)


# 1. KNN-MICE Imputation

# In[ ]:


# Import KNN from fancyimpute
from fancyimpute import KNN

# Copy diabetes to diabetes_knn_imputed
diabetes_knn_imputed = df_2.copy(deep=True)

# Initialize KNN
knn_imputer = KNN()

# Impute using fit_tranform on diabetes_knn_imputed
diabetes_knn_imputed.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn_imputed)


# Import IterativeImputer from fancyimpute
from fancyimpute import IterativeImputer

# Copy diabetes to diabetes_mice_imputed
diabetes_mice_imputed = df_2.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer()

# Impute using fit_tranform on diabetes
diabetes_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(diabetes_mice_imputed)


# # Evaluation of Imputation Techniques
# 

# **Basic Linear Model**
# 
# Actually, I have to make some preprocessing implementation, such as scaling,splitting data, etc.. before fitting model algorithms on data , after fitting, to implement hypertuning.However, in this kernel we are focusing on missing values and i will evaluate effects of imputations' techniques by simple linear model result

# In[ ]:


# Basic Graphics to demonstrate bias

df['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)
df_drop['Skin_Fold'].plot(kind='kde')
diabetes_mean['Skin_Fold'].plot(kind='kde')
#diabetes_median['Skin_Fold'].plot(kind='kde')
#diabetes_mode['Skin_Fold'].plot(kind='kde')
#diabetes_constant['Skin_Fold'].plot(kind='kde')
diabetes_knn_imputed['Skin_Fold'].plot(kind='kde')
diabetes_mice_imputed['Skin_Fold'].plot(kind='kde')
labels = ['First_Df','Baseline (Drop Any)', 'Mean Imputation', 'KNN Imputation',
'MICE Imputation']
plt.legend(labels)
plt.xlabel('Skin Fold')

#'Median_Imputation','Mode_Imputation','Constant_Imputation'


# **Summary**
# 
# At the final graphic, we can see which imputation technique how generates bias from original dataset. While mean imputation is completely out of shape as compared to other impututations, the MICE and the KNN has lower bias. However, dataset which is generated by dropping any values is most resembled to original dataset.
