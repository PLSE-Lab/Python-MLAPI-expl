#!/usr/bin/env python
# coding: utf-8

# Hello Everyone!!!!!
# 
# This is a simple tutorial for EDA(exploratory Data Analysis) using Iris Dataset.

# Please Upvote if you like my work....

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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# We start with importing the main libraries.

# In[ ]:


#import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pandas_profiling as pp
get_ipython().run_line_magic('matplotlib', 'inline')


# Read the Dataset

# In[ ]:


#read the data
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')


# In[ ]:


#getting the top 5 rows Of the dataset
iris.head()


# Let's start by looking into the description of Data 

# In[ ]:


#info() looks into the datatypes and non null value counts
iris.info()


# In[ ]:


#describe() looks into the statistical measures of the numerical variables present in the dataset
iris.describe()


# In[ ]:


#for Categorical variables 
iris['Species'].describe()


# In[ ]:


#Full profile report of the dataset
pp.ProfileReport(iris)


# In[ ]:


#total no rows
iris.index


# In[ ]:


#list of columns
iris.columns


# Next step is Data Cleaning.....

# In[ ]:


#drop unwanted columns
iris.drop(['Id'],axis=1)


# In[ ]:


#check for any null values
iris.isnull().any()


# In[ ]:


#check the total number of null values in each column
iris.isnull().sum()


# As this dataset contains no null values,so nothing to worry about....
# 
# If there had been null values you should either remove null values if it contains more than 60-70% null values in a column or treat them if the no of null values is less. 

# In[ ]:


#no of unique values in a specific column
iris['Species'].nunique()


# In[ ]:


#array of unique values in a specific Column
iris['Species'].unique()


# Lets start with Data Analysis(EDA)

# We should start with Univariate Analysis(1 variable)

# In[ ]:


#count of each species(non graphical for categorical variable)
species=iris['Species'].value_counts()
species


# In[ ]:


#Graphical for categorical variable(pie chart)
plt.pie(species,labels=species.index,autopct='%1.1f%%')
plt.title('Percentage Distribution of each Species')


# In[ ]:


#Graphical for categorical variable(count or bar chart)
sns.countplot(x='Species',data=iris)
plt.title('Count of each Species')


# In[ ]:


#kernel density or histogram(Numerical variable)
sns.kdeplot(iris['SepalLengthCm'])
plt.title('Distribution of Sepal Length in cm for every Species')


# In[ ]:


#Use of FacetGrid for mapping histogram(numerical variable) with Categorical variable
sns.FacetGrid(iris,hue='Species',size=5).map(sns.kdeplot,'SepalLengthCm').add_legend()


# In[ ]:


#boxplot helps in determining Median,25th percentile,75th percentile and outliers
sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
plt.title('Boxplot distribution for each species')


# In[ ]:


#use of boxplot with points distribution
sns.boxplot(x='Species',y='SepalLengthCm',data=iris)
sns.stripplot(x='Species',y='SepalLengthCm',data=iris,jitter=True)
plt.title('Boxplot with points distribution of each species')


# In[ ]:


#better to use violin plot if you dont want to use the above plots
#tells us about the density of points
#fatter where more points
#thinner where less points
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.title('violin density plot for each Species')


# We have finished with univariate analysis.....
# 
# Lets start with multivariate analysis or bivariate analysis...

# In[ ]:


#scatterplot to compare two numerical variables
iris.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm') 
plt.title('Distribution of SepalWidth vs SepalLength for all Species')


# In[ ]:


#use of jointplot to see scatter plot as well as histogram for each numerical variable
sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris)


# In[ ]:


#use of FacetGrid for comparing 2 Numerical variables with 1 Categorical variable
sns.FacetGrid(data=iris,hue='Species',size=6).map(plt.scatter,'SepalWidthCm','SepalLengthCm').add_legend()


# In[ ]:


#use of pairplot to see scatter plot between each numerical variable 
sns.pairplot(iris.drop(['Id'],axis=1),hue='Species')


# In[ ]:


#encoding of categorical variable
species_Cat=pd.get_dummies(iris['Species'],columns=['Species'])


# In[ ]:


#join encoded dataframe to main dataframe
iris_df=iris.join(species_Cat)


# In[ ]:


iris_df


# In[ ]:


iris_df.drop(['Id','Species'],axis=1)


# In[ ]:


# finding correlation matrix
corr=iris_df.corr()


# In[ ]:


#correlation matrix
corr


# In[ ]:


#heatmap to find correlation between variables using correlation matrix
sns.heatmap(corr,annot=True)
plt.title("heatmap(correlation between variables)")

