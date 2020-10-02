#!/usr/bin/env python
# coding: utf-8

# This is my personal note for common functions in data analysis so that I can refer to in times of need.

# # Basics

# ## Loading libraries

# In[ ]:


import numpy as np # linear algebra - arrays & matrices
import pandas as pd # for data structures & tools, data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp #for integrals, solving differential equations, optimization
import matplotlib.pyplot as plt #for plots, graphs - data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plots - heat maps, time series & violin plots
import sklearn as sklearn #machine learning models
import statsmodels as stmodels #explore data, estimate statistical models, & perform statistical test

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Loading a dataset

# In[ ]:


#load train dataset
data = '../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
dataset = pd.read_csv(data) #for those datasets without headers can use dataset = pd.read_csv(data, header = None)
dataset_withoutheaders = pd.read_csv(data, header = None)
dataset.shape


# ## Exploring a dataset

# In[ ]:


dataset.columns #finding name of columns of the dataset


# **Description on columns**
# * sl_no          :           Serial Number
# * gender         :           Gender - Male='M, Female='F'
# * ssc_p          :           Secondary Education percentage - 10th Grade
# * ssc_b          :           Board of Education - Central/ Others
# * hsc_p          :           Higher Secondary Education percentage- 12th Grade
# * hsc_b          :           Board of Education - Central/ Others
# * hsc-s          :           Specialization in Higher Secondary Education
# * degree_p       :           Degree Percentage
# * degree_t       :           Under Graduation(Degree type) - Field of degree education
# * workex         :           Work Experience
# * etest_p        :           Employability test percentage (conducted by college)
# * specialisation :           Post Graduation(MBA) - Specialization
# * mba_p          :           MBA percentage
# * status         :           Status of placement- Placed/Not placed
# * salary         :           Salary offered by corporate to candidates
# 
# <br> 
# 
# **What is in it?** <br>
# This dataset consists of Placement data of students at Jain University Bangalore. It includes secondary and higher secondary school percentage and specialization. It also includes degree specialization, type and Work experience and salary offers to the placed students.
# 
# <br>
# 
# **Questions we can ask about this dataset** <br>
# Which factor influenced a candidate in getting placed? <br>
# Does percentage matters for one to get placed? <br>
# Which degree specialization is much demanded by corporate? <br>
# Play with the data conducting all statistical tests. <br>

# In[ ]:


dataset.head()


# In[ ]:


dataset_withoutheaders.head()


# Since this dataset consists of headers already, when I use header = None, it treated the headers as rows or data points. <br> <br>
# Lets say the dataset doesn't contain headers and so we used header = None to import data, we can assign column names/headers with a panda method as below: <br> <br>
# headers = ["sl_no","gender","ssc_p","hsc_p","hsc_s" etc...] <br>
# dataset.columns = headers

# To export the modified dataset to file <br> <br>
# export_path = "C:\Windows\....\exportedfile.csv" <br>
# df.to_csv(export_path) <br> <br>
# 
# **Different formats** <br>
# * For csv format, reading: pd.read_csv(), exporting: df.to_csv()
# * For json format, reading: pd.read_json(), exporting: df.to_json()
# * For Excel format, reading: pd.read_excel(), exporting: df.to_excel()
# * For SQL format, reading: pd.read_sql(), exporting: df.to_sql()

# **Different Datatypes**
# * Pandas Type: object, Native Python Type: string, Description: numbers and strings
# * Pandas Type: int64, Native Python Type: int, Description: Numeric characters
# * Pandas Type: float64, Native Python Type: float, Description: Numberic characters with decimals
# * Pandas Type: datetime64,timedelta[ns], Native Python Type: refer to datetime module in Python's standard library, Description: time data

# In[ ]:


#checking datatypes of features

dataset.dtypes


# General rule of thumb, prices & measurement values should be float64 - otherwise, it will be a problem in an analysis.

# In[ ]:


dataset.describe() #returns a statistical summary - but this method does not include object datatype columns, basically skip rows & columns that do not contain numbers


# In[ ]:


#for full summary with every column & row
dataset.describe(include = "all")


# * unique - number of distinct objects in the column
# * top - most frequently occuring object
# * freq - number of times the top object appears in the column
# * NaN - Not A Number

# In[ ]:


dataset[['sl_no', 'salary', 'gender']].describe(include = "all") #describing only sl_no & salary columns in the dataset, notice I can arrange the way columns appear sl_no --> salary --> gender

#include = "all" is added to display gender column since it is an object type, else not needed


# In[ ]:


dataset.info()


# ## Accessing databases with Python

# In[ ]:


#DB-API
from dbmodule import connect

#Create a connection object
connection = connect('databasename', 'username', 'password')

#Create a cursor object
cursor = connection.cursor()

#Run queries
cursor.execute('select * from tablename')
results = cursor.fetchall()

#Free resources
Cursor.close()
connection.close()


# # Data Pre-processing

# ## Identifying & handling missing values

# 
# Missing values appear in the dataset as "?", "NaN", "0" or a blank cell.
# 

# **Dealing with missing values**
# 
# * Check with the data collection source
# * Drop the missing values - either by dropping the variable (whole column) or just the data entry with the missing value (just the row)
# * Replacing the missing values - replacing with an mean/ average value (of similar datapoints) -> this will not work on categorical data, replacing it by mode/ most frequent value, replacing it based on other functions (based on domain knowledge or experience)
# * Leaving it as a missing data - least preferable

# ### Dropping rows with NaN values

# In[ ]:


#output missing data
missing_data = dataset.isnull()
missing_data.head(10)


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# In[ ]:


dataset.head(10)


# In[ ]:


dataset.dropna(subset=["salary"], axis = 0, inplace = True) #axis = 0 will drop the entire row with NaN value & axis = 1 will drop the entire column with NaN value, inplace = True will write the result back to the original dataset

#dataset.dropna(subset=["salary"], axis = 0, inplace = True) is the same as dataset = dataset.dropna(subset=["salary"], axis = 0)
#if want to drop all rows with NaN value regardless of column/variable just use dataset.dropna(axis = 0, inplace = True)


# In[ ]:


dataset.head(10)


# ### Replacing missing values with new values

# In[ ]:


#reset the dataset to original
dataset = pd.read_csv(data)
dataset.head(10)


# In[ ]:


#dataframe.replace(missing_value, new_value)
#replacing with mean value
mean = dataset["salary"].mean()
dataset["salary"].replace(np.nan, mean, inplace = True)
dataset.head(10)


# ## Data formatting

# Data entries can be in different formats for one expression/ value. <br>
# For example, Singapore can be in S.G., SG, S.G, S.G.P, singapore etc. <br>
# Standardizing different formats of same expression is necessary in data analysis. <br>
# Sometimes values can also be in a unit that is not preferred in the calculation/analysis such as entries are in miles instead of kilometers etc. <br> <br>
# 
# **Changing format of data entry values** <br>
# df["city-mpg"] = 235/df["city-mpg"] <br>
# df.rename(columns={"city-mpg" : "city-L/100km"}, inplace = True) <br> 
# -------------------------------------------------------------------------------------------------------------------<br>
# Another scenario could be numerical values being in object datatype and not included in the calculation. <br> <br>
# **Converting datatypes** <br>
# df["salary"] = df["salary"].astype("float")

# ## Data Normalization (Centering/Scaling)

# Data normalization is uniforming the features values with different ranges to have a fair comparison between different features.

# In[ ]:


dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']].head(10)


# Here, we can see salary feature has ridiculously high values compared to other features. This will produce a data bias in an analysis. <br> <br>
# 
# **3 methods of normalization** <br>
# * Simple Feature Sacling - dividing each value with the maximum value of that feature, will give a new value range between 0 & 1.
# * Min-Max Scaling - ((current value - minimun value)/(maximun value - minimum value)), will give a new value range between 0 & 1.
# * Z-Score Scaling - (current value - average value of feature)/standard deviation (sigma) - will give a value around 0 (typically between -3 & +3).

# In[ ]:


#Simple Feature Scaling
dataset["salary"] = dataset["salary"]/data["salary"].max()

#Min-Max Scaling
dataset["salary"] = (dataset["salary"] - dataset["salary"].min())/(dataset["salary"].max()-dataset["salary"].min())

#Z-Score Sacling
dataset["salary"] = (dataset["salary"]-dataset["salary"].mean())/dataset["salary"].std()


# ## Data Binning

# Binning is grouping of values into bins to have a better understanding of data distribution or to improve accuracy of the model.

# In[ ]:


bins = np.linspace(min(dataset["salary"]), max(dataset["salary"]),4) #to have 3 bins, need 4 equally spaced numbers hence 4 in code
group_names = ["Low", "Medium", "High"]
dataset["salary-binned"] = pd.cut(dataset["salary"], bins, labels = group_names, include_lowest = True)


# In[ ]:


dataset.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(dataset["salary"])

# set x/y labels and plot title
plt.pyplot.xlabel("Salary")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Salary Bins")


# In[ ]:


pyplot.bar(group_names, dataset["salary-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Salary")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Salary Bins")


# ## Turning Categorical values to numeric variables

# In[ ]:


dataset.gender.unique()


# In[ ]:


#To see which values are present in a particular column, we can use the ".value_counts()" method:
dataset['gender'].value_counts()


# In[ ]:


#We can see that males are the most common type. We can also use the ".idxmax()" method to calculate for us the most common type automatically:
dataset['gender'].value_counts().idxmax()


# **One-hot encoding method** <br>
# So we have 2 categorical values in gender feature. <br>
# We can encode these categorical values of gender into numeric values by adding dummy variables for each unique category and assign 0/1 in each category.

# In[ ]:


dummy_variable_1 =  pd.get_dummies(dataset["gender"])
dummy_variable_1.head()


# In[ ]:


dataset.head(10)


# In[ ]:


# merge data frame "dataset" and "dummy_variable_1" 
dataset = pd.concat([dataset, dummy_variable_1], axis=1)

# drop original column "gender" from "dataset"
dataset.drop("gender", axis = 1, inplace=True)


# In[ ]:


dataset.head(10)


# # Exploratory Data Analysis (EDA)

# EDA is used to -:
# * Summarize main characteristics of the data
# * Gain better understanding of the data set
# * Uncover relationships between variables
# * Uncover important variables <br> <br>
# 
# In this dataset, using EDA, we can uncover what are the characteristics that have the most impact on the salary? But as mentioned before, my focus on this notebook is not performing a full analysis on the dataset, rather testing out and noting down the Data Analysis functions for my future reference.

# ## Descriptive Statistics

# Descriptive analysis provide understanding on basic features of the data & give short summary on the sample and measure of the data.

# In[ ]:


#reset the dataset to original
dataset = pd.read_csv(data)
dataset.head(10)


# In[ ]:


#basic descriptive statistics function
dataset.describe(include = "all")


# In[ ]:


#summarizing categorical data
print(dataset["gender"].value_counts())
print()
print(dataset["ssc_b"].value_counts())
print()
print(dataset["hsc_b"].value_counts())
print()
print(dataset["hsc_s"].value_counts())
print()
print(dataset["degree_t"].value_counts())
print()
print(dataset["workex"].value_counts())
print()
print(dataset["specialisation"].value_counts())
print()
print(dataset["status"].value_counts())


# In[ ]:


#Box Plots
sns.boxplot(x="gender", y="salary", data=dataset)


# **How to read a box plot**
# * the horizontal line at the top is upper extreme value
# * the horizontal line at the bottom is lower extreme value
# * the horizontal line at the center of the box is median vlaue (middle datapoint)
# * the upper quartile is the 75th percentile of data
# * the lower quartile is the 25th percentile of data
# * the dots outside the upper & lower extreme values are outliers
# <br> <br>
# The data between the upper and lower quartile represents the interquartile range. <br>
# The upper & lower extreme values are calculated as 1.5 times the interquartile range above the 75th percentile and as 1.5 times the interquartile range below the 25th percentile.

# In[ ]:


#Scatter Plots
x = dataset["etest_p"]
y = dataset["salary"]
plt.scatter(x,y)

plt.title("Employment Test Percentage vs Salary offered by corporate to candidates")
plt.xlabel("Employment Test Percentage")
plt.ylabel("Salary Offered")


# Scatter plots are suitable to show  the relationship between 2 continuous data. <br>
# Scatter plot has 2 variables - independent variable & dependent variable. Independent variable is used to predict the dependent variable. <br>
# Normally, an independent variable is placed on the x-axis & a dependent variable is placed on the y-axis.

# In[ ]:


#Groupby Function
dataset_test = dataset[['gender', 'degree_t', 'salary']]
dataset_grp = dataset_test.groupby(['gender', 'degree_t'], as_index = False).mean() #.mean() is used to see how average salary differ across different groups
dataset_grp


# In[ ]:


#Pivot Table
dataset_pivot = dataset_grp.pivot(index = 'gender', columns = 'degree_t')
dataset_pivot


# In[ ]:


#Heatmap Plot
fig, ax = plt.subplots()
im = ax.pcolor(dataset_pivot, cmap='RdBu')

#label names
row_labels = dataset_pivot.columns.levels[1]
col_labels = dataset_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(dataset_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(dataset_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[ ]:


#Correlation
sns.regplot (x="etest_p", y="salary", data=dataset)
plt.ylim(0,)


# There seems to be a positive linear relationship between 2 variables albeit rather weak. <br>
# But note that correlation does not mean causation

# ### Correlation Strength
# #### Pearson Correlation Method
# 
# * Measure the strength of the correlation between two features (Correlation coefficient & P-value)
# * Correlation coefficient (Close to +1: Strong positive relationship, Close to -1: Strong negative relationship, Close to 0: No relationship)
# * P-value (P-value<0.001 Highly confident in the result/relationship, P-value<0.05 Moderately confident, P-value<0.1 Slightly confident, P-value>0.1 Not confident)

# In[ ]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataset['etest_p'], dataset['salary'])


# Error encountered due to NaN values involved.

# In[ ]:


#replacing with mean value
etest_mean = dataset["etest_p"].mean()
dataset["etest_p"].replace(np.nan, etest_mean, inplace = True)


# In[ ]:


salary_mean = dataset["salary"].mean()
dataset["salary"].replace(np.nan, salary_mean, inplace = True)


# In[ ]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataset['etest_p'], dataset['salary'])


# In[ ]:


print("Correlation Coefficient is " + str(pearson_coef))
print("p_value is " + str(p_value))


# Correlation coefficient is not much. So it is a weak positive correlation. <br>
# And p_value is less than 0.05, so we are moderately certain that it is a weak positive correlation.

# In[ ]:


#finding correlation value between multiple features
dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']].corr()


# In[ ]:


#Correlation Heatmap
corr_mat = dataset.corr()

plt.figure(figsize = (13,5))
sns_plot = sns.heatmap(data = corr_mat, annot = True, cmap='GnBu')
plt.show()


# Note that the correlation heat map left out features with categorical values. To include them, we need to encode them to numerical labels such as 1 for Male, 0 for Female in gender etc.

# ### Analysis of Variance (ANOVA)
# #### ANOVA is finding a correlation between different groups of a categorical variable
# 
# ANOVA returns 2 values:
# * F-test score: variation between sample group means divided by variation within sample group
# * p-value : confidence degree
# 
# Small F-test score means a weak correlation between variable categories & a target variable.

# In[ ]:


dataset.specialisation.unique()


# In[ ]:


dataset_anova = dataset[["specialisation", "salary"]]
grouped_anova = dataset_anova.groupby(["specialisation"])

f_val, p_val = stats.f_oneway(grouped_anova.get_group("Mkt&HR")["salary"],grouped_anova.get_group("Mkt&Fin")["salary"])
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# The result is not good. A small F test score showing a weak correlation and a not so small P value implying that it is not certain about statistical significance.
