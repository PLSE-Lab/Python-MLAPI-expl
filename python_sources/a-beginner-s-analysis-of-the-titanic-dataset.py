#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Introduction
# 
# This notebook is meant as a guide for myself and for beginners to data analysis to understand the steps and Python coding that can be done to familiarize with and explore the data, and then carry out predictive modelling. Previously, I have written code to do predictive modelling, but without any discussions or thoughts going into how/why I carried out a particular task. In this notebook, I aim to write all my thought processes that go into exploring and analysing the data.

# # Studying the Dataset
#    
# The first step in any data analysis project is to look at the data. We need to see how many observations/rows, how many features/columns are contained, what these columns mean, and so on. This will help us warm up and get familiar with the dataset, and might even help us to evaluate which features are important and which aren't. To do this, we import the relevant Python libraries and read in the `train.csv` CSV file as a data frame.

# In[ ]:


# Import data analysis libraries and future
import __future__
import numpy as np
import pandas as pd


# In[ ]:


# Import visualization libraries and set favourite grid style
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


# In[ ]:


# Read in the dataset
titanic_train = pd.read_csv('../input/titanic/train.csv')


# A quick way to check out the contents of a data frame is by calling the first 5 rows using the `.head()` method without specifying the argument within brackets. If we want to check out the first 10 rows, we put in 10 as the argument.

# In[ ]:


# Check out the first 10 rows of the training dataset
titanic_train.head(10)


# We can count 12 features describing each person on board the Titanic. Our **target** feature (also known as the **independent variable**) is the Survived column, which is 1 if the person survived and 0 if not. In this case, it is easy to infer that the Survived column only contains numerical 1's and 0's because the Kaggle description said so. For other features (or **dependent variables**) however, it may not be possible to deduce the type of data contained at first glance.  For instance, the Ticket column contains a mixture of letters and numbers for some rows but other rows contain only numbers. In order to quickly generate a table of data types contained within each column, we use the `.info()` method.

# In[ ]:


# Info about data frame dimensions, column types, and file size
titanic_train.info()


# Now we know there are 7 numerical columns (floats and integers), 5 non-numerical columns, and their names as well. The output also tells us there are 891 rows/entries and 12 columns. For numerical columns such as Age, Fare, etc., we would like to find out their mean, maximum and minimum values to see if the data is reasonably distributed or if there are any anomalies or mistakes, such as percentages above 100%. To draw up a column of statistics for each column, we use the `.describe()` method.

# In[ ]:


# Generate a summary of statistics for each numerical column of the data frame
titanic_train.describe()


# From the summary, we can see that Age has a count less than the total number of columns, which means there are some missing values (indicated by `NaN`). Columns having missing values can also be detected from the `info` method previously discussed. In fact, the Cabin column contains more than 600 missing values and the Embarked column contains 2 missing values. The rest of the columns seem to be in good shape. The only number that catches the eye could be `min(Age)` being 0.42 years, a possible 5 month old baby on board. Further information from [titanicfacts.net/titanic-victims/](http://) indicate this to be Master Gilbert Danbom, who was 4 months 29 days old at the time of sinking.

# We can further differentiate the variables into nominal, ordinal, discrete and continuous data types. The method `.nunique()` can be used to count the number of unique values/levels within in each variable, which will give us a rough idea of which variable belongs to which data type.

# In[ ]:


titanic_train.nunique()


# For the variables with less than 10 levels, we can list what these levels are using the method `.unique()`:

# In[ ]:


lessthanten = []
for col in titanic_train.columns:
    lessthanten.append(titanic_train[col].nunique() < 10)

for col in titanic_train[titanic_train.columns[lessthanten]]:
    print(col, titanic_train[col].unique())


# Another useful method for exploration is the `.value_counts()` method, which counts the number of occurrences for each unique category within a column. Consider the code below as an example:

# In[ ]:


titanic_train['Embarked'].value_counts()


# From the above outputs and from our initial familiarization with the data, we differentiate the variables as follows, and produce a dataframe to put them into a table.
#    
# **Label:** PassengerId
# 
# **Nominal:** Survived, Sex, Name, Ticket, Cabin, Embarked
# 
# **Ordinal:** Pclass 
# 
# **Discrete:** SibSp, Parch
# 
# **Continuous:** Age, Fare

# In[ ]:


pd.DataFrame({'Integer': ['Survived','Pclass','SibSp, Parch','-'], 
              'Float': ['-','-','-','Age, Fare'], 
              'Object': ['Sex, Name, Ticket, Cabin, Embarked','-','-','-']}, 
              index = ['Nominal','Ordinal','Discrete','Continuous'])


# Now that we have familiarized ourselves with the dataset, we can move on to visualizing the data. This stage is called **Exploratory Data Analysis**.

# # Exploratory Data Analysis
#   
# Refer to the data again:

# In[ ]:


# First 10 rows of the data
titanic_train.head(10)


# Firstly, we want to check out the target column Survived to see how many people survived and how many died (at least in the training dataset). A **countplot** is an easy way to visualize this type of categorical data. A table of counts and percentages can also be generated to see the numbers that go into the plot. We do this by defining a function to create the countplot and generate the table at the same time. 
#    
# **Note:** When the same snippet of code is written more than once, it is time to define a function. This is a more elegant solution than duplicating code and reduces errors in copying and pasting as well. (Also because I found myself writing the same code again and again.)
#    
# Define the fuction `count_n_plot` to generate a table of Count and Percentage, and create countplots (and barplots) of the specified feature.

# In[ ]:


# Define the count_n_plot function
def count_n_plot(df, col_name, countsplit = None, bar = False, barsplit = None):
    
    """
    Creates countplots and barplots of the specified feature 
    (with options to split the columns) and generates the 
    corresponding table of counts and percentages.
    
    Parameters
    ----------
    df : DataFrame
        Dataset for plotting.
    col_name : string
        Name of column/feature in "data".
    countsplit : string
        Use countsplit to specify the "hue" argument of the countplot.
    bar : Boolean
        If True, a barplot of the column col_name is created, showing
        the fraction of survivors on the y-axis.
    barsplit: string
        Use barsplit to specify the "hue" argument of the barplot.
    """
    
    if (countsplit != None) & bar & (barsplit != None):
        col_count1 = df[[col_name]].groupby(by = col_name).size()
        col_perc1 = col_count1.apply(lambda x: x / sum(col_count1) * 100).round(1)
        tcount1 = pd.DataFrame({'Count': col_count1, 'Percentage': col_perc1})
        
        col_count2 = df[[col_name,countsplit]].groupby(by = [col_name,countsplit]).size()
        col_perc2 = col_count2.apply(lambda x: x / sum(col_count2) * 100).round(1)
        tcount2 = pd.DataFrame({'Count': col_count2, 'Percentage': col_perc2})
        display(tcount1, tcount2) 
        
        figc, axc = plt.subplots(1, 2, figsize = (10,4))
        sns.countplot(data = df, x = col_name, hue = None, ax = axc[0])
        sns.countplot(data = df, x = col_name, hue = countsplit, ax = axc[1])
        
        figb, axb = plt.subplots(1, 2, figsize = (10,4))
        sns.barplot(data = df, x = col_name, y = 'Survived', hue = None, ax = axb[0])
        sns.barplot(data = df, x = col_name, y = 'Survived', hue = barsplit, ax = axb[1])
        
    elif (countsplit != None) & bar:
        col_count1 = df[[col_name]].groupby(by = col_name).size()
        col_perc1 = col_count1.apply(lambda x: x / sum(col_count1) * 100).round(1)
        tcount1 = pd.DataFrame({'Count': col_count1, 'Percentage': col_perc1})
        
        col_count2 = df[[col_name,countsplit]].groupby(by = [col_name,countsplit]).size()
        col_perc2 = col_count2.apply(lambda x: x / sum(col_count2) * 100).round(1)
        tcount2 = pd.DataFrame({'Count': col_count2, 'Percentage': col_perc2})
        display(tcount1, tcount2)
        
        fig, axes = plt.subplots(1, 3, figsize = (15,4))
        sns.countplot(data = df, x = col_name, hue = None, ax = axes[0])
        sns.countplot(data = df, x = col_name, hue = countsplit, ax = axes[1])
        sns.barplot(data = df, x = col_name, y = 'Survived', hue = None, ax = axes[2])
        
    elif countsplit != None:
        col_count1 = df[[col_name]].groupby(by = col_name).size()
        col_perc1 = col_count1.apply(lambda x: x / sum(col_count1) * 100).round(1)
        tcount1 = pd.DataFrame({'Count': col_count1, 'Percentage': col_perc1})
        
        col_count2 = df[[col_name,countsplit]].groupby(by = [col_name,countsplit]).size()
        col_perc2 = col_count2.apply(lambda x: x / sum(col_count2) * 100).round(1)
        tcount2 = pd.DataFrame({'Count': col_count2, 'Percentage': col_perc2})
        display(tcount1, tcount2)
        
        fig, axes = plt.subplots(1, 2, figsize = (10,4))
        sns.countplot(data = df, x = col_name, hue = None, ax = axes[0])
        sns.countplot(data = df, x = col_name, hue = countsplit, ax = axes[1])
        
    else:
        col_count = df[[col_name]].groupby(by = col_name).size()
        col_perc = col_count.apply(lambda x: x / sum(col_count) * 100).round(1)
        tcount1 = pd.DataFrame({'Count': col_count, 'Percentage': col_perc})
        display(tcount1)        
        
        sns.countplot(data = df, x = col_name)


# Use `count_n_plot` to see the overall survival rate (which is not an actual rate but a percentage):

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Survived')


# Only around 40% of people survived. Is there a way to find out what kinds of people are more likely to survive? Are there special features shared by survivors? To answer these questions, we need to look at the other columns and their relationships to the Survived column.
#   
# The PassengerId column is just another index column on top of the original index starting from 0, so it cannot influence a passenger's survival. 
#   
# Observe the Pclass column. There are three categories in this column: 1st Class, 2nd Class and 3rd Class passengers represented by the numbers 1, 2 and 3, respectively. We would expect 1st Class passengers to be given priority in boarding the ship and perhaps priority in being saved in the lifeboats. Hence, we could expect a higher survival rate of 1st Class passengers compared to 2nd or 3rd Class. 
#   
# We use `count_n_plot` to explore the number of passengers that survived in each class (by splitting the Pclass column based on the Survived column):

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Pclass', countsplit = 'Survived')


# The majority of passengers were 3rd Class, making up 55% of the passengers. Interestingly, there were more 1st Classes (216 passengers) than 2nd Classes (184 passengers) although the difference is not too much. This would make sense because the Titanic was designed to be a luxurious cruise ship for to accomodate the 1st class passengers.
# 
# Countplots are useful for displaying the counts but if we want to see the fraction/percentage of survivors based on a particular feature, we use the **barplot**, which is already integrated into the `count_n_plot` function:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Pclass', countsplit = 'Survived', bar = True)


# The barplot indicates above 60% survival rate for 1st Class passengers, below 50% for 2nd Classes and only about 25% for 3rd Classes, even though they were the majority. This is a non-rigorous, informal study using **data visualization** tools to confirm that survival rate is linked to passenger class.
#   
# **Note:** Care must be taken to interpret the barplot because the sum of percentages do not add up to 100%. The percentages only indicate the percentage of survivors for each passenger class.

# We conduct a similar investigation for the Sex column. The Wikipedia article on the RMS Titanic said "the 'women and children first' protocol was generally followed when loading the lifeboats" so we could expect a higher survival rate for women and children. Invoke `count_n_plot` to see this:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Sex', countsplit = 'Survived', bar = True)


# Although 65% of passengers were male (seen from the leftmost countplot), the majority of females survived, at a survival rate of 75% (from the barplot on the right) compared to the male survival rate at 20%. This confirms another link to the target column Survived: females were more likely to survive than males. To go even further, we can split the Sex columns based on Pclass by specifying the `barsplit` argument in `count_n_plot`:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Sex', countsplit = 'Pclass', bar = True, barsplit = 'Pclass')


# This barplot brings us more insights: 1st and 2nd Class females were twice as likely to survive than 3rd Class females (or males of any class), and the least likeliest passengers to survive are 2nd and 3rd Class males. 
#  
# **Note:** When interpreting barplots showing fractional numbers/probabilities such as this, it should be kept in mind that counts and probabilities are different from each other. For instance, consider a hypothetical case in which there are 100 females and 1000 males. The expected survival rates are 75% for females and 20% for males, indicating that about 75 females and 200 males survived. A higher count of males survived even though the survival percentage for females is higher.

# The last categorical feature we can look at is the port of embarkment in the Embarked column. Passengers embarked from three different ports named Cherbourg, Queenstown and Southampton, abbreviated with the letters C, Q and S, respectively. We have already seen that Sex and Pclass influence the survival rate of passengers. Does Embarked influence one's survival? Personally, I don't expect the port of embarkment to be a major link to survival rate but we can still use `count_n_plot` to find out:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Embarked', countsplit = 'Survived', bar = True)


# From the left countplot (and table), over 72% of passengers embarked from S(outhampton), 19% from C(herbourg) and 9% from Q(ueenstown). However, the survival rate was highest at 55% for Cherbourg, around 40% for Queenstown and 34% for Southampton. We can go further by splitting the Embarked countplots and barplots on other features such as Sex and Pclass. First, we see how many males and females are from each port and their survival rates:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Embarked', countsplit = 'Sex', bar = True, barsplit = 'Sex')


# The second countplot (1, 2) split by Sex has a very similar shape to the countplot split by Survived (from the previous `count_n_plot` code). In both cases, the ratio of Survived to Not Survived and Female to Male is roughly similar for the Cherbourg and Queenstown ports but markedly different for Southampton. There were more than twice as many males compared to females from Southampton, which explains why the overall survival rate for Southampton is much lower, at around 34%.
#     
# In the last barplot (2, 2), survival rates for females are around the expected 75% except for Cherbourg, where female survival rate is around 85%. For males, the unexpected survival rates are at 30% for Cherbourg and about 5% for Queenstown, compared to the expected 20%. Perhaps these anomalies can be explained by looking at column splits based on Pclass:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Embarked', countsplit = 'Pclass', bar = True, barsplit = 'Pclass')


# Indeed, 50% of passengers who embarked from Cherbourg were in first class, whereas 72/77 passengers from Queenstown were in third class, thus explaining the slight deviation from the expected 20% survival rate.

# So far, we have studied 4/7 categorical columns (the remaining three are Name, Ticket, and Cabin, which are textual and alphanumeric data), so we now turn towards numerical data (SibSp and Parch, excluding the continuous variables Age and Fare) to see their distribution of values. 
#   
# The column SibSp specifies the number of siblings/spouses on board the Titanic that are related to the passenger. Does having siblings/spouses on the trip influence the survival of the passenger? We find out by investigating this column:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'SibSp', countsplit = 'Survived', bar = True, barsplit = 'Sex')


# The first plot (1,1) indicates that nearly 70% of passengers had no siblings/spouses with them, and the rest had one or more. The second plot (1,2) counts the survived number of passengers, and the third plot (2,1) puts those counts as a fraction. Passengers having one or two sibling/spouse had a higher survival rate (around 55%) than those with no sibling/spouse (35%). The last plot (2,2) is the most interesting, since we can make some conjectures based on the Sex of the passenger and whether they had a spouse or not.
#    
# In the last plot, using the expectation of 75% female survival and 20% male survival rates, we can see that 0 and 2 SibSp categories roughly align with this benchmark. The sample size for 3 or more SibSp is too small, so these categories need not adhere to the expection. The interesting bit is the higher than expected 30% survival rate for male passengers with 1 SibSp. This could occur, for instance, when both the man and his wife are 1st class passengers on the trip and they board the lifeboats together.

# The column Parch specifies the number of parents/children related to the passenger that are also on board the Titanic. If the evacuation protocol "women and children first" was followed, then naturally, the mother and child will board the lifeboats together. We use `count_n_plot` to see any interesting information about the column:

# In[ ]:


count_n_plot(df = titanic_train, col_name = 'Parch', countsplit = 'Survived', bar = True, barsplit = 'Sex')


# Again, similar to the SibSp column, passengers with no parents/children were less likely to survive (around 35% as seen from the third plot (2,1)) than those with one or two parents/children (around 55%). From the last plot (2,2), males with one or two parents/children also had a higher survival rate (30%) compared to those with no parents/children.

# # Data Preprocessing
# 
# From the EDA (Exploratory Data Analysis), we should have identified which variables are missing. Now, in the data preprocessing stage, we can start dealing with them. The percentage of missing data values can be calculated using the `.isnull()` method combined with `.sum()` and dividing by the number of rows in the dataset:

# In[ ]:


titanic_train.isnull().sum()*100 / len(titanic_train)


# Only 3 columns out of 12 contain missing data. The Age column contains about 20% missing data, which can be dealt with using a technique called **imputation**, which means replacing the missing values with a known value, such as the mean, median or mode. Age is quantitative, so either mean or median imputation can be done. Embarked is a categorical variable, so mode imputation can be done. 
#  
# As for the Cabin variable containing 77% missing data, we can simply delete the column. More sophisticated methods would be to see how Cabin numbers relate to the Survived or Pclass column and deal with it accordingly. For example, some of the cabins could be located near where the lifeboats are stored, so passengers residing in those cabins could have a higher chance of survival. 
#  
# Cabin, Ticket and Name are alphanumeric variables, so extracting the important numbers or letters from them require more involved analysis, which will not be done in this Beginner's Analysis.

# ## Mean Imputation
# 
# Inspect the passengers with missing age data:

# In[ ]:


titanic_train[titanic_train['Age'].isnull()]


# The code `titanic_train['Age'].isnull()` produces a list of True and False boolean values. If the Age value is missing, then the boolean is True, and vice versa. This boolean list is used as **conditional selection** to produce a subset of the dataset in which only passengers with missing Age data are shown. 

# An easy way to impute missing values is by using the `.replace()` method. First, we calculate the mean Age of the passengers using the available non-missing data (rounding it off to one decimal place):

# In[ ]:


mean_age = round(titanic_train['Age'].mean(), 1)
print(mean_age)


# Then, we specify the arguments of the replace method. The syntax is `.replace(old_value, new_value)` and the argument `inplace=True` makes the replacement permanent. 

# In[ ]:


titanic_train['Age'].replace(np.nan, mean_age, inplace=True)


# Finally, we check if some of the missing values has been replaced correctly:

# In[ ]:


# iloc means index location
titanic_train.iloc[[5,19,28,863,878],:]


# ## Mode Imputation
# 
# The same procedure can be followed for mode imputation. We simply use the `.mode()` method to get the mode of the Embarked column and replace the missing values with the mode.

# In[ ]:


mode_embarked = titanic_train['Embarked'].mode()[0] 
print(mode_embarked) 


# In the first line, Python still thinks the mode-aggregated object is a DataSeries (one column of a DataFrame), so we need to select the string inside the DataSeries, hence the `[0]`.

# In[ ]:


print(type(titanic_train['Embarked'].mode()))
print(type(titanic_train['Embarked'].mode()[0]))


# In[ ]:


# look at missing values for Embarked column
titanic_train[titanic_train['Embarked'].isnull()]


# In[ ]:


# replace them with mode
titanic_train['Embarked'].replace(np.nan, mode_embarked, inplace=True)


# In[ ]:


# check if replaced correctly
titanic_train.iloc[[61,829],:]


# ## Deletion
# 
# The remaining column with missing data is Cabin (the cabin number). This column contains 77% missing data, so the easiest method would be to get rid of it entirely. Columns can be deleted using the `.drop()` method:

# In[ ]:


titanic_train.drop(columns='Cabin', inplace=True)


# In[ ]:


# Check new dataset
titanic_train.head()


# Additionally, PassengerId, Name, Ticket and Embarked are going to be irrelevant for our simple analysis, so we drop those as well.

# In[ ]:


titanic_train.drop(columns=['PassengerId','Name','Ticket','Embarked'], inplace=True)


# ## One-Hot Encoding
# 
# Inspect the remaining columns:

# In[ ]:


titanic_train.head()


# We now have 5 numerical columns and one categorical column (Sex). However, we need to realize that Pclass is an ordinal categorical variable, with 1st class having a higher status than 2nd class, and so on. These categorical varibales need to be changed into numerics because the machine learning algorithm can only understand numbers.
# 
# In the case of Pclass, the classes are already represented as numbers but in the reverse order. 3rd class is represented as 3 even though it signifies a lower status than 1st class, which is represented as 1. We could reverse the order and make 3rd class 1 and 1st class 3 but it would be misleading and confusing to interpret. 
# 
# Representing ordinal categorical variables as integers depending on the order of importance assigned to them is known as **label encoding**. Label encoding can also be used for nominal variables which has no inherent order, for example, Red, Green, Blue being encoded as 1, 2, 3. The disadvantage is that the machine learning algorithm would misinterpret Blue to have a higher quantitative weight than Red even though they are supposed to be equally important.
# 
# Therefore, we use **one-hot encoding** for Pclass and Sex. One-hot encoding separates categories into binary values of 0 and 1. This is best explained through writing Python code:

# In[ ]:


titanic_train = pd.get_dummies(titanic_train, columns=['Sex'])


# In[ ]:


titanic_train.head()


# Consider the Sex column first. Initially, it contains the categories "Male" and "Female", specifying the sex of the passenger. We would like to encode these categories as numbers instead of letters so we apply the pandas method `.get_dummies()` onto the Sex column. After applying the `.get_dummies()` method, we see two new columns Sex_female and Sex_male, and the original Sex column has disappeared. 
# 
# In the Sex_male column, if the passenger is male, then he is encoded as 1 and if not she is encoded as 0. The same thing is repeated for the Sex_female column. However, this repetition is undesirable to have because all the required information is already captured within one column. Either keep the Sex_male column and drop the Sex_female, or keep the Sex_female and drop the Sex_male.

# In[ ]:


# Drop Sex_female
titanic_train.drop('Sex_female', axis=1, inplace=True)


# `axis=1` specifies that a column is being dropped. If we want to drop rows, we specify `axis=0`.

# In[ ]:


titanic_train.head()


# We perform the same **dummification** process of getting dummy columns (the Sex_male and Sex_female are called dummy variables, which are obtained from the original Sex column) for the Pclass column. This time, we add an additional argument `drop_first=True` to the `get_dummies()` method to drop one irrelevant column:

# In[ ]:


titanic_train = pd.get_dummies(titanic_train, columns=['Pclass'], drop_first=True)


# In[ ]:


titanic_train.head()


# In the Pclass column, we had 3 categories: first, second and third class passengers. One-hot encoding for 3 categories works like this: if the passenger is in 1st class, `Pclass_1 = 1` and `Pclass_2 = Pclass_3 = 0`. If the passenger is in 2nd class, `Pclass_2 = 1` and `Pclass_1 = Pclass_3 = 0`, and similarly for 3rd class passengers. 
# 
# In this case, all the information is captured in two columns (the irrelevant column was already dropped by specifying the `drop_first=True` argument in the previous line of code). Likewise, if we have 4 categories in a column, we create 3 dummies and drop one, and so on. 

# ## Preprocessing Test Dataset
# 
# 

# At this stage, the train dataset, with all its numerical columns, is ready to use as input for the ML algorithm. However, we still need to deal with the test dataset, which will be done in this section.
# 
# **Read:**

# In[ ]:


# Read in the test dataset
titanic_test = pd.read_csv('../input/titanic/test.csv')


# **Inspect:**

# In[ ]:


# Inspect the test dataset
titanic_test.head()


# In[ ]:


titanic_test.info()


# In[ ]:


# Check for null values
titanic_test.isnull().sum()


# **Clean:**

# In[ ]:


# Cleaning Age column
mean_age_test = titanic_test['Age'].mean()
print(mean_age_test)


# In[ ]:


titanic_test['Age'].replace(np.nan, mean_age_test, inplace=True)


# In[ ]:


# Cleaning Fare column
mean_fare_test = titanic_test['Fare'].mean()
print(mean_fare_test)


# In[ ]:


titanic_test['Fare'].replace(np.nan, mean_fare_test, inplace=True)


# In[ ]:


# Remove irrelevant columns but keep a copy of PassengerId column
eye_dee = titanic_test['PassengerId']
titanic_test.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'], inplace=True)


# We keep the PassengerId column for submitting predicted results to Kaggle. Inspect the cleaned dataset and make sure no null values remain:

# In[ ]:


titanic_test.head()


# In[ ]:


titanic_test.isnull().sum()


# **Encode:**

# In[ ]:


# One-hot encoding Pclass and Sex
titanic_test = pd.get_dummies(data=titanic_test, columns=['Pclass','Sex'], drop_first=True)


# In[ ]:


titanic_test.head()


# # Predictive Analysis
# 
# In order to make predictions about the survival rate, we first need to separate the train dataset into independent variables (all the columns except Survived) and the dependent variable (the target column Survived). The test dataset does not contain the Survived column because we are supposed to predict it. Next, we choose a machine learning algorithm and train it using the train dataset. Finally, we ask it to make predictions for the target column using the test dataset. 
# 
# Separate train dataset into independent and dependent variables:

# In[ ]:


# independent varibles, represented by a capital X; dependent variables represented by lowercase y
# [:,1:] means select all rows, and columns from 1st column onwards
X_train = titanic_train.iloc[:,1:]
y_train = titanic_train['Survived']
X_test = titanic_test


# A simple and easily interpretable model to use would be **logistic regression**. Import the LogisticRegression library:

# In[ ]:


from sklearn.linear_model import LogisticRegression


# Here, we have only imported the library. We still need to create an object/function that can be used on the dataset. This is called **instantiation**, or creating an instance of the LogisticRegression function. Create an instance:

# In[ ]:


# Specify the optimisation algorithm as 'lbfgs' (to silence the warning)
logistic_model = LogisticRegression(solver='lbfgs')


# Train model on X_train and y_train using the method `.fit()`:

# In[ ]:


logistic_model.fit(X_train, y_train)


# Make predictions on X_test using the method `.predict()`:

# In[ ]:


pred = logistic_model.predict(X_test)


# In[ ]:


pred


# In[ ]:


len(pred)


# In[ ]:


type(pred)


# These are the predictions of survival made by the algorithm for each of the 418 passengers in the test dataset. The values are within a numerical array, so we need to put them into a data series (one column of a data frame) before attaching it to the PassengerId column:

# In[ ]:


predictions = pd.Series(data=pred, name='Survived')


# In[ ]:


# The prediction values are inside a data series
type(predictions)


# In[ ]:


predictions.head()


# Now, we attach the PassengerId column with the predictions and export to Excel as a csv file. We use the `.concat()` method from Pandas to stick together (or concatenate) two data frame columns:

# In[ ]:


sub = pd.concat([eye_dee, predictions], axis=1)


# In[ ]:


sub.head()


# Export as a csv file using the method `.to_csv()`

# In[ ]:


# Specify index=False so we don't get the index column when exporting to excel
submission1 = sub.to_csv('submission1.csv', index=False)


# Download the file and submit predictions on the main competition page.
# 
# **Submission1 Accuracy: 70.3%**
