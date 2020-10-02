#!/usr/bin/env python
# coding: utf-8

# # Python for Data 19: Frequency Tables
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# Discovering relationships between variables is the fundamental goal of data analysis. Frequency tables are a basic tool you can use to explore data and get an idea of the relationships between variables. A frequency table is just a data table that shows the counts of one or more categorical variables.
# 
# To explore frequency tables, we'll revisit the Titanic training set. We will start by performing a couple of the same preprocessing steps from lesson 14:

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


titanic_train = pd.read_csv("../input/train.csv")      # Read the data

char_cabin = titanic_train["Cabin"].astype(str)    # Convert cabin to str

new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

titanic_train["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var


# ## One-Way Tables

# Create frequency tables (also known as crosstabs) in pandas using the pd.crosstab() function. The function takes one or more array-like objects as indexes or columns and then constructs a new DataFrame of variable counts based on the supplied arrays. Let's make a one-way table of the survived variable:

# In[5]:


my_tab = pd.crosstab(index=titanic_train["Survived"],  # Make a crosstab
                     columns="count")                  # Name the count column

my_tab


# In[ ]:


type(my_tab)             # Confirm that the crosstab is a DataFrame


# Let's make a couple more crosstabs to explore other variables:

# In[6]:


pd.crosstab(index=titanic_train["Pclass"],  # Make a crosstab
            columns="count")                # Name the count column


# In[7]:


pd.crosstab(index=titanic_train["Sex"],     # Make a crosstab
                      columns="count")      # Name the count column


# In[12]:


cabin_tab = pd.crosstab(index=titanic_train["Cabin"],  # Make a crosstab
                        columns="count")               # Name the count column

cabin_tab 


# You can also use the value_counts() function to on a pandas series (a single column) to check its counts:

# In[10]:


titanic_train.Sex.value_counts()


# Even these simple one-way tables give us some useful insight: we immediately get a sense of distribution of records across the categories. For instance, we see that males outnumbered females by a significant margin and that there were more third class passengers than first and second class passengers combined.
# 
# If you pass a variable with many unique values to table(), such a numeric variable, it will still produce a table of counts for each unique value, but the counts may not be particularly meaningful.
# 
# Since the crosstab function produces DataFrames, the DataFrame operations we've learned work on crosstabs:

# In[13]:


print (cabin_tab.sum(), "\n")   # Sum the counts

print (cabin_tab.shape, "\n")   # Check number of rows and cols

cabin_tab.iloc[1:7]             # Slice rows 1-6


# One of the most useful aspects of frequency tables is that they allow you to extract the proportion of the data that belongs to each category. With a one-way table, you can do this by dividing each table value by the total number of records in the table:

# In[14]:


cabin_tab/cabin_tab.sum()


# ## Two-Way Tables

# Two-way frequency tables, also called contingency tables, are tables of counts with two dimensions where each dimension is a different variable. Two-way tables can give you insight into the relationship between two variables. To create a two way table, pass two variables to the pd.crosstab() function instead of one:

# In[15]:


# Table of survival vs. sex
survived_sex = pd.crosstab(index=titanic_train["Survived"], 
                           columns=titanic_train["Sex"])

survived_sex.index= ["died","survived"]

survived_sex


# In[16]:


# Table of survival vs passenger class
survived_class = pd.crosstab(index=titanic_train["Survived"], 
                            columns=titanic_train["Pclass"])

survived_class.columns = ["class1","class2","class3"]
survived_class.index= ["died","survived"]

survived_class


# You can get the marginal counts (totals for each row and column) by including the argument margins=True:

# In[17]:


# Table of survival vs passenger class
survived_class = pd.crosstab(index=titanic_train["Survived"], 
                            columns=titanic_train["Pclass"],
                             margins=True)   # Include row and column totals

survived_class.columns = ["class1","class2","class3","rowtotal"]
survived_class.index= ["died","survived","coltotal"]

survived_class


# To get the total proportion of counts in each cell, divide the table by the grand total:

# In[18]:


survived_class/survived_class.loc["coltotal","rowtotal"]


# To get the proportion of counts along each column (in this case, the survival rate within each passenger class) divide by the column totals:

# In[19]:


survived_class/survived_class.loc["coltotal"]


# To get the proportion of counts along each row divide by the row totals. The division operator functions on a row-by-row basis when used on DataFrames by default. In this case we want to divide each column by the rowtotals column. To get division to work on a column by column basis, use df.div() with the axis set to 0:

# In[20]:


survived_class.div(survived_class["rowtotal"],
                   axis=0)


# Alternatively, you can transpose the table with df.T to swap rows and columns and perform row by row division as normal:

# In[21]:


survived_class.T/survived_class["rowtotal"]


# ## Higher Dimensional Tables

# The crosstab() function lets you create tables out of more than two categories. Higher dimensional tables can be a little confusing to look at, but they can also yield finer-grained insight into interactions between multiple variables. Let's create a 3-way table inspecting survival, sex and passenger class:

# In[22]:


surv_sex_class = pd.crosstab(index=titanic_train["Survived"], 
                             columns=[titanic_train["Pclass"],
                                      titanic_train["Sex"]],
                             margins=True)   # Include row and column totals

surv_sex_class


# Notice that by passing a second variable to the columns argument, the resulting table has columns categorized by both Pclass and Sex. The outermost index (Pclass) returns sections of the table instead of individual columns:

# In[23]:


surv_sex_class[2]        # Get the subtable under Pclass 2


# The secondary column index, Sex, can't be used as a top level index, but it can be used within a given Pclass:

# In[24]:


surv_sex_class[2]["female"]   # Get female column within Pclass 2


# Due to the convenient hierarchical structure of the table, we still use one division to get the proportion of survival across each column:

# In[25]:


surv_sex_class/surv_sex_class.loc["All"]    # Divide by column totals


# Here we see something quite interesting: over 90% of women in first class and second class survived, but only 50% of women in third class survived. Men in first class also survived at a greater rate than men in lower classes. Passenger class seems to have a significant impact on survival, so it would likely be useful to include as a feature in a predictive model.

# ## Wrap Up

# Frequency tables are a simple yet effective tool for exploring relationships between variables that take on few unique values. Tables do, however, require you to inspect numerical values and proportions closely and it is not always easy to quickly convey insights drawn from tables to others. Creating plots is a way to visually investigate data, which takes advantage of our innate ability to process and detect patterns in images.

# ## Next Lesson: [Python for Data 20: Plotting With Pandas](https://www.kaggle.com/hamelg/python-for-data-20-plotting-with-pandas)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
