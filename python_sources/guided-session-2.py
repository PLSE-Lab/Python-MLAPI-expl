#!/usr/bin/env python
# coding: utf-8

# ## Guided session 2
# 
# This is the first notebook for the second session of the [Machine Learning workshop series at Harvey Mudd College](http://www.aashitak.com/ML-Workshops/).
# 
# Main topics for today's session:
# * Split-apply-combine operations by grouping rows of a dataframe
# * Encoding categorical variables
# * Concatentating and merging dataframes

# In[ ]:


import pandas as pd
import re # For regular expressions


# In today's guided session, we will continue exploring the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic). Let us set *Passengerid* as the index.

# In[ ]:


path = '../input/'
df = pd.read_csv(path + 'train.csv') 
df = df.set_index('PassengerId')
df.head()


# ### 1. [GroupBy object](https://pandas.pydata.org/pandas-docs/version/0.22/groupby.html)
# In the last exercise session, we noticed the *Age* column has a lot of missing values. To fill these values, we can group the passengers based on the titles derived from their name and then take the median value from each group to fill the missing values of the group.
# 
# The below code is a repetition from the exercises in the previous session to create a new column named *Title* from the *Name* column using regular expressions. 

# In[ ]:


df['Title'] = df['Name'].map(lambda name: re.findall("\w+[.]", name)[0])

title_dictionary = {'Ms.': 'Miss.', 'Mlle.': 'Miss.', 
              'Dr.': 'Rare', 'Mme.': 'Mr.', 
              'Major.': 'Rare', 'Lady.': 'Rare', 
              'Sir.': 'Rare', 'Col.': 'Rare', 
              'Capt.': 'Rare', 'Countess.': 'Rare', 
              'Jonkheer.': 'Rare', 'Dona.': 'Rare', 
              'Don.': 'Rare', 'Rev.': 'Rare'}

df['Title'] = df['Title'].replace(title_dictionary)

df.head()


# We can use [`groupby()`](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.groupby.html) to group the rows of the dataframe based on column(s), say *Title*, but we need to apply some operation on the grouped object to derive a dataframe.

# In[ ]:


df.groupby('Title')


# One of the ways to derive a dataframe from a groupby object is by aggregation, that is computing a summary statistic (or statistics) about each group. For example, we can get the median values for the columns in each group of titles.

# In[ ]:


df.groupby('Title').median()


# In[ ]:


df.groupby('Title').mean()


# The median age vary greatly for each group ranging from 3.5 to 48 years.

# The most common way to derive a dataframe from a groupby object is by transformation. We create a new column *MedianAge* which consists of the groupwise median age depending on the passengers' title using [`transform()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transform.html).

# In[ ]:


df['MedianAge'] = df.groupby('Title')['Age'].transform("median")
df.head(15)


# Now we fill in the missing values in the *Age* column using the values in the *MedianAge* column.

# In[ ]:


df['Age'] = df['Age'].fillna(df['MedianAge'])
df.head()


# We drop off the *MedianAge* column since we no longer need it.

# In[ ]:


df = df.drop('MedianAge', axis=1)
df.head()


# Let us check for the missing values. There are none in the *Age* column!

# In[ ]:


df.isnull().sum()


# ### 2. Encoding categorical variables
# Let us check the datatype of each column. Hint: Use `dtypes`.

# In[ ]:


df.dtypes


# There are two columns with `object` datatype - *Sex* and *Embarked*. These two along with *Pclass* are categorical variables. The feature *Pclass* has an innate order in its categories and hence, is ordinal, whereas *Sex* and *Embarked* are inordinal categorical variables. Most machine learning models require the features or input variables to be numerical. One way to accomplish that is to encode the categories with numbers.

# Convert the gender values to numerical values 0 and 1. Hint: Use `replace` with the suitable dictionary. 

# In[ ]:


df['Sex'].value_counts()


# In[ ]:


df = df.replace({'male': 0, 'female': 1})


# Check the datatypes again and make note of datatype for the column *Sex*. Discuss what can possibly go wrong with randomly assigning numbers to categories.

# In[ ]:


df.dtypes


# In[ ]:


df['Embarked'].value_counts()


# Numbers have a natural order and so do ordered categories such as passengers' ticket class in our case. Number also have  an inherent quantitive value attached to them that categories do not. For example, the difference between the numbers 1 and 2 is the same as the difference between the numbers 2 and 3 but the same cannot be said for ordinal categories. So, converting categories to numbers means adding untrue assumptions that may or may not adversely affect our model. 
# 
# For this reason, the prefered method is one-hot encoding. In this method, we build a one-hot encoded vector with dimension equal to the number of classes in the categories. This vector consists of all 0's except for a 1 corresponding to the class of the instance. For example, the *Embarked* column will have one-hot encoded vectors of [1,0,0], [0,1,0] and [0,0,1] for the three ports.
# 
# One-hot encoding is accomplished in pandas using `get_dummies` as given below. It simply creates a column for each class of a categorical variable.

# In[ ]:


pd.get_dummies(df['Embarked']).head()


# We want the column names to be `'Port_C', 'Port_Q', 'Port_S'`. Copy the above code with `get_dummies` and modify it to [make use of the `prefix` keyword](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) to alter the column names. Next, save this to a new dataframe named `port_df`.

# In[ ]:


port_df = pd.get_dummies(df['Embarked'], prefix='Port')


# In[ ]:


port_df.head()


# To add this dataframe of two new columns to the original dataframe, we can use `concat` with `axis=1`.

# In[ ]:


df = pd.concat([df, port_df], axis=1)


# Now check that the new columns are added. 

# In[ ]:


df.head()


# Next, drop the column for *Embarked*. 

# In[ ]:


df = df.drop('Embarked', axis=1)


# Note: if you run the above cell more than once, it will give an error, since the column *Embarked* is no more present in the dataframe for the code to work. 

# Next, we check the columns in our dataframe.

# In[ ]:


df.columns


# The expected output is  
# ```Index(['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Port_C', 'Port_Q', 'Port_S'], dtype='object')```

# Notes:
# - One of the columns in the one-hot encoding obtained in the above manner is always redundant. In case of features with just two classes such as gender in our dataset, one-hot encoding is not truly useful. One of its column is same as what we obtained by simply replacing classes with 0 and 1 and the other is redundant.  
# - The main disadvantage of using one-hot encoding is the increase in the number of features that can negatively affect our model which we will discuss in the later sessions.
# 

# ### Acknowledgement:
# * [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic) dataset openly available in Kaggle is used in the exercises.
# 
# **Note:**
# The solutions for this exercise can be found [here](https://github.com/AashitaK/ML-Workshops/blob/master/Session%202/Guided%20session%202.ipynb).

# ### Next step:
# 
# Please proceed to the [hands-on exercises](https://www.kaggle.com/aashita/exercise-2).

# In[ ]:




