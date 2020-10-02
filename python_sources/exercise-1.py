#!/usr/bin/env python
# coding: utf-8

# ## Exercise notebook for the first session (60 min)

# This is the exercise notebook for the first session of the [Machine Learning workshop series at Harvey Mudd College](http://www.aashitak.com/ML-Workshops/). It involves new concepts not covered in the guided session. You are encouraged to work in groups of 2-4 if that helps to speed things up. Please ask for help from the instructor and/or TAs. The session is time-bound, so make sure you are not stuck at a problem for too long before asking for help.  

# First we import the relevant python modules:

# In[ ]:


import numpy as np
import pandas as pd

# The following two modules matplotlib and seaborn are for plots
import matplotlib.pyplot as plt
import seaborn as sns # Comment this if seaborn is not installed
get_ipython().run_line_magic('matplotlib', 'inline')

# The module re is for regular expressions
import re


# In this exercise session, we will explore the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic).

# In[ ]:


# Uncomment the below two lines only if using Google Colab
# from google.colab import files
# uploaded = files.upload()
df = pd.read_csv('../input/train.csv')
df.head()


# ### 1. Exploring the dataset (25 min)
# 
# Use `describe` function for numerical features (columns) to get a brief overview of the statistics of the data.

# In[ ]:


df.describe()


# Do the same as above for qualitative (non-numerical) features. Hint: Use `include='O'` parameter in the `describe` function.

# In[ ]:





# Use the functions `isnull()` and `sum()` on the dataframe to find out the number of missing values in each column.

# In[ ]:


df.isnull().sum()


# Some questions to consider:
# 1. Suppose the final goal is to design a model to predict whether a passenger survives or not, which of these features(columns) seems like important predictors? How can you analyse the data in view of this objective?
# 2. What are the possible ways to understand the correlation of features with survival? Does correlation always implies causation? 
# 
# Detecting missing values is an important first step in Feature Engineering, that is preparing the features (independent variables) to use for building the machine learning models. The next step is to handle those missing values. Depending on the data, sometimes it is a good idea to drop the rows or columns that have some or a lot of missing values, but that also means discarding relevant information. Another way to handle missing values is to fill them with something appropriate. 
# 
# 3. Discuss the pros and cons of dropping the rows and/or columns with missing values in general. Should you drop none, all or some of the columns for this particular dataset in view of building the predictive model? Same question for dropping the rows with missing values.
# 3. If you consider filling the missing values, what are the possible options? Can you make use of other values in that column to fill the missing values? Can you make use of other values in that row as well as values in that column to fill the missing values 
# 4. Can the title in the name column be used for guessing a passengers' age based on the age values of other passengers with the same title?

# Make the `PassengerId` column the index of the Data Frame. Hint: Use `set_index()`.

# In[ ]:





# Check again whether the index has been changed in the original dataframe. 

# In[ ]:


df.head()


# If not, there are two options to fix this. One is to set `inplace` parameter in the `set_index()` function as `True` and another is to use assignment operator `=` as in `df = df.function()`. 
# 
# ***Question***: Why is the `inplace` keyword False by default? This is true not just for `set_index()` but for most built-in functions in pandas. 
# 
# Answer: To facilitate method chaining or piping i.e. invoking multiple operations one after the other. For example, `df.isnull().sum()` used above. Chaining is more commonly used in pandas as compared to another programming style i.e. using nested function calls. Please read more [here](https://towardsdatascience.com/the-unreasonable-effectiveness-of-method-chaining-in-pandas-15c2109e3c69), if interested.

#  Use the built-in pandas function to count the number of surviving and non-surviving passengers. Hint: Use `value_counts()` on the column `df['Survived']`.

# In[ ]:





# Below is a pie chart of the same using `matplotlib`:

# In[ ]:


plt.axis('equal')
plt.pie(df['Survived'].value_counts(), labels=('Died', "Survived"));


# Below is a bar chart for the survival rate among male and female passengers using `seaborn`. Here is [Seaborn cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf).

# In[ ]:


sns.barplot(x = 'Sex', y = 'Survived', data = df);


# Plot the survival rate among passengers in each ticket class.

# In[ ]:





# We can also check the survival rate among both genders within the three ticket classes as follows.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df);


# From the above chart, do you think that the gender affect the chance of survival for all the three ticket classes equally? Or does it seem like gender's effect is more pronounced for a certain ticket class passengers than others? We plot the  point estimates and confidence intervals for each sub-category to see it more clearly.

# In[ ]:


sns.pointplot(x='Sex', y='Survived', hue='Pclass', data=df);


# Notice the steeper slope for the second class.

# It seems that gender and ticket class put together give more information about the survival chance than both of them separately. Please feel free to later explore other variables and combination of variables in depth in your own time.

# How many children were on board?

# In[ ]:





# How many of the children on board survived?

# In[ ]:





# What is the most common port of embarkment? Hint: Check the frequency (counts) of each value in the Embarked column using an built-in function. 

# In[ ]:





# As we saw earlier, there are missing values in the column for Embarked. Fill them with the most commonly occuring value. Hint: Use `fillna()`.

# In[ ]:





# We should remove the *Cabin* column from the DataFrame-- too many values are missing. Hint: Use `drop()` with appropriate value for the `axis` keyword. 

# In[ ]:





# Let us check whether the column is indeed dropped. If not, modify the code above accordingly.

# In[ ]:


df.head()


# We check again to see the missing values in the DataFrame. 

# In[ ]:


df.isnull().sum()


# ### 2. Feature Engineering: Creating a new column for the titles of the passengers (20 min)
# 
# Now, we are going to create a new feature (column) for titles of the passengers. For that, let us first take at the passengers' names. 

# In[ ]:


df.loc[:20, 'Name'].values


# We notice one of the identifying characteristics of the titles above are that they end with a period. Regular expressions are very useful in the process of data extraction and we will use them using the python module `re` to extract the titles from the *Name* column. We will use regular expressions characters to construct a pattern and then use built-in function `findall` for pattern matching.
# 
# Some useful regular expression characters:
# - `\w`: pattern must contain a word character, such as letters.
# - `[ ]`: pattern must contain one of the characters inside the square brackets. If there is only one character inside the square brackets, for example `[.]`, then the pattern must contain it.
# 
# Let's try this.

# In[ ]:


re.findall("\w\w[.]", 'Braund, Mr. Owen Harris')


# It worked! let us try it on another name:

# In[ ]:


re.findall("\w\w[.]", 'Heikkinen, Miss. Laina')[0]


# So, we want a pattern that automatically detects the length of the title and returns the entire title.
# 
# For regular expressions, \+ is added to a character/pattern to denote it is present one or more times. For example, `\w+` is used to denote one or more word characters. Fill in the regular expression in the below cell that will detect a period preceeded by one or more word characters.

# In[ ]:


# Fill in below:
re.findall("FILL IN HERE", 'Heikkinen, Miss. Laina')[0]


# The output should be `'Miss.'`

# Summary: For pattern matching the titles using regular expressions:
# - First we make sure it contains a period by using `[.]`. 
# - Secondly, the period must be preceeded by word characters (one or more), so we use `\w+[.]`.
# 
# Write a function `get_title` that takes a name, extracts the title from it and returns the title.

# In[ ]:





# Check that the function is working properly by running the following two cells.

# In[ ]:


get_title('Futrelle, Mrs. Jacques Heath (Lily May Peel)')


# The output should be `'Mrs.'`. Note: Make sure that the funtion returns a string and not a list. Please modify the above function accordingly.

# In[ ]:


get_title('Simonius-Blumer, Col. Oberst Alfons')


# The output should be `'Col.'`.

# Create a new column named Title and extract titles from the Name column using the above function `get_title`. Hint: Use built-in `map()` function. The syntax is `df['New_column'] = df['Relevant_column'].map(function_name)`.

# In[ ]:





# In[ ]:


df.head()


# List all the unique values for the titles along with their frequency. Hint: Use an inbuilt pandas function

# In[ ]:





# Now, we want to replace the various spellings of the same title to a single one. Hint: Use the below dictionary with the `replace` function
# 
# `title_dictionary = {'Ms.': 'Miss.', 'Mlle.': 'Miss.', 
#               'Dr.': 'Rare', 'Mme.': 'Mr.', 
#               'Major.': 'Rare', 'Lady.': 'Rare', 
#               'Sir.': 'Rare', 'Col.': 'Rare', 
#               'Capt.': 'Rare', 'Countess.': 'Rare', 
#               'Jonkheer.': 'Rare', 'Dona.': 'Rare', 
#               'Don.': 'Rare', 'Rev.': 'Rare'}`

# In[ ]:





# List all the unique values for the titles along with their frequency to check that the titles are replaced properly.

# In[ ]:





# ### 3. More Feature Engineering: Working on the Age column  (10 min)
# 
# What is the age of the oldest person on board? 

# In[ ]:





# Find all the passenger information for the oldest person on board. Hint: Use `loc[]` method with `idxmax()` for the Age column.

# In[ ]:





# What is the average age of the passengers?

# In[ ]:





# What is the median age of the passengers?

# In[ ]:





# Discuss with your team to come up with a single best approximation to fill in the missing values for the Age column and then write the code to fill them. Hint: Use `fillna`.

# In[ ]:


# We first make a copy of the dataframe in case we want 
# to use it later before we fill in missing values
df2 = df.copy() 


# Note: In the next session, we will create a title column and use it to group the passengers into different title-based groups and make use of the groupings to fill the missing age values. 

# What is the median age of passengers with the title 'Miss.'? Hint: Use `loc[]` method for slicing off the select rows and the *Age* column.

# In[ ]:





# What is the median age of passengers with the title 'Mrs.'?

# In[ ]:





# Is there a noticeble difference in the median ages for the passengers with the above two titles? Should we take titles into account while filling the missing values for the *Age* column? If yes, how?
# 
# ***Optional (preferably come back to this at the end)***:
# 1. Find the list of indices of the missing values for the age column using the dateframe `df2`.
# 2. Group the passengers with respect to their titles using `groupby("Title")` and then get the median age of passengers in each group using `transform("median")` on the *Age* column.
# 3. Create a new column *MedianAge* which consists of the groupwise median age depending on the passengers' title.
# 4. Next use this column to fill in the missing values for the age column using `fillna`.
# 5. Finally compare the age column for `df` and `df2` for the list of indices from the first step.

# In[ ]:





# ### 4. Correlation between variables (5 min)
# 
# Pearson correlation coefficients measures the linear correlation between the variables.
# 
# $$\rho_{X,Y} = \frac{cov(X, Y)}{\sigma_X, \sigma_Y}$$
# where 
# - $cov(X, Y)$ is the covariance.    
# - $\sigma_X, \sigma_Y$ are standard deviations of $X$ and $Y$ respectively.
# 
# The correlation between two variables ranges from -1 to 1. The closer in absolute value a correlation is to 1, the more dependent two features are each other.
# 
# Get the correlation matrix for the variables (columns) in the dataset. Hint: Use a built-in function.

# In[ ]:





# * From the above matrix, note which feature has the highest correlation with the survival. 
# * Do features have high correlation among themselves? 
# * Note that this matrix has excluded some categorical variables like gender, port of embarkment, etc. 
# 
# The correlation matrix can also be visualized using heatmaps as shown below.

# In[ ]:


correlation_matrix = df.corr();
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(correlation_matrix);


# In[ ]:


plt.figure(figsize=(14, 10))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df);


# To build a prediction model to classify the passengers is same as drawing a clear boundary separating the orange and blue points. The task seems non-trivial by using only the two numerical features *Age* and *Fare*. In the later sessions, we will build models using other features as well that would classify points with the accuracy of around 80%.

# #### Topics covered in today's session:
# - Reading csv files using `read_csv()`
# - Slicing and indexing dataframes using conditionals as well as `iloc[]` and `loc[]` methods
# - Statistical summary and exploration using `describe()`, `median()`, `mean()`, `idxmax()`, `corr()`, etc.
# - Detecting and filling missing values in the dataset using `isnull()` and `fillna()`
# - Dropping columns using `drop()`
# - Basic operations such as `set_index()`, `replace()`, `value_counts()`, `columns`, `index`, etc.
# - Regular expressions for data extraction
# - Feature engineering such as creating a new feature for titles
# - Some basic plots
# - Correlation among features

# #### Acknowledgment:
# * [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic) dataset openly available in Kaggle is used in the exercises.
# 
# **Note:**
# The solutions for this exercise can be found [here](https://github.com/AashitaK/ML-Workshops/blob/master/Session%201/Exercise%201%20with%20solutions.ipynb).

# In[ ]:




