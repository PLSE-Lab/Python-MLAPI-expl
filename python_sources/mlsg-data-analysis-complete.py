#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to work through an example of a data analysis. We will show how we perform some common preprocessing / analysis tasks to gather more information about the data. This will inform and prepare us for the modelling step.
# 
# The data we will be looking at is taken from https://www.kaggle.com/epa/fuel-economy. If it is not already in the */kaggle/input/fuel-economy* folder then you can add it as follows: Go to file --> Add or upload data. Look for "fuel economy" and click on _Add_ next to **Vehicle Fuel Economy Estimates, 1984-2017**.
# 
# The goal with this dataset is to see if we can predict the fuel economy of a car based on some of its characteristics.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')


# We will first read the data in and look at some basic information.

# In[ ]:


# ignore warning about "Dtypes"
cars = pd.read_csv('/kaggle/input/fuel-economy/database.csv')


# In[ ]:


# print out the number of rows and columns (dimensions)
cars.shape


# In[ ]:


# print the names of the columns
cars.columns


# There are too many columns to analyse right now so I'm going to focus on a few. Columns related to MPG all say the same thing so I'll pick one. Also the columns that come after the MPG columns don't look very relevant so we'll skip them as well.

# In[ ]:


cars = cars[['Vehicle ID', 'Year', 'Make', 'Model', 'Class', 'Drive', 'Transmission',
            'Transmission Descriptor', 'Engine Index', 'Engine Descriptor',
           'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
           'Supercharger', 'Fuel Type', 'Fuel Type 1',
            'Combined MPG (FT1)']]
cars.shape


# The easiest way to get to know your data is to have a look at it in its raw form.

# In[ ]:


# print first 5 rows
cars.head()


# In[ ]:


cars.tail()


# Some early observations about the data:
# * Some columns have missing values (NaNs)
# * There is a mix of numeric columns and categorical columns
# * The MPG column is a integer value whereas Engine Cylinders and Engine Displacement are not. Doesn't make sense for Engine Cylinders to be a decimal value, will require further investigation
# * Turbocharger and Supercharger only seem to have two values (missing or T/S)
# * Possible duplication of vehicles (at least when looking at the columns of interest)
# 
# Next we go one step deeper and look at all the data and summaries of the values.

# In[ ]:


cars.info()


# Here we see some columns have a considerable amount of missing values. We will need to consider what we do with these values later one. We also see here confirmed that all values in certain columns are of the same type.

# In[ ]:


cars.describe()


# Looking at summary values is more difficult. What we're looking for is anything out of the ordinary. We can safely ignore the *Vehicle ID* and *Engine Index* values as these don't mean anything. As for *Year* we can confirm that the data is for the period 1984 and 2017. 
# 
# **All prior information about the data is an assumption until confirmed by the data**
# 
# Looking at Engine Cylinders and MPG we may have an outlier that we have to deal with. Outliers are important because they tend to have an outsized effect on the model results. Finally, in the Engine Displacement column we see a minimum of zero which may point to an error.

# ### Duplicates
# 
# Duplicated data can be caused by errors during data entry or during data collection. For example, your SQL query may have included a join which resulted in duplicates. If all records were duplicated then it's not a huge problem other than that you have too much data and training your model may take extra time. However if only some records are duplicated their effect on the model results will be enlarged.
# 
# In this example it's not straightforward that we're dealing with duplicates. Take the first rows:

# In[ ]:


cars.head(2)


# It's the same car except for the Engine Index/Descriptor but that doesn't seem to have an impact on the MPG. However we can't make that conclusion based on only one example. So we want to look at a subset of columns to judge duplicates.

# In[ ]:


id_columns = ['Make', 'Model', 'Class', 'Drive', 'Transmission',
            'Transmission Descriptor', 'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
           'Supercharger', 'Fuel Type', 'Fuel Type 1',
            'Combined MPG (FT1)']
duplicates = cars.duplicated(subset=id_columns)

duplicates.sum()


# In[ ]:


cars[duplicates].head()


# We can drop a few more columns from the identifier columns which will probably result in more duplicates but we'll leave it at that.

# In[ ]:


cars_dedup = cars.drop_duplicates(subset=id_columns)
cars_dedup.shape


# ### Categorical Values
# 
# We summarise categorical values by looking at their counts.

# In[ ]:


cars['Make'].value_counts()


# The thing about _Make_ and _Model_ is that they are very specific, meaning there are a lot of possible values. This makes it hard for any algorithm to extract information that is generally applicable. Just think of the situation a new car maker appears, how should the algorithm deal with that if it hasn't seen it before. Instead it's better to look at variables that do generalize.

# In[ ]:


cars['Class'].value_counts()


# There's a reasonable amount of values here but any algorithm will still have trouble learning anything from the values that only have a handful of observations. Possible solutions are to either ignore or combine these values.

# In[ ]:


# remember Drive had a few missing values
cars['Drive'].value_counts(dropna=False)


# Here where you would go back to the source and ask what exactly the difference is between these values. Perhaps some of these values can be combined.
# 
# Will come back to the missing values.

# In[ ]:


cars['Turbocharger'].value_counts(dropna=False)


# In[ ]:


cars['Supercharger'].value_counts(dropna=False)


# When a column is majority missing values you have to wonder if it is of any use to include it. 

# In[ ]:


cars['Fuel Type'].value_counts()


# Do electric cars do mileage?
# 
# ### Numerical values
# 
# We looked at summary statistics for the numerical values but we can go a little deeper by looking at the distribution of values.

# In[ ]:


cars['Engine Cylinders'].plot.hist()


# Most cars have 4 or 6 cylinders. We see a few that have 12 and 10 but the one that has 16 or 2 cylinders is odd.

# In[ ]:


cars[cars['Engine Cylinders']==16]


# Ok maybe not that odd.

# In[ ]:


cars[cars['Engine Cylinders']==2].head()


# In[ ]:


cars[cars['Engine Cylinders']==2].shape


# In[ ]:


cars[cars['Engine Cylinders']==2]['Make'].unique()


# In[ ]:


cars[cars['Engine Cylinders']==2].tail()


# In[ ]:


cars['Engine Displacement'].plot.hist()


# So most cars are small cars so it makes sense the displacement distribution tends to the left. It is also not weird that there are some large cars in the dataset. I don't really know if a displacement of 8 is unusual. A displacement of 0 is however.

# In[ ]:


cars[cars['Engine Displacement']==0]


# In[ ]:


cars[cars['Engine Displacement']==8].head()


# In[ ]:


cars['Combined MPG (FT1)'].plot.hist()


# In[ ]:


cars[cars['Combined MPG (FT1)']>120]


# Makes sense these are all electric cars. It might be worth removing all electric cars because there is obviously no relationship between MPG and the other variables for this type of car (if that is what we're trying to predict).

# ### Missing Values
# 
# Dealing with missing values is necessary because ML algorithm don't know what to do with _NaN_ values. In most programming languages _NaN_ is a special value that is neither numeric nor character. So it requires manual intervention to make sure _NaN_ values are dealt with correctly.
# 
# There are a number of strategies we can apply here and they are dependent on what the source of the missing values is. There is various research available on this topic, referred to as missing value imputation, but we'll stick to some simple solutions.
# 
# #### Errors
# 
# When missing values are the result of errors, whether in the data gathering part or the data collection part (i.e. wrong query), I think it is best to ignore the records if possible.

# In[ ]:


not_missing_drive = cars['Drive'].notnull()
not_missing_drive.sum()


# #### Laziness
# 
# Sometimes a missing values implies the information is either not known or not applicable. In these case we simply make it explicit by providing a fill value. There are other methods of "filling" the gaps, see the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) for more information.

# In[ ]:


cars['Turbocharger'] = cars['Turbocharger'].fillna('No')
cars['Supercharger'] = cars['Supercharger'].fillna('No')


# ### Outliers
# 
# Outliers can too be caused by errors (e.g. misreading in measurement equipment) or they might not be representative of the relationships we're considering. In either case it's a matter of detecting them and removing them.
# 
# A simple way of detecting outliers is to consider the mean/median and add one or two standard deviations. Anything above these values can be considered an outlier. 

# In[ ]:


no_mpg_outlier = cars['Combined MPG (FT1)'] < (cars['Combined MPG (FT1)'].median() + cars['Combined MPG (FT1)'].std())
no_mpg_outlier.sum()


# In[ ]:


no_electric_cars = cars['Fuel Type'] != 'Electricity'
cars_cleaned = cars[(~duplicates) & (not_missing_drive) & (no_mpg_outlier)]
cars_cleaned = cars_cleaned[['Class', 'Drive', 'Engine Cylinders', 
                             'Engine Displacement', 'Turbocharger','Supercharger', 
                             'Fuel Type', 'Combined MPG (FT1)']]

cars_cleaned.shape


# ## Relationship Analysis
# 
# The above is all to do with cleaning the data to make modeling easier. But before we move on to that stage we can still extract some more information from the dataset by looking at relationships between variables. The best way to do this is through visualisations.

# In[ ]:


import seaborn as sns
sns.pairplot(cars_cleaned)


# We can see a number of things here:
# * There are no categorical values, pairplot only shows the numerical values
# * Pairplot shows the distribution on the diagonal and scatterplots on the off-diagonals
# * There is a relationship between engine displacement and number of cylinders which makes sense
# * The number of cylinders is actually a categorical value
# * There is relationship between displacement and mpg but there is also a lot of noise
# * This relationship is even weaker between cylinders and mpg, possibly because it's not a numerical value
# 
# It is typical that midway through your analysis you uncover some insight that requires going back to the preprocessing/cleaning step.
# 
# ### Boxplots
# 
# When it comes to relationship between categorical values and numerical values, boxplots are the way to go. They display the distribution of the numerical values per category.

# In[ ]:


sns.boxplot(x='Drive', y='Combined MPG (FT1)', data=cars_cleaned)


# In[ ]:


sns.boxplot(x='Turbocharger', y='Combined MPG (FT1)', data=cars_cleaned)


# In[ ]:


sns.boxplot(x='Supercharger', y='Combined MPG (FT1)', data=cars_cleaned)


# In[ ]:


sns.boxplot(x='Engine Cylinders', y='Combined MPG (FT1)', data=cars_cleaned)


# There are various incarnations of the boxplot (violinplot, swarmplot) that may give you more information but in this case that would be overkill. 
