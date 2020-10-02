#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# DATA ENGINEERING OF TITANIC RAW DATA (TEST DATA)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# **THE DATA**

# In[ ]:


## Importing test Data
test=pd.read_csv('test_titanic.csv')
test.head()


# In[ ]:


test.shape


# Exporatory Data Analysis (EDA) on test dataset
# 
# MISSING DATA

# In[ ]:


# We will start by checking missing values in each of the columns in our test dataset
# Missing Data
# We can do this by making use of or creating a simple heatmap to see where we have missing values in the test dataset
sns.heatmap(test.isnull(),yticklabels= False,cbar=False,cmap='viridis')


# Since we dont have Column "Survived" in our test dataset we can only replacing the missing values in column "Age".
# 
# we will try and see how the Age is distributed, using histogram.

# In[ ]:


sns.distplot(test['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


test['Age'].hist(bins=30,color='darkred',alpha = 0.7)


# In[ ]:


sns.countplot(x='SibSp',data = test)


# In[ ]:


test['Fare'].hist(color = 'green', bins = 40, figsize=(8,4))


# Data Cleaning
# 
# We want to fill in missing Age data instead of just dropping the age missing data rows. One way to do this is by filling in the mean age of all passengers(imputation). However, we can be smarter about this and check the average age by passenger class.
# 
# For example:

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age',data = test, palette= 'winter')


# Note: From the Boxplot above.
# 
# (1) We can see that in Pclass1, the mean or average age is 42yrs, meaning that we have older passengers in pclass1, hence we are going to replace the missing age of those passengers in pclass 1 by 42.
# 
# (2) We can see that in Pclass2, the mean or average age is 26yrs, meaning that we have less older passengers in pclass2 compare to pclass1 , hence we are going to replace the missing age of those passengers in pclass2 with 26.
# 
# (3) We can see that in Pclass3, the mean or average age is 23yrs, meaning that we very much younger passengers in pclass3 compare to pclass2, hence we are going to replace the missing age of those passengers in pclass3 by 23.

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

      if Pclass == 1:
          return 42

      elif Pclass == 2:
          return 26

      else:
          return 23   

    else:
       return Age


# In[ ]:


## Once we finish with the above code, we will now make use of the code below to fix those missing ages above
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# Now, let's have a look at our heatmap if truly all the missing passengers Ages have been replaced accordingly in our test dataset

# In[ ]:


## To do this, we are going to recall our heatmap code and check if there are still missing age data or not
sns.heatmap(test.isnull(),yticklabels= False,cbar=False,cmap='viridis')


# In[ ]:


## Just for us to justify what we are have in our heatmap above 
## and to see where else we re still having missing data (e.g Embarked)
test.isnull().sum()


# In[ ]:


## Now, let's try and fill those only one missing data in Fare column first using "Mode"
test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])


# In[ ]:


## Next, is the Cabin column but based on the fact that over 50% of the Cabin data are missing in train dataset above
## Hence, the most reasonable approach is to remove the entire Cabin Column in order to have the same result like
## train dataset
## Thus, to do this we have the following code
test.drop(['Cabin'],axis=1,inplace=True)


# In[ ]:


## Hence, to confirm that there are no missing values in any of the columns in our test dataset
## We try and run our heatmap code again
sns.heatmap(test.isnull(),yticklabels= False,cbar=False,cmap='viridis')


# In[ ]:


## Also again to justify what we have in our heatmap above 
## and to see that there are no missing values again in our train dataset
test.isnull().sum()


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


## we will now save the cleaned test dataset as "formulatedtest.csv" as follows:
test.to_csv('formulatedtest.csv',index=False)


# In[ ]:




