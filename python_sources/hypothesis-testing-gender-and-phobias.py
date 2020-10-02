#!/usr/bin/env python
# coding: utf-8

# ***Exploratory Data Analysis and Hypothesis Testing***

# *While growing up, I had stage fright. One of the most important lessons I have learnt in my life is to face your fears rather than running away from them. The goal of this notebook is to analyze the data in terms of Gender and the Phobias. We will perform some Exploratory data analysis around phobias  and their relationship with Gender*
# 
# *Since this is my first Kernel, I would be happy to see some remarks and suggestions, feel free to upvote if you feel like*

# In[ ]:


#import pandas for data manipulation and exploratory data analysis
import pandas as pd

#importing matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


# Read the data into the notebook
young = pd.read_csv('../input/responses.csv')


# In[ ]:


# It is important to get the sense of the data using the head function
young.head(10)


# In[ ]:


young.tail(10)
#there are 1010 rows and 150 variables in the dataset


# In[ ]:


# before starting the analysis get summary of the data
young.describe()


# In[ ]:


# It is helpful to know which columns have missing data, since we have a lot of columns in this dataset, it's better that we visualize it

nulls = young.isnull().sum().sort_values(ascending=False)
nulls.plot(
    kind='bar', figsize=(23, 5))

# we notice that height and weight have the most missing values but our analysis of the hypothesis testing of phobias does not get affected
#We also notice that gender has 6 missing values, moving on, we can remove them from the dataframe as it would not affect our analysis


# *we notice that height and weight have the most missing values but our analysis of the hypothesis testing of phobias does not get affected
# We also notice that gender has 6 missing values, moving on, we can remove them from the dataframe as it would not affect our analysis*

# *It is a good idea to know the column names to check if any column names have any space in front of them
# it was practically impossible to know by just looking at the dataframe*

# In[ ]:


young.columns


# In[ ]:


# we can call the shape function to look at the number of rows and columns of the dataset

young.shape

# our data has 1010 rows and 150 columns, this exercise turned out to be useful as for further analysis, we can focus on the question that needs to be anaswered


# In[ ]:


# For further analysis, we can use the info method

young.info()


# In[ ]:


# we already know from the visualization above that the Gender variable has 6 missing columns, we are also interested to know how the data is spread across male and female
young.Gender.value_counts(dropna = False)


# In[ ]:


#Data visualization is a way to spot obvious errors and outliers
# It is helpful in planning the data cleaning pipeline

#The question of interest for us in the dataset is " Do women fear certain phenomena significantly more than men?
# Hence we will plot Phobias on a histogram and then boxplots for the spread of phobias across men and women 

young.Flying.plot('hist')


# In[ ]:


young.Storm.plot('hist')


# In[ ]:


young.Darkness.plot('hist')


# In[ ]:


young.Heights.plot('hist')


# In[ ]:


young.Spiders.plot('hist')


# In[ ]:


young.Snakes.plot('hist')


# In[ ]:


young.Rats.plot('hist')


# In[ ]:


young.Ageing.plot('hist')


# In[ ]:


young['Dangerous dogs'].plot('hist')


# In[ ]:


young['Fear of public speaking'].plot('hist')


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Flying', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Storm', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Darkness', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Height', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Spiders', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Snakes', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Rats', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Ageing', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Dangerous dogs', data = young)


# In[ ]:


bp = sns.boxplot(x = 'Gender', y = 'Fear of public speaking', data = young)


# In[ ]:


# In this part of the exercise, I am going to drop all the records with missing values in either phobias or gender
young.dropna(subset = ['Flying','Storm', 'Darkness','Heights','Spiders','Snakes','Rats','Ageing','Dangerous dogs',
                       'Fear of public speaking','Gender'])
young.shape
# After dropping the columns, we have 984 records left for Hypothesis testing


# We will be using the Chi-Square test as both the variables are categorical and we need the proportions
# In this context the two random measures are often called factors
# Since the burden of proof is that the Gender and phobias are related, not that they are unrelated, the problem of testing the theory on Gender and Phobias can be formulated as :
# 
# ### Ho - Gender and Phobia are independent
# ### Ha - Gender and Phobia are not independent

# In[ ]:


# to run the hypothesis test, we will be using scipy.stats package

from scipy.stats import chi2


# In[ ]:


test = pd.DataFrame()
def table_creation(row, col):
    test = pd.crosstab(index=row,columns=col,margins=True)
    test.columns = ["1.0","2.0","3.0","4.0","5.0","rowtotal"]
    return(test);
    
def chisq_test(t, i):
    # Get table without totals for later use
    observed = t.ix[0:2,0:5]   
    #To get the expected count for a cell.
    expected =  np.outer(t["rowtotal"][0:2],t.ix["All"][0:5])/1010
    expected = pd.DataFrame(expected)
    expected.columns = ["1.0","2.0","3.0","4.0","5.0"]
    expected.index= test.index[0:2]
    #Calculate the chi-sq statistics
    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
    print("Chi-sq stat")
    print(chi_squared_stat)
    crit = chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = i)   # *
    print("Critical value")
    print(crit)
    p_value = 1 - chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=i)
    print("P value")
    print(p_value)
    return;


# In[ ]:


test = table_creation(young["Gender"],young["Flying"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Storm"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Heights"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Spiders"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Snakes"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Rats"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Ageing"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Dangerous dogs"])
print(test)
chisq_test(test,4)


# In[ ]:


test = table_creation(young["Gender"],young["Fear of public speaking"])
print(test)
chisq_test(test,4)


# ## Conclusion : In every test except the Gender vs. Heights, the critical value is less than the test statistic, hence the decision is to reject the null hypothesis
# ## The data provided has sufficient evidence, at 5% level of significance, to conclude the Gender and Phobias are not independent.
