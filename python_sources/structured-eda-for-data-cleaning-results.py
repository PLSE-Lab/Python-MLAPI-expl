#!/usr/bin/env python
# coding: utf-8

# This kernel acts as the answer key for [Structured EDA for Data Cleaning](https://www.kaggle.com/sohier/structured-eda-for-data-cleaning/).

# ### 1) Read the manual
# We'll use the credit card dataset from [the R package AER](https://cran.r-project.org/web/packages/AER/AER.pdf).
# 
# Going into this exercise, the only information we have is from the dataset manual
# ```Format
# A data frame containing 1,319 observations on 12 variables.
# 
# card: Factor. Was the application for a credit card accepted?  
# reports: Number of major derogatory reports.   
# age: Age in years plus twelfths of a year.   
# income: Yearly income (in USD 10,000).   
# share: Ratio of monthly credit card expenditure to yearly income.   
# expenditure: Average monthly credit card expenditure.  
# owner: Factor. Does the individual own their home?  
# selfemp: Factor. Is the individual self-employed?  
# dependents: Number of dependents.  
# months: Months living at current address.  
# majorcards: Number of major credit cards held.  
# active: Number of active credit accounts.
# 
# Details
# According to Greene (2003, p. 952) dependents equals 1 + number of dependents, our calculations
# suggest that it equals number of dependents.
# Greene (2003) provides this data set twice in Table F21.4 and F9.1, respectively. Table F9.1 has just
# the observations, rounded to two digits. Note that age has some suspiciously low values (below one year) for some applicants.```
# 
# Potential issues: 
# - We'll want to convert `income` into the same units as `expenditure`.
# - Both `dependents` and `age` are flagged as problematic so we should take special note of those columns.
# - `months` might be more useful if we converted it to the same units as `age`.
# 
# ### 2) Review the Data Types
# This is the simplest check. At this stage we're mostly looking to confirm if the data types line up with the manual. We'll also want to note how the data is actually stored as it can be easy to otherwise miss errors like a number being stored as stored as a string.

# In[ ]:


import pandas as pd
import random


# In[ ]:


df = pd.read_csv("../input/AER_credit_card_data.csv")


# In[ ]:


df.info()


# Potential issues:
# - All of the columns labeled `Factor` in the manual are stored as strings, we will probably want to either categorize those or convert them to binary depending on what we see upon closer inspection.
# 
# ### 3) Print Sample Rows
# I've printed a random slice of rows to preserve the row order in case the data happens to be sorted or has other inter-row structure. Depending on your dataset, it might be helpful to have a domain expert talk you through this step and stage 4 to ensure that you understand what you're looking at.

# In[ ]:


random.seed(42)
df[random.randint(0, len(df)):].head()


# Potential issues:
# - The binary value columns do need to be converted to True/False.
# - There's a zero in the dependents column so the manual's explanation of that column can't be correct. Obvious conflicts with the manual are a bad sign; in a business context this could be a good time to consider letting the boss know that preparing the data will take a bit longer than expected.
# 
# ### 4) Summary Statistics
# This stage allows you to do basic sanity checks about the distribution of the data. For example, we know that none of the values can be negative.

# In[ ]:


df.describe()


# Potential issues:
# - The minimum age is implausibly low, as the manual noted.
# - 50% of the population falls within a 15 year age range. I would have expected a random sample of credit card holders to be more uniformly distributed across the adult age range. If possible, it would be helpful to understand the original sampling process.
# - `majorcards` appears to be a boolean rather than a count, contrary to the manual. 
# - The maximum value for `Active` is suprisingly high, but theoretically possible. A domain expert might be able to shed light on this.
# - The `share` column is supposed to equal the ratio `of monthly credit card expenditure to yearly income`. Given that the minimum expenditure is zero and the minimum share is not, there's either an error in the manual or with the calculation of share.
# 
# ### 5) Plotting
# Since this dataset is quite small, we'll use plot both basic histograms and relationships between columns. Each plot is pretty small... but it's so easy to do this way that you can include this step in even a cursory data review.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


sns.pairplot(df);


# The dependents histogram looks odd. Why would there be gaps? Let's take a closer look. Since we know there are only a small number of unique values, we can just print the raw value counts.

# In[ ]:


df.dependents.value_counts()


# The gaps were just the result of the default histogram's binning method. 
# 
# ### Next Steps
# This dataset definitely isn't clean yet, but we've identified several items to dig into. In a business context my next step would be to go and talk to whoever generated the data as the conflicts with the manual make me nervous.
# 
# Beyond that, the next steps would depend a great deal on the goal for your project. Based on what we've seen so far this dataset isn't ready if you want to make claims about the ages of credit card holders. If you just need the number of bad credit reports per credit card holder it might be fine.
# 
# ### Challenges
# - At least one more pair of the columns presents conflicting data in some cases. Can you identify those columns?
# - If you repeat the process for users who were and were not approved for another card, does anything stand out? Hint: this will reveal a *serious* problem for applying machine learning to the raw dataset.

# In[ ]:




