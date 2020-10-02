#!/usr/bin/env python
# coding: utf-8

# # Assignment 6: FPM
# 
# ### Instructions
# 
#  1. To get started on this assignment, first fork this Notebook by clicking "Copy and Edit" in the top right corner. You should be presented with a personal version of the assignment that you can work in. Make sure that the visibility of your Notebook is set to private!
#  2. Make sure you can run the first two code blocks (import and df.head()). If so, you're all set to start the assignment, good luck!
#  

# In[ ]:


###################
# Import packages #
###################
import numpy as np
import pandas as pd
import seaborn as sns


pd.options.display.max_rows
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


########################
# Read the input files #
########################
# Use the pandas library to read the csv file.
df = pd.read_csv("/kaggle/input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Print the first few rows
df.head()


# ## 1. Data Exploration and Preprocessing (4pt)

# ### Task 1: Overview (0.5pt)
# Print the columns in the dataset with their type and [unique](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html) values (without duplicates).

# In[ ]:


# TODO: list columns and values


# ### Task 2: Clean up (1pt)
# You should see that there's something wrong with the columns "SeniorCitizen" and "TotalCharges". We want the first one to be of type 'object' with values "Yes" and "No". The latter column should be a 'float' type, but since it also contains empty values (empty or spaces), you will first need to convert those to `np.nan`. Furthermore the column "customerID" is not required and can be dropped from the dataframe. 
# 
# 
# Clean up the data as described above. Then print their type and values again.

# In[ ]:


# TODO: Fix datatype of TotalCharges


# TODO: Fix datatype of SeniorCitizen and replace values


# TODO: Drop the column customerID (Hint: add errors="ignore" to allow this statement to run multiple times)


# In[ ]:


for col in ['SeniorCitizen', 'TotalCharges']:
    # TODO: print type and values
    pass


# ### Task 3: Visualization (0.25pt)
# There are better ways to visualize continuous variables. Render a plot for each continuous column that shows its distribution.

# In[ ]:


# TODO: Render cotinuous variables. Hint: There is a single pandas function that does exactly this.


# ### Task 4: Correlation (0.25pt)
#  1. Inspect the pairwise correlation between these variables (pearson).
#  2. Create a plot to show the correlation.

# In[ ]:


# TODO: Show the pairwise correlation. Hint: There is a single pandas function that does exactly this.


# In[ ]:


# TODO: Create a visualization for the pairwise correlation between all continuous variables. (Hint: check out seaborn pairplots)


# ### Task 5: Discretize Continuous Columns (0.5pt)
# There are still some issues in the dataset with respect to pattern mining. Continuous variables often cannot form patterns because their exact values simply don't occur often enough to meet the minsup threshold. We need to group them together first if we want to use them in pattern mining.
# 
# 
# Apply a discretization for the columns "tenure", "MonthlyCharges" and "TotalCharges". Divide them in 10 buckets/bins of equal width. Then inspect the amount of times each value occurs.

# In[ ]:


# TODO discretize "tenure", "MonthlyCharges" and "TotalCharges" in 10 equal width bins
# df['tenure-bin'] = 
# df['MonthlyCharges-bin'] = 
# df['TotalCharges-bin'] = 

for col in ['tenure-bin', 'MonthlyCharges-bin', 'TotalCharges-bin']:
    print(col)
    # TODO inspect the frequency of each bucket.
    


# ### Task 6: Categorical Correlation (1.5pt)
# Gender obviously has no influence on churn rate. Find at least two variables that do have a higher correlation with Churn. Plot them together as shown and explain why you think they are interesting.

# In[ ]:


df.groupby(['Churn', 'gender']).size().unstack(fill_value=0).plot.bar()


# In[ ]:


# TODO: Plot two more interesting variables together with churn.
print("=========== My 1st example ===========")
# TODO: Plot

# TODO: Explain why
print("Is interesting because", "TODO")


# In[ ]:


print("=========== My 2nd example ===========")
# TODO: Plot

# TODO: Explain why
print("Is interesting because", "TODO")


# ## 2. Frequent Pattern Mining (6pt)
# For this part you will implement the Eclat algorithm and perform frequent pattern mining on the dataset. Then you can derive association rules from the itemsets and look for interesting patterns. 

# ### Task 1: Frequent Itemsets (2.5pt)
# Implement the eclat algorithm in the template below (fill in TODOs).

# In[ ]:


def dataframeToTransactions(df):
    """ Converts a Pandas dataframe to a list of transactions """
    rows = df.to_dict(orient="records")
    data = [tuple(f"{k}={v}" for k, v in row.items()) for row in rows]
    return data

transactions = dataframeToTransactions(df)
[", ".join(t) for t in transactions[:5]]


# In[ ]:


from collections import defaultdict

def transactionsToTidlist(transactions):
    """ Converts transactions matrix to tidlist.
        Return: List of the form [(item1, {tids1}), (item2, {tids2})]
        (Hint: Store them in a dict d (item -> set) and return list(d.items()) 
    """
    # TODO: Implement
    return []

# DEBUG CODE
tidlist = transactionsToTidlist(transactions)
for item, tids in tidlist[:10]:
    print(item, len(tids))
     
# == Expected Output ==
# gender=Female 3488
# SeniorCitizen=No 5901
# Partner=Yes 3402
# Dependents=No 4933
# tenure=1 613
# PhoneService=No 682
# MultipleLines=No phone service 682
# InternetService=DSL 2421
# OnlineSecurity=No 3498
# OnlineBackup=Yes 2429


# In[ ]:


def eclat(df, minsup):
    transactions = dataframeToTransactions(df)
    tidlist = transactionsToTidlist(transactions)
    return _eclat([], tidlist, minsup)
    
    
def _eclat(prefix, tidlist, minsup):
    """ Implement the Eclat algorithm recursively.
        prefix: items in this depth first branch (the set alpha).
        tidlist: tids of alpha-conditional db.
        minsup: minimum support.
        return: list of itemsets with support >= minsup. Format: [({item1, item2}, supp1), ({item1}, supp2)]
    """
    # TODO: Implement
    return []

# DEBUG CODE
# Reference implementation takes at most 10 seconds to compute the following
itemsets = eclat(df, 800)
print("Itemsets:", len(itemsets))
sorted(itemsets, key=lambda x: x[1], reverse=True)[:10]

# == Expected Output (or similar) ==
# Itemsets: 18427
# [({'PhoneService=Yes'}, 6361),
#  ({'SeniorCitizen=No'}, 5901),
#  ({'PhoneService=Yes', 'SeniorCitizen=No'}, 5323),
#  ({'Churn=No'}, 5174),
#  ({'Dependents=No'}, 4933),
#  ({'Churn=No', 'PhoneService=Yes'}, 4662),
#  ({'Churn=No', 'SeniorCitizen=No'}, 4508),
#  ({'Dependents=No', 'PhoneService=Yes'}, 4457),
#  ({'PaperlessBilling=Yes'}, 4171),
#  ({'Churn=No', 'PhoneService=Yes', 'SeniorCitizen=No'}, 4056)]


# ### Task 2: Association Rules (1.5pt)
# Derive association rules from itemsets.

# In[ ]:


from itertools import chain, combinations

def subsets(itemset):
    """ List all strict subsets of an itemset without the empty set
        subsets({1,2,3}) --> [{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}]
    """
    s = list(itemset)
    return map(set, chain.from_iterable(combinations(s, r) for r in range(1, len(s))))

def deriveRules(itemsets, minconf):
    """ Returns all rules with conf >= minconf that can be derived from the itemsets.
        Return: list of association rules in the format: [(antecedent, consequent, supp, conf), ...]  where antecedent and consequent are itemsets.
    """
    # TODO implement
    return []
        

# DEBUG CODE
rules = deriveRules(itemsets, 0.9)
print("Rules:", len(rules))
sorted(rules, key=lambda x: x[3], reverse=True)[:5]

# == Expected Output (or similar) ==
# Rules: 747880
# [({'MonthlyCharges-bin=(78.55, 88.6]'}, {'PhoneService=Yes'}, 953, 1.0),
#  ({'InternetService=Fiber optic', 'SeniorCitizen=Yes'},
#   {'PhoneService=Yes'},
#   831,
#   1.0),
#  ({'StreamingMovies=No internet service'},
#   {'MonthlyCharges-bin=(18.15, 28.3]'},
#   1526,
#   1.0),
#  ({'StreamingMovies=No internet service'},
#   {'MonthlyCharges-bin=(18.15, 28.3]', 'StreamingTV=No internet service'},
#   1526,
#   1.0),
#  ({'StreamingTV=No internet service'},
#   {'MonthlyCharges-bin=(18.15, 28.3]', 'StreamingMovies=No internet service'},
#   1526,
#   1.0)]


# ### Task 3: Inspect Patterns (2pt)
# It is clear that classical frequent pattern mining with so many columns leads to too many itemsets and association rules. Select a subset of the columns that you think is interesting and mine patterns on that subset. Then put the resulting association rules in a dataframe and filter it so only the ones with "Churn=Yes" in the consequent remain. Pick one pattern and describe what it means and why you think it is interesting.
# 

# In[ ]:


# TODO: Create a subset of the data


# TODO: Mine itemsets and association rules (find a good minsup and minconf value yourself)


# The rules inside a DataFrame
rulesFrame = pd.DataFrame(rules, columns=['Antecedent', 'Consequent', 'Supp', 'Conf'])

# TODO: Filter rulesFrame to only show rows with "Churn=Yes" as consequent


# Describe one of the patterns in the table above and explain why it is interesting:
# > Your answer

# ## 4. Feedback (0pt)
# Optionally include feedback on the assignment, specifically remarks on the workload, difficulty and relevance of the assignment. Suggestions on how to improve it towards next year are also welcome. Naturally what you write here will not affect your grades in any way.
# 

#  > Feedback here
