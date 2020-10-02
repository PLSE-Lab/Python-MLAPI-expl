#!/usr/bin/env python
# coding: utf-8

# # Hello MLWorld; Part 2, an extended Introduction to Machine Learning.
# **A workspace for the [Machine Learning course](https://www.kaggle.com/learn/machine-learning) (Level 2).**
# 
# # Missing Values.
# Your data can contain missing values for a whole host of reasons - provided you've passed the first step, acceptance, here are three approaches to dealing with missing values, and a comparison.
# 
# 
# ## 1) Drop Columns with Missing Values.
# Let's explore [Listwise Deletion](https://www.theanalysisfactor.com/when-listwise-deletion-works/).  
# _A method_ for dealing with nulls - but a very blunt one, use with caution.  
# Mismatches between test & training data can occur, and your model will lose access to all of the data from dropped columns.
# However, if your column is mostly nulls, it may a viable solution.     
# Think about what might be a nail when using this hammer.

# In[ ]:


import pandas as pd

def null_dict(df): return(dict(zip(df.columns, [l for l in df.isnull().sum()])));
iowa_path = '../input/house-prices-advanced-regression-techniques/test.csv'
melb_path = '../input/melbourne-housing-snapshot/melb_data.csv'
orig_data = pd.read_csv(iowa_path)
orig_missing_cols = [l for l in orig_data.columns if orig_data[l].isnull().any()]

print("1) Amount of nulls per column [ orig_data ] \n%s\n"
    % (null_dict(orig_data)))

data_drop_na = orig_data.dropna(axis=1)
print("2) Amount of nulls per column after .dropna [ data_drop_na ] \n%s\n"
     % (null_dict(data_drop_na)))
            
data_drop_lc = orig_data.drop(orig_missing_cols, axis=1)
print("3) Amount of nulls per column after safe drop [ column_dropped_data ] \n%s"
     % (null_dict(data_drop_lc)))


# To keep the munging process transparent and repeatable, it is best to declare, or use consistent means to obtain, the column(s) requiring removal from your dataset.
# Examples **2** and **3** from above achieve the same result, but the way example **3** had it's columns dropped can be repeated on another dataset _(think train/test split)_, which is a safe way of avoiding the assumptions, and consequent errors that come as a result of blindly applying `df.dropna()`. 
# 
# ## 2) Imputation -  Filling Missing Values
# To _impute_ is to:
# > Assign [a value] to something by inference from the value of the products or processes to which it contributes.
# 
# Imputation is the process of filling missing values. These imputed values won't be absolutely correct, but will often lead to more accurate results from your model (as is better covered in [this article](http://www.stat.columbia.edu/~gelman/arm/missing.pdf) from "Data Analysis Using Regression and Multilevel/Hierarchical Models" By Andrew Gelman.
#  
#  Scikit-learn offers a few handy utilities for imputing missing values; `SimpleImputer()` will be explored below to illustrate the process.

# In[ ]:


from sklearn.impute import SimpleImputer

# new up, so as not to impact original data
new_data = orig_data.copy()
cols_with_missing = (col for col in new_data.columns
                               if new_data[col].isnull().any())

na_cols = (col for col in new_data.columns
                               if new_data[col].isna().any())

[print(i) for i in cols_with_missing]
    
my_imputer = SimpleImputer()
imputed_data = my_imputer.fit_transform(orig_data)
new_data = pd.DataFrame(imputed_data)
new_data.columns = original_data.columns


# Imputation can be included in scikit-learn pipelines, 
