#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# 1. **Data cleaning** - initial cleaning completed and columns assessed

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re


# In[ ]:


school_df = pd.read_csv("../input/2016 School Explorer.csv")


# In[ ]:


school_df.head()


# In[ ]:


school_df.dtypes


# # Data cleaning
# - Removing percentage symbols from all percentage columns and converting them into numeric values
# - Parsing School Income Estimate to produce numeric values and removing currency symbols
# - Discard a few columns that don't have any meaningful values: Adjusted Grade, New?, Other Location Code in LCGMS
# - Create a new column to indicate if a school is public by checking if the string "P.S." exists in it
# - Convert ratings to categorical variables

# ## Handling percent

# In[ ]:


percentage_columns = ['Percent ELL', 'Percent Asian', 'Percent Black', 'Percent Hispanic', 'Percent Black / Hispanic', 'Percent White', 'Student Attendance Rate', 'Percent of Students Chronically Absent', 'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %', 'Effective School Leadership %', 'Strong Family-Community Ties %', 'Trust %']


# In[ ]:


def parse_percent(val):
    """
        If nan or empty string, return nan
        else remove percentage sign from the string if present
        and cast to integer
    """
    percent_sign = "%"
    if( pd.isnull(val) or len(val) == 0):
        return np.nan
    
    if (percent_sign in val):
        return float(val.replace(percent_sign, ""))
    
    return float(val.replace(percent_sign, ""))


# In[ ]:


school_df[percentage_columns] = school_df[percentage_columns].applymap(parse_percent)


# In[ ]:


school_df[percentage_columns].dtypes


# ## Handling School Income Estimate

# In[ ]:


def parse_income(val):
    """
        Parses a string representing an income by:
        1. Removing dollar and commas in the representation
        2. Returning the float value of the same
    """
    if( pd.isnull(val) or len(val) == 0):
        return np.nan
    
    val = re.sub('[$,]', '', val)
    
    return float(val)


# In[ ]:


school_df['School Income Estimate'] = school_df['School Income Estimate'].apply(parse_income)


# ## Columns with no valuable information: Adjusted Grade, New?, Other Location Code in LCGMS

# In[ ]:


school_df[school_df['Adjusted Grade'].notnull()]['Adjusted Grade']


# In[ ]:


school_df[school_df['New?'].notnull()]['New?']


# In[ ]:


school_df[school_df["Other Location Code in LCGMS"].notnull()]["Other Location Code in LCGMS"]


# In[ ]:


school_df = school_df.drop(["Adjusted Grade", "New?", "Other Location Code in LCGMS"], axis=1)


# ## Create new column `Is public`
# 

# In[ ]:


def is_public(val):
    """
        Returns true if the string val
        contains the substring 'P.S'
    """
    return 'P.S.' in val


# In[ ]:


school_df["Is public"] = school_df["School Name"].apply(is_public)


# ## Convert ratings to categorical variables

# In[ ]:


rating_columns = ['Rigorous Instruction Rating', 'Collaborative Teachers Rating', 'Supportive Environment Rating', 'Effective School Leadership Rating', 'Strong Family-Community Ties Rating', 'Trust Rating', 'Student Achievement Rating']


# In[ ]:


school_df[rating_columns] = school_df[rating_columns].apply(lambda x: x.astype('category'))

