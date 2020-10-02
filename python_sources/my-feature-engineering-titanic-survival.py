#!/usr/bin/env python
# coding: utf-8

# Libraries involved in this Kernel are
# 
# **Pandas** for data manipulation and ingestion.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
from matplotlib import pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merged = pd.concat([train, test], sort = False)


# **1.Processing Cabin**
# 
# Cabin is alphanumeric type variable with no special characters (like ., /, % etc) between letters and numbers.NaN will be flagged as 'X' and only the 1st character will be retained wherever Cabin has alphanumeric values. 

# In[ ]:


##Flag all the NaNs of Cabin as 'X'
merged.Cabin.fillna(value = 'X', inplace = True)
##Keep only the 1st character where Cabin is alphanumerical
merged.Cabin = merged.Cabin.apply( lambda x : x[0])
display(merged.Cabin.value_counts())


# **2.Processing Name**
# 
# Professionals like Dr, Rev, Col, Major, Capt will be put into 'Officer' bucket. Titles such as Dona, Jonkheer, Countess, Sir, Lady, Don were usually entitled to the aristocrats and hence these titles will be put into bucket 'Aristocrat'. Also Mlle and Ms is replaced with Miss and Mme by Mrs as these are French titles.

# In[ ]:


display(merged.Name.head(8))


# In[ ]:


##Create a new variable Title that extracts titles from Name
merged['Title'] = merged.Name.str.extract('([A-Za-z]+)\.')
##Create a bucket Officer and put Dr, Rev, Col, Major, Capt titles into it
merged.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)
##Put Dona, Jonkheer, Countess, Sir, Lady, Don in bucket Aristocrat
merged.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)
##Replace Mlle and Ms with Miss. And Mme with Mrs
merged.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
##Count the extracted categories of Title from Name
display(merged.Title.value_counts())


# **3.Processing SibSp & Parch**
# 
# Since these two variables SibSp & Parch together indicate the size of a family,  a new variable 'Family_size' is created from these two variables.

# In[ ]:


##Merge SibSp and Parch to create a variable Family_size.'''
merged['Family_size'] = merged.SibSp + merged.Parch + 1  # Adding 1 for single person
##Create buckets of single, small, medium, and large and then put respective values into them.'''
merged.Family_size.replace(to_replace = [1], value = 'single', inplace = True)
merged.Family_size.replace(to_replace = [2,3], value = 'small', inplace = True)
merged.Family_size.replace(to_replace = [4,5], value = 'medium', inplace = True)
merged.Family_size.replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)
display(merged.Family_size.value_counts())


# **4.Processing Ticket**
# 
# Ticket is an alphanumeric type variable. There will be two groups created - one for number and another fo character extracted from string .If a row contains both character and number, character will be retained.

# In[ ]:


##Assign N if there is only number and no character. If there is a character, extract the character only
ticket = []
for x in list(merged.Ticket):
    if x.isdigit():
        ticket.append('N')
    else:
        ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])
        
##Swap values
merged.Ticket = ticket
##Count the categories in Ticket.'''
display(merged.Ticket.value_counts())


# References : https://www.kaggle.com/eraaz1/a-comprehensive-guide-to-titanic-machine-learning#About-this-Kernel
