#!/usr/bin/env python
# coding: utf-8

# **Import Libraries.**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display


# **Some Notebook Settings.**

# In[2]:


warnings.filterwarnings('ignore') # ignore warnings.
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True # autocomplete feature.')
pd.options.display.max_rows = None # set maximum rows that can be displayed in notebook.
pd.options.display.max_columns = None # set maximum columns that can be displayed in notebook.
pd.options.display.precision = 2 # set the precision of floating point numbers.


# **Check Encoding of Data.**

# In[3]:


# # Check the encoding of data. Use ctrl+/ to comment/un-comment.

# import chardet

# rawdata = open('../input/database.csv', 'rb').read()
# result = chardet.detect(rawdata)
# charenc = result['encoding']
# print(charenc)
# print(result) # It's ascii with 100% confidence.


# **Read Data.**

# In[4]:


df = pd.read_csv('../input/database.csv', encoding='utf-8')
df.drop_duplicates(inplace=True) # drop duplicates if any.
df.shape # num rows x num columns.


# <hr>

# **Check for missing values.**

# In[5]:


miss_val = (df.isnull().sum()/len(df)*100).sort_values(ascending=False) # columns and their missing values in percentage.
miss_val[miss_val>0]


# **Take a look at the data.**

# In[6]:


df.head()


# **Year vs Incident.**

# In[7]:


df.groupby('Incident Year').size().plot()


# It seems that the incidents keep on increasing every year. But, this might not be the case.<br>
# *"Awareness in terms of reporting wildlife incidents has increased over time, as has the ease of reporting, due to advances in telecommunications. This means that although the number of records has increased over the years, this does not necessarily indicate a true increase in the number of incidents that have occurred."* -> https://aertecsolutions.com/2019/02/04/the-impact-of-wildlife-on-aviation/?lang=en

# **Month vs Incident.**

# In[8]:


df.groupby('Incident Month').size().plot()


# It seems that Jul-Aug-Sep in particular see maximum incidents.

# <hr>

# **Chi-Square Test of Independence.**

# Let us check using chi-square test of independence, whether Aircraft Strikes and Incident Month are independent.

# Below is a list of column names. These columns are categorical with 0-1 binary values. They record if a particular part of Aircraft udergo strike with wildlife or not.

# In[9]:


strike = ['Radome Strike', 'Windshield Strike', 'Nose Strike', 'Engine1 Strike', 'Engine2 Strike', 'Engine3 Strike',
          'Engine4 Strike', 'Propeller Strike', 'Wing or Rotor Strike', 'Fuselage Strike', 'Landing Gear Strike',
          'Tail Strike', 'Lights Strike', 'Other Strike']


# **Freuency Table.**

# In[10]:


table = df.groupby('Incident Month')[strike].sum()
table # Incident Month vs Strike.


# In[11]:


(table.sum(axis=0) # column sum.
,table.sum(axis=1)) # row sum.


# In[12]:


table.sum(axis=0).sum(), table.sum(axis=1).sum() # sum of all rows and all columns.


# Sum of all rows and all columns of table are equal, as expected.

# Now, lets find the minimum frequencies in the table.

# In[13]:


df.groupby('Incident Month')[strike].sum().min()


# The frequencies of groupmust be greater than or equal to 5. That is the case here. Thus, we can proceed with the test.

# **Hypothesis and test.**

# Null Hypothesis,      H0: Incident month and Aircraft Strike are independent of each other.<br>
# Alternate Hypothesis, H1: Incident month and Aircraft Strike are not independent of each other.

# In[14]:


from scipy import stats

chi2_stat, p_val, dof, ex = stats.chi2_contingency(table)
print("===Chi2 Stat===")
print(chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(dof)
print("\n")
print("===P-Value===")
print(p_val)
print("\n")
print("===Contingency Table===")
pd.DataFrame(ex)


# p-value is 0 (<0.01). This means that Null hypothesis is rejected at 1% level of significance.<br>
# This means that Incident month and Aircraft Strike are infact not independent from each other.

# <hr>
