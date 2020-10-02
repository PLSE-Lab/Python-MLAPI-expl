#!/usr/bin/env python
# coding: utf-8

# ## 0. START FROM THE BEGINNING
# (Please, sorry about english mistakes. I'm not a native speaker)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns
import os

df = pd.read_csv("../input/insurance.csv")
pd.options.display.max_columns = None


# In[ ]:


df.info()


# In[ ]:


df.describe()


# Above, you can verify that the range to the age is acceptable (18-64), as is for the the number of children (0-5), charge and bmi. 
# Therefore, no inconsistencies were identified.

# ## 1. CREATING NEW VARIABLES

# I have identified opportunities for creating new categorical variables. The main reason is identify possible groups of data. 
# 
# #### 1.1 BMI range description.

# Below, i'll create a new variable to describe the categorical range of BMIs.
# Source: https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm

# In[ ]:


df['bmi_desc'] = 'default value'
df.loc[df.bmi >= 30, 'bmi_desc'] = 'Obesity'
df.loc[df.bmi <18.5, 'bmi_desc'] = 'Underweight'
df.loc[ (df.bmi >= 18.5) & (df.bmi < 24.9), 'bmi_desc'] = 'Normal'
df.loc[ (df.bmi > 24.9) & (df.bmi <30), 'bmi_desc'] = 'Overweight'
df.head()


# #### 1.2 BMI range description simplified combine with smoker people[](http://)

# In[ ]:


df['bmi_smoker'] = 'default value'
df.loc[(df.bmi_desc == 'Obesity') & (df.smoker == 'yes'), 'bmi_smoker'] = 'obese_smoker'
df.loc[(df.bmi_desc == 'Obesity') & (df.smoker == 'no'), 'bmi_smoker'] = 'obese_no_smoker'
df.loc[(df.bmi_desc != 'Obesity') & (df.smoker == 'yes'), 'bmi_smoker'] = 'other_smoker'
df.loc[(df.bmi_desc != 'Obesity') & (df.smoker == 'no'), 'bmi_smoker'] = 'other_no_smoker'
df.head()


# ## 2. CHARGE VARIABLE Analysis

# #### Distribution

# In[ ]:


sns.distplot(df[['charges']])


# #### The distribution is the same to smoker people?
# 
# There are differences (see below), but is not conclusive.

# In[ ]:


sns.distplot(df.loc[df.smoker == 'yes']['charges'].values.tolist())


# In[ ]:


sns.distplot(df.loc[df.smoker == 'no']['charges'].values.tolist())


# ## 3. AGE VARIABLE Analysis

# #### Distribution

# In[ ]:


sns.distplot(df[['age']])


# #### This variable is correlated with another?
# 

# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges']).set_title('3.1 - Charges vs Age without filter')
sns.despine()


# Above, you can verify that there are one positive trend to '3 groups'. 
# 
# **There are a categorical variable in the dataset that justify this behavior?
# Yes.** 
# 
# 
# After analysis of the subsequent graphs is possible to conclude that:
# * The charge with obese people is not necessairly higher than compared with other categories. However is possible note that are a subset of obese people with higher costs that other obese.  (3.4 graph).
# * The charge with Non-smokers is lower than compared with smoker people (3.5 graph).
# * The charge with the smokers are clearly divide in two 'categories'. Is possible note that people who are obese AND smoke cost more to the insurer than people who are not obese and smoke (3.6 graph).
# * Is not possible identify strong relationship from other variables (graphs 3.1 -> 3.4)
# 
# These are the key findings for this variable. Graphs below.

# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges'], hue='sex', data=df).set_title('3.1 - Charges vs Age filter by sex')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges'], hue='children', data=df).set_title('3.2 - Charges vs Age filtered by the number of children')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges'], hue='region', data=df).set_title('3.3 - Charges vs Age filtered by region')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges'], hue='bmi_desc', data=df).set_title('3.4 - Charges vs Age filtered by bmi description')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges'], hue='smoker', data=df).set_title('3.5 - Charges vs Age filtered by smoker')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['age'] , y= df['charges'], hue='bmi_smoker', data=df).set_title('3.6 - Charges vs Age filtered by bmi_smoker')
sns.despine()


# ### 4. BMI VARIABLE Analysis

# In[ ]:


sns.distplot(df[['bmi']])


# In[ ]:


sns.scatterplot(x= df['bmi'] , y= df['charges']).set_title('4.1 - Charges vs Bmi')
sns.despine()


# Above, you can verify that there is no clear relantionship from this variables. 
# 
# **There are a categorical variable in the dataset that justify this behavior?
# Yes.** 
# 
# 
# After analysis of the subsequent graphs is possible to conclude that:
# * The charge with the smokers are clearly divide in two 'categories'. Is possible note that people who are obese AND smoke cost more to the insurer than people who are not obese and smoke (4.5 graph)
# * Is not possible identify strong relationship from other variables (graphs 4.1 -> 4.4)
# 
# These are the key findings for this variable. Graphs below.

# In[ ]:


sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='sex', data=df).set_title('4.2 - Charges vs Bmi filtered by sex')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='region', data=df).set_title('4.3 - Charges vs Bmi filtered by region')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='children', data=df).set_title('4.4 - Charges vs Bmi filtered by number of children')
sns.despine()


# In[ ]:


sns.scatterplot(x= df['bmi'] , y= df['charges'], hue='smoker', data=df).set_title('4.5 - Charges vs Bmi filtered by smoker_bmi')
sns.despine()

