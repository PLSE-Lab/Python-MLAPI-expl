#!/usr/bin/env python
# coding: utf-8

# **WELCOME  **
# 
# **ABOUT  KIVA**
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower.
# 
# In Kaggle Datasets' inaugural Data Science for Good challenge, Kiva is inviting the Kaggle community to help them build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loans. Unlike traditional machine learning competitions with rigid evaluation criteria, participants will develop their own creative approaches to addressing the objective. Instead of making a prediction file as in a supervised machine learning problem, submissions in this challenge will take the form of Python and/or R data analyses using Kernels, Kaggle's hosted Jupyter Notebooks-based workbench.
# 
# Kiva has provided a dataset of loans issued over the last two years, and participants are invited to use this data as well as source external public datasets to help Kiva build models for assessing borrower welfare levels. Participants will write kernels on this dataset to submit as solutions to this objective and five winners will be selected by Kiva judges at the close of the event. In addition, awards will be made to encourage public code and data sharing. With a stronger understanding of their borrowers and their poverty levels, Kiva will be able to better assess and maximize the impact of their work.
# 
# The sections that follow describe in more detail how to participate, win, and use available resources to make a contribution towards helping Kiva better understand and help entrepreneurs around the world.

# **IMPORTING THE  LIBRARIES**
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization of the data
import seaborn as sns # 
import re


# In[ ]:


df_kiva = pd.read_csv("../input/kiva_loans.csv")


# **ANALYSE THE DATA**
#  
# * shape of the data
# * Type of the data
# * Is there any Missing values?
# 
# 
# 

# In[ ]:


print(df_kiva.shape)# rows and columns



# In[ ]:


df_kiva.head(10)


# In[ ]:


print(type(df_kiva))# type of the data


# **MISSING DATA**

# In[ ]:


total_missing=df_kiva.isnull().sum().sort_values(ascending=False)

print(total_missing)


# In[ ]:


# due to high frequency of missing data in tags- 171416 ,region-56800,funded_time-48331,partner_id-13507
#use-4228,borrower_genders -4221,disbursed_time-2396,country_code -8
# we can't anlayse these features 
# let's analyse remaining features


          
           


# In[ ]:


df_kiva.nunique()# unique data


# **Popular Sector**
# 
# The Sector which is the most popular for loans is provided in the bar chart below
# 

# In[ ]:


plt.figure(figsize=(12,6))
df_kiva['sector'].value_counts().head(10).plot.bar()


# In[ ]:


plt.figure(figsize=(12,6))
df_kiva['country'].value_counts().head(10).sort_values(ascending=False).plot.bar()


# 

# **REPAYMENT  INTERVAL**
# 

# In[ ]:


df_kiva['repayment_interval'].value_counts().unique()
df_kiva['repayment_interval'].value_counts().head(10).plot.barh()
plt.title("Types of repayment intervals", fontsize=16)


# 

# 

# In[ ]:


df_kiva['activity'].sort_values().unique()


# In[ ]:


plt.figure(figsize=(12,6))
df_kiva['activity'].value_counts().head(10).plot.barh()


# 
# 
# **CONCLUSION**
# 
# 
# Philippines is most frequent countries who got more loans followed by Kenya.
# 
# Agriculture Sector is more frequent in terms of number of loans followed by Food.
# 
# Types of interval payments monthly, irregular, bullet and weekly. Out of which monthly is more frequent and weekly is less frequent.
# 
# Top 3 loan activity which got more number of funded are Farming , general Store,personal housingexpense.
# 

# In[ ]:




