#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv("../input/Loan 2.csv")
data.drop('title',axis=1,inplace=True)
data.drop('earliest_cr_line',axis=1,inplace=True)
data.drop('inq_last_6mths',axis=1,inplace=True)


# In[ ]:


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# In[ ]:


data.head()


# ### Problem #1
# #### Using Bivariance 
# ##### To find pattern of defaulters 

# In[ ]:


sns.countplot(x='loan_status',data=data)


# ##### There are very less defaulters

# In[ ]:


sns.countplot(x='loan_status',hue='term',data=data)


# ##### the 60 month loan contain more number of defaulters

# In[ ]:


sns.countplot(x='loan_status',hue='grade',data=data)


# ##### There are more number of defaulters in class B,C and D

# In[ ]:


sns.countplot(x='loan_status',hue='emp_length',data=data)


# ##### More defaulter were present in the employee of 10+ year and 3 year

# In[ ]:


sns.countplot(x='loan_status',hue='verification_status',data=data)


# ##### More numbers of defaulters are from non verified category

# In[ ]:


sns.countplot(x='loan_status',hue='last_pymnt_d',data=data)


# ##### People from Year 2012-2013 were mostly Defaulters
# 

# ### Related to term

# #### Term #1

# In[ ]:


sns.countplot(x='term',data=data)


# ##### People prefer to take 36 month term loan instead of 60 month

# #### With respect to grade

# In[ ]:


sns.countplot(x='term',hue='grade',data=data)


# ##### Grade A and B people are tend to take more loan then other grades

# #### With respect to Subgrades

# In[ ]:


plt.figure(figsize=(15,15))
sns.countplot(x='term',hue='sub_grade',data=data)


# ##### A4 Grade people take most of the loans of 36 month period

# #### With respect to Emp_length

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='term',hue='emp_length',data=data)


# ##### Most of the loans are often taken by new employees or the employees which are having 10+ years of service

# #### With respect to home ownership

# In[ ]:


sns.countplot(x='term',hue='home_ownership',data=data)


# ##### Most of the loan cases are for making homes ( Home Loan) 

# #### With respect to verification status

# In[ ]:


sns.countplot(x='term',hue='verification_status',data=data)


# ##### Most of the 36 month Period loans are not verified

# #### With respect to Loan status

# In[ ]:


sns.countplot(x='term',hue='loan_status',data=data)


# ##### People often fail to pay 60 Month Period loan

# #### With respect to Purpose

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='term',hue='purpose',data=data)


# ##### Maximum people take loan to pay other loans 

# ## Analysis Based On Loan Amount
# #### Univariance

# In[ ]:


fig = data.loan_amnt.hist(bins=50) 
fig.set_title('Loan Requested Amount')
fig.set_xlabel('Loan Amount')
fig.set_ylabel('Number of loans')


# ##### We can see by this that the Most of the People opt for 10k loan money

# ## Summary 

# * As we can deduce by our visualizations 
# * The loan taken was mostly due to lack of house or due to their inability to pay old loans
# * further the new employees and employees with >10 year of service are more likely to take loans
# * The Employee who take loan and are at 3rd year of their service are least likely to repay the loan amount 
# * Most of the People opt for 10k loan money
# 

# ## the variables which are strong indicators of default

# * loan status
# * grade
# * term
# * employee length
# * grade
# * verification status
# * loan status
# * purpose
# * verification status
# * last payment date
# * subject grade
# * employee length
# * home ownership

# In[ ]:




