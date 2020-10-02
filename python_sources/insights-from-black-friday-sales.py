#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
pd.set_option("display.max_rows",1001)
pd.set_option("display.max_columns",300)


# **Step 1**
# 
# I am going to start by looking at some sample records in the data and then will decide what to analyze

# In[ ]:


bfd = pd.read_csv("../input/BlackFriday.csv")


# In[ ]:


bfd.head()


# **Observation 1**
# 
# So, looking the first 5 records, it is clear that there are **multiple records for some User_IDs**.
# 
# **Step 2**
# 
# Let's check if for a User_ID+Product_ID combination, there are multiple records or no

# In[ ]:


grp = bfd.groupby(['User_ID','Product_ID']).count()['Purchase'].reset_index()
grp[grp.Purchase>1]


# **Observation 2**
# 
# There are no duplicate records for User_ID+ Product_ID combination
# 
# **Step 3**
# 
# Let's check how many unique customers are there in the dataset and look at the various categories

# In[ ]:


grp = bfd.groupby(['Gender']).nunique()['User_ID']
grp.apply(lambda x : 100*x/grp.sum()).reset_index()


# In[ ]:


grp_city = bfd.groupby(['Gender','City_Category']).nunique()['User_ID'].reset_index()
pivot_city = grp_city.pivot(index='City_Category',columns='Gender',values='User_ID')
colors = ["#006D2C", "#31A354"]
pivot_city.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Gender by City_category')


# In[ ]:


grp_city = bfd.groupby(['Gender','City_Category']).sum()['Purchase'].reset_index()
pivot_city = grp_city.pivot(index='City_Category',columns='Gender',values='Purchase')
pivot_city.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Purchase value by Gender and City_category')


# In[ ]:


grp_age = bfd.groupby(['Gender','Age']).nunique()['User_ID'].reset_index()
pivot_age = grp_age.pivot(index='Age',columns='Gender',values='User_ID')
pivot_age.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Gender by Age')


# In[ ]:


grp_occ = bfd.groupby(['Gender','Occupation']).nunique()['User_ID'].reset_index().sort_values(by='User_ID',ascending=False)
pivot_occ = grp_occ.pivot(index='Occupation',columns='Gender',values='User_ID')
pivot_occ.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Gender by Occupation')


# In[ ]:


grp_occ = bfd.groupby(['Gender','Occupation']).sum()['Purchase'].reset_index()
pivot_occ = grp_occ.pivot(index='Occupation',columns='Gender',values='Purchase')
pivot_occ.loc[:,['F','M']].plot.bar(stacked=True, color=colors, figsize=(20,10),title='Distribution of Purchase amount by Gender and Occupation')


# **Observation 3**
# 
# * Only 28.28% of the customers are Female.
# * Most occupations have a bias towards Men, except Occupation=9 which is tilted towards the Female population.
# * Most customers belong to city_category 'C'. But the most purchase amount was spent by city_category 'B'.
# * Number of people employed in an occupation is directly proportional to the total amount of purchase made by people in the occupation.
# 
# **Step 4**
# 
# Lets dive deep into the products and see what Product_ID is purchase more by which segment of customers.
# 
# Here is the list of **top 20 products** bought by each Age group separated by Gender 

# In[ ]:


age_prd = bfd.groupby(['Age','Gender','Product_ID']).count()['User_ID'].reset_index().sort_values(by=['Age','Gender','User_ID'],ascending=False).reset_index(drop=True)
top_20 = age_prd.groupby(['Age','Gender'])['Product_ID','User_ID'].apply(lambda x : x.head(20))
top_20 = top_20.reset_index()
top_20.dtypes
#top_5.head()


# In[ ]:


pivot_top_20 = top_20.pivot_table(index='Product_ID',columns=['Gender','Age'],values='User_ID',fill_value=0)
pivot_top_20


# **Observation 5**
# 
# * The above table shows a lot of insights such as the follows:
# 
#     1. P00010742 is popular among multiple age groups of Males but only among 51-55 among Females.
#    
#     2. P00265242 is popular almost among every age group and gender.
# 
#     3. P00251242 is only popular among Females in 46-50 age group

# In[ ]:




