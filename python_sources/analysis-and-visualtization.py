#!/usr/bin/env python
# coding: utf-8

# # Questions
# ### 1) Which factor influenced a candidate in getting placed?
# ### 2) Does percentage matters for one to get placed?
# ### 3) Is there a relation of gender bias in placement status?
# ### 4) Which degree specialization is much demanded by corporate?
# ### 5)  How is the salary compared to HSC Subject of the candidate?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


# In[ ]:


df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv',index_col='sl_no') 
df_b=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv',index_col='sl_no') ##backup data for another set of analysis


# In[ ]:


df.head()


# ##  Which factor influenced a candidate in getting placed?

# In[ ]:


df['status'].value_counts()


# In[ ]:


a=df.corr() ## finding the correation of every column to know the relation better
a


# #### As we want to find the co-relation for all values in getting placed we need to convert some categorical values to binary form

# In[ ]:


df['workex']=df['workex'].map({'Yes':1,'No':0})
df['gender']=df['gender'].map({'M':1,'F':0})
df['status']=df['status'].map({'Placed':1,'Not Placed':0})


# In[ ]:


df.head()


# #### now we can plot the co-realtion of various features with the placement status

# In[ ]:


a=df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(a,cmap='coolwarm')
plt.title('Co-relation of every feature to see relation with placement status')
plt.show()


# ### We can see that SSC percentage has the highest priority in placement status and then the priority status is
# #### 2) hsc percentage
# #### 3) degree percentage
# #### 4) work experience
# #### 5) job quality test
# #### 6) mba percentage
# 
# ### SO we can also say that percentage does matter in getting placed

# In[ ]:


## we can further verify the above analysis by a boxplot of ssc percentage and placement status
plt.figure(figsize=(10,6))
sns.boxplot('ssc_p','status',data=df_b,palette='magma')
plt.title('SSC percantage spread vs Placement status')
plt.show()


# ### Is there a relation of gender bias in placement status?

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='gender',data=df_b,hue='status',palette='rainbow')
plt.title('Placement status v/s Gender')
plt.show()


# ### we can see there is a slight gender bias towards males in placement status but that maybe because the dataset is small

# ## Which degree specialization is much demanded by corporate?

# In[ ]:


df['specialisation'].value_counts()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='specialisation',data=df_b,hue='status',palette='plasma_r')
plt.title('PLacement numbers according to specialisation in MBA')
plt.show()


# ### We see that marketing and finance specialisation has a much more demand by corporate than marketing and HR
# #### We can further compare the salaries provided to each specialisation.

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='specialisation',y='salary',data=df,palette='winter')
plt.title('Salary comparison of different specialisation in MBA')
plt.show()


# ## How is the salary compared to HSC Subject of the candidate?

# In[ ]:


plt.figure(figsize=(10,6))
sns.violinplot(x='hsc_s',y='salary',data=df,palette='twilight')
plt.title('Salary comparison with respect to HSC subject')


# ### We can observe that candidates having commerce or science have a higher number in placements than of arts
# ### Also candidates with commerece background had a higher salary range than that of science background whereas arts background students had a limited salary

# In[ ]:




