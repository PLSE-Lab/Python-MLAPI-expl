#!/usr/bin/env python
# coding: utf-8

# 
# 
# ![](http://www.kcarplaw.com/wp-content/uploads/2016/11/bigstock-Unemployment-rate-loose-job-lo-141287576-800x490.jpg)
# 
# **INTRODUCTION**
# 
# 1. Check the missing data
# 2. Which state has the most data
# 3. Take the mean of rate and visualize the data
# 4. Showing ten states with exact 
# 5. Which county has highly yearly fluctations

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# read salaries datasets
unemployment = pd.read_csv('../input/output.csv',low_memory=False)


# In[3]:


#check the head of data
unemployment.head()


# In[4]:


#check the back of the data
unemployment.tail()


# In[5]:


#info about data
unemployment.info()


# In[6]:


#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(unemployment.isnull(),cbar=False)


# In[17]:


#Which state has the most data
color = sns.color_palette()
cnt_srs = unemployment.State.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('States', fontsize=12)
plt.title('Count the states', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[44]:


# take the mean of rate state by state
grouped_df = unemployment.groupby(["State"])["Rate"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['State'].values, grouped_df['Rate'].values, alpha=0.8, color=color[2])
plt.ylabel('Mean rate', fontsize=12)
plt.xlabel('States', fontsize=12)
plt.title("Average of mean", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-instacart


# In[42]:


#see the number of unique states
unemployment.State.nunique()


# In[45]:


#See exact numbers
make_total = unemployment.pivot_table("Rate",index=['State'],aggfunc='mean')
topstate=make_total.sort_values(by='Rate',ascending=False)[:47]
print(topstate)


# In[39]:


#Calculate  which models has highest yearly fluncations
maketotal_1 = unemployment.pivot_table(values='Rate',index=['Month','State','County'],aggfunc=np.std)
df1 = maketotal_1.reset_index().dropna(subset=['Rate'])
df2 = df1.loc[df1.groupby('State')['Rate'].idxmax()]
for index,row in df2.iterrows():
    print(row['State'],"State which",row['County'],"has the highest yearly fluncation.")

