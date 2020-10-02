#!/usr/bin/env python
# coding: utf-8

# ** Consider the following Python dictionary data and Python list labels:**
# 
# data = {'birds': ['Cranes', 'Cranes', 'plovers', 'spoonbills', 'spoonbills', 'Cranes', 'plovers', 'Cranes', 'spoonbills', 'spoonbills'],
#         'age': [3.5, 4, 1.5, np.nan, 6, 3, 5.5, np.nan, 8, 4],
#         'visits': [2, 4, 3, 4, 3, 4, 2, 2, 3, 2],
#         'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
# 
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# 

# In[ ]:


import pandas as pd
import numpy as np

data = {'birds': ['Cranes', 'Cranes', 'plovers', 'spoonbills', 'spoonbills', 'Cranes', 'plovers', 'Cranes', 'spoonbills', 'spoonbills'], 'age': [3.5, 4, 1.5, np.nan, 6, 3, 5.5, np.nan, 8, 4], 'visits': [2, 4, 3, 4, 3, 4, 2, 2, 3, 2], 'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


# **1. Create a DataFrame birds from this dictionary data which has the index labels.**

# In[ ]:


df = pd.DataFrame(data)
df


# **2. Display a summary of the basic information about birds DataFrame and its data.**

# In[ ]:


df.describe()


# **3. Print the first 2 rows of the birds dataframe **

# In[ ]:


print(df.head(2))


# **4. Print all the rows with only 'birds' and 'age' columns from the dataframe**

# In[ ]:


print(df['birds'])
print(df['age'])


# **5. select [2, 3, 7] rows and in columns ['birds', 'age', 'visits']**

# In[ ]:


print(df['birds'].iloc[2], df['age'].iloc[2])
print(df['age'].iloc[3], df['age'].iloc[3])
print(df['visits'].iloc[7], df['age'].iloc[7])


# **6. select the rows where the number of visits is less than 4**

# In[ ]:


df[df['visits'] < 4]


# **7. select the rows with columns ['birds', 'visits'] where the age is missing i.e NaN**

# In[ ]:


df[df['age'].isnull()]


# **8. Select the rows where the birds is a Cranes and the age is less than 4**

# In[ ]:


df[(df['birds'] == 'Cranes') & (df['age'] < 4)]


# **9. Select the rows the age is between 2 and 4(inclusive)**

# In[ ]:


df[(df['age'] >= 2) & (df['age'] <= 4)]


# **10. Find the total number of visits of the bird Cranes**

# In[ ]:


df[(df['birds'] == 'Cranes') & (df['visits'] > 0)].sum()


# **11. Calculate the mean age for each different birds in dataframe.**

# In[ ]:


df[['age']].mean()


# **12. Append a new row 'k' to dataframe with your choice of values for each column. Then delete that row to return the original DataFrame.**

# In[ ]:


s = df.xs(3)
df.append(s, ignore_index=True)
df.drop([df.index[9]])


# **13. Find the number of each type of birds in dataframe (Counts)**

# In[ ]:


df.groupby(df["birds"]).count()


# **14. Sort dataframe (birds) first by the values in the 'age' in decending order, then by the value in the 'visits' column in ascending order.**

# In[ ]:


sort_by_age = df.sort_values('age')
print(sort_by_age.head())
print("------------------------------------")
sort_by_vists = df.sort_values('visits',ascending=False)
print(sort_by_vists.head())


# **15. Replace the priority column values with'yes' should be 1 and 'no' should be 0**

# In[ ]:


df.priority.map(dict(yes=1, no=0))


# **16. In the 'birds' column, change the 'Cranes' entries to 'trumpeters'.**

# In[ ]:


df.birds.map(lambda x: 'trumpeters' if x=='Cranes' else x)

