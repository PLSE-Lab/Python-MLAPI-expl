#!/usr/bin/env python
# coding: utf-8

# **PANDAS**
# 
# Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.

# In[74]:


import pandas as pd # Standard way to import pandas


# In[75]:


# Create data for table/dataframe
header = ["Name", "Attendance", "CGPA"]
rows = [["Mihir", 93, 8.9],
       ["Suriya", 90, 9.1],
       ["Kumar", 96, 8.2],
       ["Paul", 85, 7.9],
       ["Biju", 97, 9.5],
       ["Dipak", 82, 8.4]]


# In[76]:


df = pd.DataFrame(rows, columns=header)


# In[77]:


df


# In[78]:


df.head() # display top 5 rows


# In[79]:


df.columns


# **CSV Files**
# 
# Comma-separated files : Contain data in the form of tables.
# 
# Each row is usually separated by a newline character '\n' (not always)
# 
# Each column is usually separated by a comma ',' or space ' ' or tab '\t' (not always)

# In[11]:


df.to_csv("attendance.csv", header=True, index=False) # Write the dataframe to CSV file


# In[12]:


df = pd.read_csv("attendance.csv")


# In[13]:


df


# In[14]:


print(df.Attendance.mean())
# OR
print(df["Attendance"].mean())


# In[15]:


df.CGPA.std() # standard deviation


# In[16]:


df.sort_values(["Attendance"]) # Sort ASC based on `Attendance` column


# In[17]:


df.sort_values(["Attendance"]).reset_index(drop=True) # Reset index


# In[18]:


df.values # get the numpy array from the dataframe
# NOTE: dtype of array is `object`


# In[72]:


df_copy = df.copy()
df_copy.Attendance = df_copy.Attendance.astype(float) # change dtype of pandas dataframe
df_copy


# In[22]:


print(df.CGPA.values)
ind = df.CGPA.values.argmax() # Get index with maximum CGPA
print(ind)
df.loc[ind]                   # Get row with index `ind`


# In[23]:


df.CGPA.plot()


# In[24]:


df.Attendance.plot(c='r')


# In[25]:


df[["Attendance", "CGPA"]].plot()


# In[26]:


df["Age"] = [45, 23, 43, 34, 29, 40] # Add new column


# In[27]:


df


# In[28]:


df["useless_column"] = [1, 2, 3, 4] # Try to add column with different number of rows


# In[29]:


df.shape


# In[30]:


df.loc[df.shape[0]] = ["Anand", 60, 9.0, 42] # Add new entry


# In[31]:


df


# In[32]:


print(df.shape)
# OR
print(df.values.shape)


# In[37]:


# IMPORTANT - `loc` function
# .loc can take one or two indices.
# If both are given, first is for rows, second is for columns
df_copy = df.copy()

df_copy.loc[df_copy.Attendance < 75, "Attendance"] = 75.0

print(df)
print()
print(df_copy)


# In[38]:


df.loc[df.shape[0]] = ["Kamal", 57, 7.0, 39] # Add new entry


# In[41]:


df_copy = df.copy()
print("Attendance < 75% : ", (df_copy.Attendance < 75).values)
print("CGPA > 8.5       : ", (df.CGPA > 8.5).values)

indices = (df_copy.Attendance < 75).values * (df.CGPA > 8.5).values # students with CGPA > 8.5 and attendance < 75%
df_copy.loc[indices, "Attendance"] = 75.0
df_copy


# In[42]:


df


# In[47]:


df.loc[3: 5, ["CGPA", "Name"]] # NOTE: 3: 5 - both 3 and 5 are inclusive


# In[48]:


df.loc[3: 4] # if only one index is provided, all columns are included, i.e. equivalent to .loc[3: 4, :]


# In[ ]:





# In[53]:


print(df.Attendance.values)
df.Attendance.hist(figsize=(10, 6))

