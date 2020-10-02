#!/usr/bin/env python
# coding: utf-8

# <img src="https://image.ibb.co/dGvhMT/naukri_logo.jpg" alt="naukri logo" border="0" />

# In[179]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Read data sheet**

# In[180]:


basic_df = pd.read_csv('../input/naukri_com-job_sample.csv')
#display first five rows
basic_df.head(5)


# In[181]:


#display last five rows
basic_df.tail(5)


# In[182]:


#display all columns datatypes
basic_df.dtypes


# In[183]:


#data frame shape
basic_df.shape


# **Payrate columns split** 

# In[184]:


pay_split = basic_df['payrate'].str[1:-1].str.split('-', expand=True)
pay_split.head()


# **Remove character and comma in pay_split data frame first, second columns**

# **pay_split first column**

# In[185]:


#remove space in left and right 
pay_split[0] =  pay_split[0].str.strip()
#remove comma 
pay_split[0] = pay_split[0].str.replace(',', '')
#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
pay_split[0] = pay_split[0].str.replace(r'\D.*', '')
#display 
pay_split[0]


# **pay_split second column**

# In[186]:


#remove space in left and right 
pay_split[1] =  pay_split[1].str.strip()
#remove comma 
pay_split[1] = pay_split[1].str.replace(',', '')
#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
pay_split[1] = pay_split[1].str.replace(r'\D.*', '')
#display 
pay_split[1]


# **pay_split first and second columns change data type string to float **

# In[187]:


pay_split[0] = pd.to_numeric(pay_split[0], errors='coerce')
pay_split[0]


# In[188]:


pay_split[1] = pd.to_numeric(pay_split[1], errors='coerce')
pay_split[1]


# ** Select basic_df useful columns**

# In[189]:


main_df = basic_df[['company','education','experience','industry','jobdescription','joblocation_address','jobtitle','numberofpositions','postdate','skills']]


# ** Rename columns **

# In[190]:


main_df.rename(columns={'jobdescription':'description', 'joblocation_address':'address', 'numberofpositions':'npositions'}, inplace=True)

main_df.head(2)


# ** Insert min_payment and max_payment column **

# In[191]:


main_df.insert(0, 'max_payment', pay_split[1])
main_df.insert(0, 'min_payment', pay_split[0])


# In[192]:


main_df.head(100)


# ** Split experience column and find minimum and maximum experience required **

# In[193]:


experience_split = basic_df['experience'].str[1:-1].str.split('-', expand=True)
experience_split.head()


# In[194]:


#remove space in left and right 
experience_split[0] =  experience_split[0].str.strip()
#remove comma 
experience_split[0] = experience_split[0].str.replace(',', '')
#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
experience_split[0] = experience_split[0].str.replace(r'\D.*', '')
#display 
experience_split[0]


# In[195]:


#remove space in left and right 
experience_split[1] =  experience_split[1].str.strip()
#remove comma 
experience_split[1] = experience_split[1].str.replace(',', '')
#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
experience_split[1] = experience_split[1].str.replace(r'\D.*', '')
#display 
experience_split[1]


# In[196]:


experience_split[0] = pd.to_numeric(experience_split[0], errors='coerce').fillna(0).astype(np.int64)
experience_split[1] = pd.to_numeric(experience_split[1], errors='coerce').fillna(0).astype(np.int64)
experience_split.head(5)


# In[197]:


experience_split.dtypes


# In[198]:


main_df.head(2)


# In[199]:


main_df.insert(8, 'min_experience', experience_split[0])
main_df.insert(9, 'max_experience', experience_split[1])


# In[200]:


main_df.head(3)


# In[201]:


main_df['postdate'] = main_df['postdate'].str.strip()
main_df['postdate'] = main_df['postdate'].str.replace('0000','')
main_df['postdate'] = main_df['postdate'].str.replace('+','')
main_df['postdate'] = main_df['postdate'].str.strip()
main_df['postdate'] = pd.to_datetime(main_df['postdate'])


# In[202]:


main_df.dtypes


# In[212]:


plt.figure(figsize=(10, 5))
main_df['min_payment'].hist(rwidth=0.9, bins=15, color='#10547c')
plt.title('Minimum payment')
plt.xlabel('Payment')
plt.ylabel('Position')
plt.show()


# In[204]:


plt.figure(figsize=(10, 5))
main_df['max_payment'].hist(rwidth=0.9, bins=15, color='r')
plt.title('Maximum payment')
plt.xlabel('Payment')
plt.ylabel('Position')

plt.show()


# In[205]:


main_df[main_df.max_payment > 5000000].shape


# In[206]:


main_df[main_df.max_payment > 5000000]


# In[207]:


plt.figure(figsize=(10, 5))
main_df['min_experience'].hist(rwidth=0.9, bins=15, color='#ff5722')
plt.title('Min Experience')
plt.xlabel('Experience year')
plt.ylabel('Position')

plt.show()


# In[208]:


plt.figure(figsize=(10, 5))
main_df['max_experience'].hist(rwidth=0.9, bins=15, color='#04ab7d')
plt.title('Max Experience')
plt.xlabel('Experience year')
plt.ylabel('Position')
plt.show()


# In[209]:


main_df[main_df.max_experience > 20].shape


# In[210]:


main_df[main_df.max_experience > 20]


# In[211]:


main_df[main_df.max_experience > 23]


# In[224]:


plt.figure(figsize=(10, 5))
main_df['npositions'].hist(rwidth=0.9, bins=15, color='#04ab7d')
plt.title('Number of positions')
plt.xlabel('Experience year')
plt.ylabel('Position')
plt.show()


# In[267]:


plt.figure(figsize=(15, 5))
main_df.loc[main_df.npositions > 10].loc[:,['npositions']].hist(rwidth=0.9, bins=15, color='#04ab7d')
plt.title('Number of positions')
plt.show()


# In[295]:


max_positions = main_df.loc[main_df.npositions > 100].loc[:,['npositions','industry']]
plt.figure(figsize=(15, 5))
hist_position_value = pd.value_counts(max_positions.industry)
hist_position_value.index
hist_position_value[hist_position_value >1].plot(kind='bar')
plt.show()

