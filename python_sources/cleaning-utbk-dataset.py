#!/usr/bin/env python
# coding: utf-8

# We need to load the dataset and get an understanding about it.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_major = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/majors.csv')
df_universities = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/universities.csv')
df_score_humanities = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/score_humanities.csv')
df_score_science = pd.read_csv('../input/indonesia-college-entrance-examination-utbk-2019/score_science.csv')


# **df_major** 

# what we gonna do with df_major are (1) set id_major as index, (2) NaN values, and (3) make sure that id_university digits is equal to three first digit of id_major  [](http://)

# In[ ]:


df_major.drop(df_major.columns[0], axis=1, inplace = True)
df_major.set_index('id_major', inplace = True)


# In[ ]:


df_major.isna().values.sum()


# In[ ]:


df_major[df_major.index.values // 10000 != df_major.id_university.values].shape


# **df_universities**

# with df_universities we gonna set id_universities as index

# In[ ]:


df_universities.drop(df_universities.columns[0], axis=1, inplace = True)
df_universities.set_index('id_university', inplace = True)


# for easy to use purpose, we merge df_major and df_universities with id_university as a key.
# we will export it as a new csv file (you can run it if you want)

# In[ ]:


df_major = df_major.reset_index().merge(df_universities, on='id_university', how="left")
df_major.set_index('id_major', inplace = True)


# In[ ]:


#df_major.to_csv("majors_fix.csv") #make a new dataset as majors_fix


# For df_score_humanities and df_score_science, we repeat the same process like before and export it as new csv file. We have additional step to look for invalid id_first_major or id_second_university, normaly its have length of 7 and look for unmatched id_first_university with id_first_major or id_second_university with id_second_university.

# **df_score_humanities**

# In[ ]:


#set id_user as index
df_score_humanities.drop(df_score_humanities.columns[0], axis=1, inplace = True)
df_score_humanities.set_index('id_user', inplace = True)


# In[ ]:


#check NaN
df_score_humanities.isna().values.sum()


# In[ ]:


#Look for invalid id_first_major or id_second_university, normaly its have length of 7
df_score_humanities = df_score_humanities[df_score_humanities.id_first_major > 1000000]


# In[ ]:


#Look for unmatched id_first_university with id_first_major or id_second_university with id_second_university
df_score_humanities[(df_score_humanities.id_first_major.values // 10000 != df_score_humanities.id_first_university.values) 
                    |(df_score_humanities.id_second_major.values // 10000 != df_score_humanities.id_second_university.values)]


# In[ ]:


#Fixing unmatched id_first_university or id_second_university
df_score_humanities.id_first_university = df_score_humanities.id_first_major // 10000
df_score_humanities.id_second_university = df_score_humanities.id_second_major // 10000


# In[ ]:


#checking it again
df_score_humanities[(df_score_humanities.id_first_major.values // 10000 != df_score_humanities.id_first_university.values) 
                    |(df_score_humanities.id_second_major.values // 10000 != df_score_humanities.id_second_university.values)]


# In[ ]:


#We also calculate average score and store it as new column
df_score_humanities['avg_score'] = df_score_humanities.iloc[:,4:].mean(axis=1)


# In[ ]:


#After all the cleaning process and adding new column. We store all the data into one new dataset. We adding new column as choice, it reminds us that id_user has two different choice when choosing major, first choice and second choice.
df_score_humanities_1 = df_score_humanities[["id_first_major", "id_first_university", "avg_score"]].copy().reset_index()
df_score_humanities_1["choice"] = 1
df_score_humanities_1.rename(columns={"id_first_major": "id_major", "id_first_university": "id_university"}, inplace=True)

df_score_humanities_2 = df_score_humanities[["id_second_major", "id_second_university", "avg_score"]].copy().reset_index()
df_score_humanities_2["choice"] = 2
df_score_humanities_2.rename(columns={"id_second_major": "id_major", "id_second_university": "id_university"}, inplace=True)

df_score_humanities_all = pd.concat([df_score_humanities_1, df_score_humanities_2])
df_score_humanities_all["type"] = "humanity"
df_score_humanities_all.shape


# In[ ]:


df_score_humanities_all.sample(10)


# In[ ]:


#df_score_humanities_all.to_csv("score_humanities_all.csv") #make a new dataset as score_humanities_all


# **df_score_science**

# What we have done with df_score_humanities. we repeat the process with df_score_science.

# In[ ]:


#set id_user as index
df_score_science.drop(df_score_science.columns[0], axis=1, inplace = True)
df_score_science.set_index('id_user', inplace = True)


# In[ ]:


#check NaN
df_score_science.isna().values.sum()


# In[ ]:


#Look for invalid id_first_major or id_second_university, normaly its have length of 7
df_score_science = df_score_science[df_score_science.id_first_major > 1000000]


# In[ ]:


#Look for unmatched id_first_university with id_first_major or id_second_university with id_second_university
df_score_science[(df_score_science.id_first_major.values // 10000 != df_score_science.id_first_university.values) 
                    |(df_score_science.id_second_major.values // 10000 != df_score_science.id_second_university.values)]


# In[ ]:


#Fixing unmatched id_first_university or id_second_university
df_score_science.id_first_university = df_score_science.id_first_major // 10000
df_score_science.id_second_university = df_score_science.id_second_major // 10000


# In[ ]:


#checking it again
df_score_science[(df_score_science.id_first_major.values // 10000 != df_score_science.id_first_university.values) 
                    |(df_score_science.id_second_major.values // 10000 != df_score_science.id_second_university.values)]


# In[ ]:


#We also calculate average score and store it as new column
df_score_science['avg_score'] = df_score_science.iloc[:,4:].mean(axis=1)


# In[ ]:


#After all the cleaning process and adding new column. We store all the data into one new dataset. We adding new column as choice, it reminds us that id_user has two different choice when choosing major, first choice and second choice.
df_score_science_1 = df_score_science[["id_first_major", "id_first_university", "avg_score"]].copy().reset_index()
df_score_science_1["choice"] = 1
df_score_science_1.rename(columns={"id_first_major": "id_major", "id_first_university": "id_university"}, inplace=True)

df_score_science_2 = df_score_science[["id_second_major", "id_second_university", "avg_score"]].copy().reset_index()
df_score_science_2["choice"] = 1
df_score_science_2.rename(columns={"id_second_major": "id_major", "id_second_university": "id_university"}, inplace=True)

df_score_science_all = pd.concat([df_score_science_1, df_score_science_2])
df_score_science_all["type"] = "science"
df_score_science_all.shape


# In[ ]:


df_score_science_all.sample(10)


# In[ ]:


#df_score_science_all.to_csv("score_science_all.csv") #make a new dataset as score_science_all


# In[ ]:




