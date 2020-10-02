#!/usr/bin/env python
# coding: utf-8

# <h1>**Performance of Students in Exams**

# This dataset contains information about marks secured by students in their college exams.

# <h1>Objective

# To explore the dataset.
# To extract useful information out of it.
# To visualize the information in easily understandable way.
# 

# <h2>Let's Get Started.

# **Import the required libraries.**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Extract the dataset it in a dataframe named 'data'**

# In[ ]:


data= pd.read_csv('../input/StudentsPerformance.csv')


# **Understanding the number of rows and columns in data**

# In[ ]:


data.shape


# **Showing the top 5 rows in our dataset.**

# In[ ]:


data.head()


# Check the data for null values, whether it needs cleansing or not.

# In[ ]:


data.isnull().sum()


# Since all the values are 0, it means we are good to go!!

# Let's add another column of 'total score' to check for the overall performance of students.

# In[ ]:


data['total_score']= data['math score']+ data['reading score']+ data['writing score']
data.head()


# Here we have got a new column with total score of students in all the three subjects.

# **Now, explore more about our data.**

# In[ ]:


data['gender'].unique()


# It means that we have 2 unique values in gender column, i.e. 'female' and 'male'. We can further use this information to understand how gender played a role in marks of students.

# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", kind='count' , palette= "ch:.25", data=data);


# In[ ]:


data['race/ethnicity'].unique()


# Here, we have 5 unique values in 'race/ethnicity' column, i.e. Group A- E. We can further utilize this information in understanding how race/ethinicity of students influenced their marks.

# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="race/ethnicity", kind='count' , palette= "ch:.25", data=data);


# In[ ]:


data['parental level of education'].unique()


# Here, we have 6 unique values in 'parental level of education' column. We can further use this information in understanding how parental level of education influenced their children's marks.

# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="parental level of education", kind='count' , palette= "pastel", data=data);


# In[ ]:


data['lunch'].unique()


# We can see two values in the 'lunch' column. We will explore more about this later.

# In[ ]:


data['test preparation course'].unique()


# Here we got to know that we have 2 values, none and completed which can be further used to understand whether test preparation course helped the students or not.

# In[ ]:


data.groupby('gender', axis = 0)['lunch'].count()


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="math score", kind='swarm' ,data=data);


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="reading score", kind='swarm' ,data=data);


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="writing score", kind='swarm' ,data=data);


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="total_score", kind='swarm' ,data=data);


# This analysis is still incomplete.!! Thanks..
