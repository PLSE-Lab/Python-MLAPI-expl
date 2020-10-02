#!/usr/bin/env python
# coding: utf-8

# <h1>Drug consumption Analysis</h1>
# <hr>
# <h3>Jacob Abello</h3>
# 
# <br>
# 
# <h3>Source:</h3>
# <br>
# http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
# 
# Original Owners of Database: 
# 
# <b>Elaine Fehrman, </b>
# <br>
# Men's Personality Disorder and National Women's Directorate, 
# Rampton Hospital, Retford, 
# Nottinghamshire, DN22 0PD, UK, 
# Elaine.Fehrman '@' nottshc.nhs.uk 
# <br>
# 
# <b>Vincent Egan, </b>
# <br>
# Department of Psychiatry and Applied Psychology, 
# University of Nottingham, 
# Nottingham, NG8 1BB, UK, 
# Vincent.Egan '@' nottingham.ac.uk 
# <br>
# 
# <b>Evgeny M. Mirkes </b>
# <br>
# Department of Mathematics, 
# University of Leicester, 
# Leicester, LE1 7RH, UK, 
# em322 '@' le.ac.uk 
# <br>
# 
# Donor: 
# <br>
# <b>Evgeny M. Mirkes </b>
# <br>
# Department of Mathematics, 
# University of Leicester, 
# Leicester, LE1 7RH, UK, 
# em322 '@' le.ac.uk 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/drug_consumption_str.data")
len(df)


# In[ ]:


df.head()


# In[ ]:


df.info()


# <h3>Unique values per Countries, Ethnicity, Age and Education </h3>

# In[ ]:


print("Number of Unique Countries: ", len(df.country.unique()))
print("Number of Unique Ethnicity: ", len(df.ethnicity.unique()))
print("Number of Unique Age: ", len(df.age.unique()))
print("Number of Unique Education: ", len(df.education.unique()))


# <hr>
# <h3>Let's check the distribution per countries, Ethnicity, Age, Gender and Education</h3>

# <h4>Country:</h4>

# In[ ]:


countries = df['country'].value_counts().plot(kind='pie', figsize=(8, 8))


# <h4>Ethnicity:</h4>

# In[ ]:


ethnicity = df['ethnicity'].value_counts().plot(kind='pie', figsize=(8, 8))


# <h4>Age:</h4>

# In[ ]:


age = df['age'].value_counts().plot(kind='pie', figsize=(8, 8))


# <h4>Gender</h4>

# In[ ]:


gender = df['gender'].value_counts().plot(kind='pie', figsize=(8, 8))


# <h4>Educational Background:</h4>

# In[ ]:


education = df['education'].value_counts().plot(kind='pie', figsize=(8,8))


# <h3>**Suprisingly most of the users have a university degree or some college experience, even the master's degree is larger than the people who left school..</h3>

# <hr>

# <h2>Now lets classify the seven class for each drug</h2>
# 

# In[ ]:




