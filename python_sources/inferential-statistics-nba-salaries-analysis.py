#!/usr/bin/env python
# coding: utf-8

# # Usage of Inferential Statistics to analyse NBA salaries

# ---

# ## Author:
# [__Gleisson Bispo__](https://github.com/gleissonbispo)
# 

# ## Hypothesis:
# In the 2017-2018 season did any NBA player have a **higher** or **lower** salary than the average?
# 

# ## Dataset:
# __[Kaggle: NBA Player Salary Dataset (2017 - 2018)](https://www.kaggle.com/koki25ando/salary)__

# ![](https://sportshub.cbsistatic.com/i/r/2018/09/15/f0e813c2-ad7f-453e-855d-097d9f4feed7/thumbnail/770x433/cdf43928ded227cc4f95dd2b8d702116/top100-cover.png)
# __<center> Let's go! </center>__
# 
# ---

# ## Importing Libraries and Reading Data

# In[11]:


#Libraries
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

#Ignoring warnings
import warnings
warnings.filterwarnings("ignore")


# In[12]:


#Reading CSV Dataset
df_NBA = pd.read_csv(r'../input/NBA_season1718_salary.csv')
df_NBA.head()


# In[13]:


#DF Key information
df_NBA.info()


# In[14]:


#Renaming and deleting columns
df_NBA.columns = ['cod', 'player', 'team', 'salary']
del df_NBA['cod']
df_NBA.head()


# ---
# ## Visualizing players by team

# In[15]:


team_index = df_NBA['team'].value_counts()
sns.catplot(data=df_NBA,
            x='team',
            order=team_index.index,
            kind='count',
            aspect=2.5,
            palette='GnBu_d')


# ## Salary Distribution

# In[16]:


#Histogram and KDE
plt.figure(figsize=(8, 4))
sns.distplot(df_NBA['salary'], bins=40)


# In[17]:


#Probability Density Function (PDF) Chart
x = df_NBA['salary']

plt.figure(figsize=(8, 4))
plt.plot(x, st.norm.pdf(x, x.mean(), x.std()))
plt.show()


# _Based on the probability density function plot is possible to identify a normal distibution, however, with a huge bilateral symmetry (right). Using a logarithm function is possible to "correct" it._

# ## Normalizing the salary data

# In[18]:


#Creating a column with the salary log to normalize the distribution
df_NBA['salary_log'] = np.log1p(df_NBA['salary'])
sns.distplot(df_NBA['salary_log'], bins=25)


# In[19]:


#Dividing by the mean and standard deviation to standardize the serie in a new column
df_NBA['norm_log_salary'] = ((df_NBA['salary_log'] - df_NBA['salary_log'].mean()) / df_NBA['salary_log'].std())
sns.distplot(df_NBA['norm_log_salary'], bins=25)


# ## Mean and Standard Deviation

# In[20]:


print(f"""Mean: {df_NBA.norm_log_salary.mean():.4f}
Standard: {df_NBA.norm_log_salary.std():.4f}""")


# ## P-Value to  2 std

# In[21]:


norm_mean = df_NBA.norm_log_salary.mean()
norm_std = df_NBA.norm_log_salary.std()

p_value = st.norm(norm_mean, norm_std).sf(2*norm_std) * 2 #to sides
p_value


# ## Calculating z-score

# In[22]:


z_score_inf = st.norm.interval(alpha=0.95, loc=norm_mean, scale=norm_std)[0]
z_score_sup = st.norm.interval(alpha=0.95, loc=norm_mean, scale=norm_std)[1]

print(f'{z_score_inf:.4f} <--------> {z_score_sup:.4f}')


# __With the Alpha limits of 0.95 we can run the inference and find out which players are earning above or below average with a 95% confidence level.__

# ---

# ## Analysing Results
# 

# ## Hypothesis:
# In the 2017-2018 season did any NBA player have a **higher** or **lower** salary than the average?
# 

# ---
# ### __1. Lower than the average__
# 

# In[23]:


#Players
df_NBA_lower = df_NBA[df_NBA['norm_log_salary'] < z_score_inf]
df_NBA_lower


# In[24]:


#Players by team
team_index = df_NBA_lower['team'].value_counts()
team_index


# In[25]:


#Plot players by team
plt.figure(figsize=(12, 5))
sns.countplot(df_NBA_lower['team'],
              order=team_index.index,
              palette='Blues_r')


# In[26]:


print(f"""Players with a lower salary than the average: 
Total - {df_NBA_lower.shape[0]}
Rate - {df_NBA_lower.shape[0] / df_NBA.shape[0] * 100:.2f}%""")


# Based on that Dataset we can affirm with 95% certainty that There are currently __45 players__ earning less than the average salary. This represents a total of __7.85%__.
# Therefore: **Fail to reject H0**
# 
# ---

# ### __2. Higher than the average__
# 

# In[27]:


#Players
df_NBA_higher = df_NBA[df_NBA['norm_log_salary'] > z_score_sup]
df_NBA_higher


# In[28]:


print(f"""Players with a higher salary than the average: 
Total - {df_NBA_higher.shape[0]}
Rate - {df_NBA_higher.shape[0] / df_NBA.shape[0] * 100:.2f}%""")


# In[29]:


#p-value and alpha max to the highest salary
p_value = st.norm(norm_mean, norm_std).sf(df_NBA['norm_log_salary'].max())
alpha = 1 - p_value
print(f'P-value: {p_value:.3f}\nAlpha Max: {alpha:.3f}\nWe can confirm that the highest salary is on the distribution!')


# Based on that Dataset we can affirm with 95% certainty that currently there are __no__ players earning higher than the average salary. The highest salary is on the average distribution. Therefore: **Reject H0**
