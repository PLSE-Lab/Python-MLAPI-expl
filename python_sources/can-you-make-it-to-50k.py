#!/usr/bin/env python
# coding: utf-8

# In this analysis, we will focus more on EDA and the insights that we can get from this dataset.

# # Problem statement

# The objective is to determine whether a person makes more than 50K a year. For this problem, we will use Logistic Regression.

# # Procedure

# A typical and well-known process for a data analysis is OSEMN, which stands for:
# 1. Obtain: the data must be obtained before any other activities can be conducted
# 2. Scrubbing: the next step is to clean, format and/or re-arrange the data for easier analysis and model training
# 3. Exploring: exploration of the whole dataset using descriptive statistics and visualisations. The task is to find any statistical significance that exists and do hypothesis testing (1-t test or relational testing). Based on the newly gained information, we will conduct feature extraction to determine which variables are suitable for our logistic model.
# 4. Modeling: the data will be split into training set and test set. We will feed the training set into the model and later feed the test set to see how our model performs. A feature selection process using DecisionTree can further improve our accuracy by providing a more significant variables
# 5. INterpreting: what insight or meaningful information can we get from the dataset? Is our choice of a predictive model good enough? What business questions can we answer with this data?

# ## 1. Obtaining the data

# This is a publicly available data so we don't have to collect and scrape the data, just need to load into memory

# In[ ]:


# Import necessary packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load the data into memory
df = pd.read_csv('../input/income-classification/income_evaluation.csv')


# In[ ]:


# Let's take a look at our data
df.head()


# In[ ]:


df.tail()


# In[ ]:


# 32k rows of data and we have 15 variables
df.shape


# ## 2. Scrubbing the data

# In[ ]:


df.info()


# There's no sight of missing data so we don't have to do any cleaning (kudos to the guys that scraped this)

# In[ ]:


df.isnull().any()


# There is one column that has quite a obscure name: fnlwgt.
# Upon further inspection, this variable is translated as 'final weight': the total number of ppl that fits to this particular row of information.
# Another thing to notice is how every names have a space before it. We need to remove it.
# 
# From my own rule, variable names should be clear and concise. Names with too more words should be separated with '_'.

# In[ ]:


df = df.rename(columns={'age': 'age',
                         ' workclass': 'workclass',
                         ' fnlwgt': 'final_weight',
                         ' education': 'education',
                         ' education-num': 'education_num',
                         ' marital-status': 'marital_status',
                         ' occupation': 'occupation',
                         ' relationship': 'relationship',
                         ' race': 'race',
                         ' sex': 'sex',
                         ' capital-gain': 'capital_gain',
                         ' capital-loss': 'capital_loss',
                         ' hours-per-week': 'hrs_per_week',
                         ' native-country': 'native_country',
                         ' income': 'income'
                        })
df.columns


# Somes of the variables have values that are binary or discrete. We can apply encoding or transform some of the string variables to category.
# Since 'income' is our target variables, we want it to be numerical for easier computation. I will create a new variables derived from the 'income'.

# In[ ]:


df['income'].unique()


# In[ ]:


df['income_encoded'] = [1 if value == ' >50K' else 0 for value in df['income'].values]
df['income_encoded'].unique()


# In[ ]:


# look better now
df.sample(5)


# ## 3. Exploring the data

# In this part, we will look at how each variables relate to each other.

# In[ ]:


# Let's check some descriptive statistics
df.describe()


# - In our sample, the mean and median age (50% percentile) are similar, I guess that this will be a normal distribution, we will check it later using visualisations.
# - The capital gain and loss variables are suspicious. All observations that are higher than 0 is in the 4th quartile.
# - In the 'hrs_per_week' columns, the min is 1 and max is 99, which are not really common in real life. We will have to investigate this later.
# - Only about one fourth of all the people are able to earn more than 50K a year.

# We will then proceed to inspect the relationships and dependency of our target variable (income) and other independent variables.
# One of the way to view the overall relationships between variables is heatmap. The seaborn package give us access to really nice graphs.

# In[ ]:


# create a blank canvas
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.title('Overview heatmap')


# There's not much of a statistical significance between our target variable and the numerical variable. The highest correlation is the one between 'income' and 'education_num'. Although, there is a relatively noticable correlations between 'income' and 'age', 'hrs per week', 'and 'capital gain'. We will have to tread carefully since this might be a case of multicollinearity.

# Next step is to look closely into how each variables affect 'income'. I will attempt to do a comprehensive analysis of each significant variables.
# 
# 

# ### 3.1. Education_num and 'income'

# In[ ]:


plt.figure(figsize=(10, 10))
sns.boxplot('income', 'education_num', data=df)


# - Seems like we have a lot of irregularity in this case. People that make more than 50K tend to have a higher education_num. <br>
# - People who make less than than are more likely to have an average of 9 to 10, with a few cases will really high or low education_num.

# ### 3.2. age and 'income'

# In[ ]:


plt.figure(figsize=(10, 10))
sns.boxplot('income', 'age', data=df)


# The bulk of the *rich* are converged in the '35-50' age groups, so if you are under 30 and still struggle financially, don't settle until 50.
# Even so, in our 32K rows of data, people well over 80 are still living confortably.
# 

# ### 3.3. working hrs a week and 'income'

# In[ ]:


plt.figure(figsize=(10, 10))
sns.boxplot('income', 'hrs_per_week', data=df)


# If you like most people, and only work for 40 hrs a week or less, chance is, you won't be living that life once you get to your 30s.
# There are noticeably a lot of outliners, we will have to investigate this variable further, see how it fares with other variables.

# **THINK**
# - Does hours here imply a full-time 9-5 job, or does it also account for side gigs and side-projects? How to do we define 'a working hour'?
# - What kinds of jobs can you work for less than 40hrs and earn more?
# - what kinds of jobs that requires you to work more but you are still underpaid?

# ### 3.4. education and income

# Before we dive into 'occupation' and 'income' analysis, I want to take a quick look over education. The consensus here is that if you have a degree, you salary is expected to be higher than those who don't have one.
# 
# 

# In[ ]:


df['education'].value_counts().to_frame()


# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot('education', hue='income', data=df)


# Some insights:
# - People with only high school degree or lower have a very slim chance of comfortably being compensated
# - College graduates are more likely to have a higher income, compared to HS grads or drop-outs.
# - There are more people that have a master, PhD, or are currently employed as tenured professor earning more than 50K than people with the same qualification earning less than 50K
# - More than three fourth of doctors and tenured professorsors earn more than 50K

# HS grads and drop-outs are the bulk of this dataset. Lets remove them and inspect other values more closely

# In[ ]:


df_filtered = df.isin({'education': ["HS-grad", "Some-college"]})
df_filtered.any()
#plt.figure(figsize=(20, 10))
#sns.countplot('education', hue='income', data=df[df_filtered])


# ### 3.5 Occupation and income

# Lets see how each profession fare by comparing how many people earn more than 50K. We will look at the total amount of workers for each field and the total number of people earning more than 50K in each.

# In[ ]:


df['occupation'].value_counts().head(3)


# In[ ]:


df[df['income'] == ' >50K']['occupation'].value_counts().head(3)


# In[ ]:


pd.crosstab(df["occupation"], df['income']).plot(kind='barh', stacked=True, figsize=(20, 10))


# #### Insight
# - Top 3 profession by total counts are Prof-specialty, Craft-repair, Exec-managerial
# - Top 3 profession by total counts of people earning more than 50K (in order) are Executive-Managerials, Profession-specialties, and sales and craft-repairs (with close margin)
# - Executive-Managerials has the highest percentage of people earning more than 50K: 48%

# In[ ]:




