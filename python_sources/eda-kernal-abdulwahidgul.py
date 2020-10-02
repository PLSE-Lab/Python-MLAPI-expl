#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv('../input/retail-marketing/retailMarketingDI.csv')
df.columns = df.columns.str.lower()
print("DataFrame Columns\n", df.columns.tolist())
df.head()


# In[ ]:


"Number of Rows:%s and Number of Columns:%s" %(len(df), len(df.columns))


# # understanding the difference between the Categorical (discrete) and Ordinal (continuous) Data

# In[ ]:


print('These are the categorical values')
print("age", df.age.unique().tolist())
print("gender", df.gender.unique().tolist())
print("ownhome", df.ownhome.unique().tolist())
print("married", df.married.unique().tolist())
print("location", df.location.unique().tolist())
print("history", df.history.unique().tolist())
print("catalogs", sorted(df.catalogs.unique().tolist()))
print("childrens", sorted(df.children.unique().tolist()))
print('----------------------------')
print("salary min:%s mean:%s median:%s max:%s mode:%s"% (df.salary.min(), df.salary.mean(),df.salary.median(), df.salary.max(), df.salary.mode()))
print("amount spent min:%s mean:%s median:%s max:%s mode:%s"% (df.amountspent.min(), round(df.amountspent.mean(), 2),df.amountspent.median(), df.amountspent.max(), df.amountspent.mode()))
print("amount spent min:%s mean:%s median:%s max:%s mode:%s"% (df.amountspent.min(), round(df.amountspent.mean(), 2),df.amountspent.median(), df.amountspent.max(), df.amountspent.mode()))


# In[ ]:


df['age'] = pd.Categorical(df.age, categories=['Young', 'Middle', 'Old'],ordered=True)
df['gender'] = df.gender.astype('category')
df['ownhome'] = df.ownhome.astype('category')
df['married'] = df.married.astype('category')
df['location'] = df.location.astype('category')
df['children'] = df.children.astype('category')
df['history'] = pd.Categorical(df.history, categories=['Low', 'Medium', 'High'],ordered=True)
df['catalogs'] = df.catalogs.astype('category')
print(df.dtypes)


# ## Understanding the values split of each categorical column. How much of the data in the table belongs to each category in percentage

# In[ ]:


print(100 * df.age.value_counts()/len(df))
print('---------------------')
print(100 * df.gender.value_counts()/len(df))
print('---------------------')
print(100 * df.ownhome.value_counts()/len(df))
print('---------------------')
print(100 * df.married.value_counts()/len(df))
print('---------------------')
print(100 * df.location.value_counts()/len(df))
print('---------------------')
print(100 * df.children.value_counts()/len(df))
print('---------------------')
print(100 * df.history.value_counts()/len(df))
print('---------------------')
print(100 * df.catalogs.value_counts()/len(df))
print('---------------------')


# ## Understanding the ratio of missing values. Where flase singals there are no null values, and true singals that there are missing values.

# Since we don't have the origin of the source data, we can't determine the reason for the missing values. If we dope the missing data rows, we are still left with 69.7% of data. This is not a bad start. 

# In[ ]:


print("age:" ,df.age.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("gender:" ,df.gender.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("ownhome:" ,df.ownhome.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("married:" ,df.married.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("location:" ,df.location.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("salary:" ,df.salary.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("children:" ,df.children.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("history:" ,df.history.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("catalogs:" ,df.catalogs.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')
print("amountspent:" ,df.amountspent.isnull().value_counts()/len(df) * 100)
print('-------------------------------------')


# In[ ]:


df[df.amountspent.isnull()]


# As per our expecting we should have 697 rows. But the new DataFrame has only 691 rows. Where did the 6 rows go?
# We have 0 value in the salary, the amount spent column. This signals missing value, based on the nature of the data. This is hypothesis can't be tested since we don't know enough about the data collection methods. We would drop rows where we have 0 value in the salary and amount spent column. 
# 
# By doing so we only lose 0.06% data and are left with 69.1%. This something we can work with.
# 
# We have a concept in machine learning of inferring the missing values from the other columns in the dataset. You can either use mean, mode or any other calculation for achieving this task. This improves the accuracy of data. But the caveat is that it can introduce a lot of bias as well. We don't any source to double-checking the result. 

# In[ ]:


df_drop_na = df.dropna()
df_drop_na = df_drop_na[df_drop_na.salary > 1].copy()
df_drop_na = df_drop_na[df_drop_na.amountspent != 0].copy()


# In[ ]:


len(df_drop_na)


# Final state of the cleaned DataFrame

# In[ ]:


df_drop_na.isnull().any()


# # Univariate Analysis

# ## Categoical Values

# In[ ]:


sns.countplot(x='age', data = df_drop_na)
plt.suptitle('Frequency of observations by Age')


# In[ ]:


sns.countplot(x='gender', data = df_drop_na)
plt.suptitle('Frequency of observations by Gender')


# In[ ]:


sns.countplot(x='ownhome', data = df_drop_na)
plt.suptitle('Frequency of observations by Own Home')


# In[ ]:


sns.countplot(x='married', data = df_drop_na)
plt.suptitle('Frequency of observations by Married')


# In[ ]:


sns.countplot(x='location', data = df_drop_na)
plt.suptitle('Frequency of observations by Locations')


# In[ ]:


sns.countplot(x='children', data = df_drop_na)
plt.suptitle('Frequency of observations by Children')


# In[ ]:


sns.countplot(x='catalogs', data = df_drop_na)
plt.suptitle('Frequency of observations by Catalogs')


# In[ ]:


sns.countplot(x='history', data = df_drop_na)
plt.suptitle('Frequency of observations by History')


# ## Ordinal Values

# In[ ]:


sns.distplot(df_drop_na['salary'])
plt.suptitle('Distribution of Salary')


# In[ ]:


sns.distplot(df_drop_na['amountspent'])
plt.suptitle('Distribution of Amount Spent')


# # Bivariate Analysis

# ## C-Q Categorical to Quantitative BY Salary

# In[ ]:


sns.boxplot(x='salary', y = 'age', data=df_drop_na)
plt.suptitle('Salary levels by Age')


# In[ ]:


sns.boxplot(x='salary', y = 'gender', data=df_drop_na)
plt.suptitle('Salary levels by Gender')


# In[ ]:


sns.boxplot(x='salary', y = 'married', data=df_drop_na)
plt.suptitle('Salary levels by Married')


# In[ ]:


sns.boxplot(x='salary', y = 'ownhome', data=df_drop_na)
plt.suptitle('Salary levels by Own Home')


# In[ ]:


sns.boxplot(x='salary', y = 'location', data=df_drop_na)
plt.suptitle('Salary levels by Locations')


# In[ ]:


sns.boxplot(x='salary', y = 'history', data=df_drop_na)
plt.suptitle('Salary levels by History')


# In[ ]:


sns.boxplot(x='salary', y = 'catalogs', data=df_drop_na)
plt.suptitle('Salary levels by Catelogs')


# ## C-Q Categorical to Quantitative BY Amount Spent

# In[ ]:


sns.boxplot(x='amountspent', y = 'age', data=df_drop_na)
plt.suptitle('Amount Spent levels by Age')


# In[ ]:


sns.boxplot(x='amountspent', y = 'gender', data=df_drop_na)
plt.suptitle('Amount Spent levels by Gender')


# In[ ]:


sns.boxplot(x='amountspent', y = 'ownhome', data=df_drop_na)
plt.suptitle('Amount Spent levels by Own Home')


# In[ ]:


sns.boxplot(x='amountspent', y = 'location', data=df_drop_na)
plt.suptitle('Amount Spent levels by Location')


# In[ ]:


sns.boxplot(x='amountspent', y = 'children', data=df_drop_na)
plt.suptitle('Amount Spent levels by Children')


# In[ ]:


sns.boxplot(x='amountspent', y = 'history', data=df_drop_na)
plt.suptitle('Amount Spent levels by History')


# In[ ]:


sns.boxplot(x='amountspent', y = 'catalogs', data=df_drop_na)
plt.suptitle('Amount Spent levels by Catalogs')


# ## Q-Q Quantitative to Quantitative Salary vs Amount Spent

# In[ ]:


sns.regplot(x='salary', y='amountspent', data=df_drop_na)


# ## C-C Categorical to Categoical split of data based on the percentage

# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('age')['gender'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('age')['ownhome'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('age')['married'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('age')['location'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('age')['children'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('age')['history'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('gender')['ownhome'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('gender')['married'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('gender')['location'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('gender')['children'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('gender')['history'].value_counts()/len(df_drop_na),2))


# In[ ]:


pd.DataFrame(round(100* df_drop_na.groupby('gender')['catalogs'].value_counts()/len(df_drop_na),2))


# In[ ]:


df_drop_na[['salary', 'amountspent', 'age']].groupby('age').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'gender']].groupby('gender').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'ownhome']].groupby('ownhome').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'married']].groupby('married').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'location']].groupby('location').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'children']].groupby('children').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'history']].groupby('history').describe().T


# In[ ]:


df_drop_na[['salary', 'amountspent', 'catalogs']].groupby('catalogs').describe().T


# ## Correlation Matrix

# In[ ]:


pd.get_dummies(df_drop_na).corr().style.background_gradient(cmap='coolwarm')


# # Findings

# As per the limited background of the data, we can conclude the following information strictly look at the data. 
# 
# We have some encode information in the data we need to assume information about the data. Some of the key finds are as follows.
# 
# 1. People that have a high history of shopping from the location, are high spenders and have higher salary bracket. They preform to shop in the catalogue 24. These people are usually married and homeowner and don't rent their property. The shopper from the married couple is usually male and middle age.
# 2. The location of shopper compared to the history of visiting. We see people the live far, have high history. The people living in the far location like to shop in catalogue 18 and 24.
# 3. If the location is looking for high spender, they should focus the campaign on middle, or old age male. Who are married and live far from the location, with the product belonging to catalogue 18, and 24.
# 4. If the location is looking for a high net worth (salary) individual. They location shop focus on middle-aged married male and house owner. With the product belonging to catalogue 24.
# 5. If the location is targeting young buyer, they a better chance of applying to them with the product belonging to catalogue 6.
# 

# In[ ]:




