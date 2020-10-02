#!/usr/bin/env python
# coding: utf-8

# # This is a practice EDA (Exploratory Data Analysis) using the White Wine data set

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Import Data

# In[ ]:


df=pd.read_csv('../input/cs178-wine-quality/winequality-white.csv',sep=';')
df.head()


# ## Data Insight

# In[ ]:


df.shape


# In[ ]:


df.columns


# Check if there is null using info

# In[ ]:


df.info()


# Since all has 4898 non-null values, hence there is no null value

# ##Understanding target variable

# In[ ]:


df.quality.unique()


# In[ ]:


df.quality.value_counts()


# # Data Visualization

# ## Check for missing values using sns heatmap. Missing values would show as white bars.

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False, cbar=False)
plt.title('')


# Plot shows that there are no missing values.

# ### Check for the correlation of the variables

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(df.sort_values(by='quality',ascending=True).corr(),cmap='viridis',annot=True)


# From the correlation matrix, we can see that density and residual sugar is strong correlated.
# Citric acid and Free Sulfur Dioxide has almost no impact to quality. We can drop these 2 features if we are using Linear Regression.

# ### Check for outliers

# In[ ]:


cols=df.columns.values
l=len(cols)
plt.figure(figsize=(24,6))
print(cols)
for i in range(0,l):
    plt.subplot(1,l,i+1)
    sns.boxplot(df[cols[i]],orient='v',color='green')
    plt.tight_layout()


# All except alcohol shows outliers. 
# 
# ### Check data skewedness

# In[ ]:


plt.figure(figsize=(24,6))
for i in range(0,l):
    plt.subplot(1,l,i+1)
    sns.distplot(df[cols[i]])


# In[ ]:





# pH column seems to be the only normally distributed feature.
# All other features are right-skewed.
