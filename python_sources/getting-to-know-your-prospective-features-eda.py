#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import math
import seaborn as sns
sns.set(style="white")
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

import time
t0 = time.time()


# In[ ]:


os.listdir("../input/")


# In[ ]:


os.listdir("../input/")
loc = "../input/"


# In[ ]:


df_orig = pd.read_csv(loc+"WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df_orig.copy()


# In[ ]:


pd.Series(df_orig.columns)


# In[ ]:


df_orig.head()


# # Data Cleaning

# In[ ]:


print("Customer ID is the primary key") if df.customerID.shape[0]==df.customerID.nunique() else print("Oops")


# In[ ]:


# Changing Total Charges to Numeric Datatype
df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# There are still eleven values that are null
print(df.isna().sum())


# In[ ]:


# Decoding all columns to numeric datatype
lb = LabelEncoder()
for column in df.select_dtypes(object).columns:
    df[column] = lb.fit_transform(df[column])


# # Descriptive Statistics

# In[ ]:


# Histograms
columns_to_view = df[['PaperlessBilling','tenure','TotalCharges','MonthlyCharges']].columns
df[columns_to_view].hist(figsize = (10,10));


# In[ ]:


# Histograms
for column in df_orig.select_dtypes(object).columns.drop(['Churn','customerID']):
    df_orig.groupby(column).count()['customerID'].plot.pie(figsize = (5,5))
    print('Distribution Pie Chart of ',column)
    plt.show()


# In[ ]:


# This plot shows that if there is a partner, then the number of dependents increase drastically
df.groupby('Partner').sum()['Dependents'].plot(kind = 'bar')
plt.ylabel("Number of Dependents")
plt.show()


# In[ ]:


# Those who do not have internet service only have DSL internet connection
df_orig.groupby(['PhoneService','InternetService'])['InternetService'].count().unstack().plot(kind = 'bar')
plt.show()


# # Inferential Statistics

# Finding Correlation Matrix to assist us in choosing the columns

# In[ ]:


# Correlation Matrix
corr = df.corr()

sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


# In[ ]:


# No Phone Service has a very low churn
df_orig.groupby('MultipleLines')['Churn'].count().plot(kind = 'bar')


# In[ ]:


# Citizens paying through Electronic Check have a good churn
df_orig.groupby('PaymentMethod')['Churn'].count().plot(kind = 'bar')


# In[ ]:


# People who pay more tend to stay with the company
df_orig.groupby('Churn')['MonthlyCharges'].mean().plot(kind = 'bar')
# Senior Citizens have a very low churn


# In[ ]:


print("time since start = ",time.time()-t0)


# In[ ]:


# ANOVA test to analyze diversity of feature
from scipy import stats


# In[ ]:


df_orig.select_dtypes(object).columns.drop('customerID')


# In[ ]:


# Calculating the impact of each variable on Churn by ANOVA tests

results = list()
for col_name in df_orig.select_dtypes(object).columns.drop(['customerID','TotalCharges','Churn']):
    all_grps = list()
    for grp in df.groupby(col_name)['Churn']:
        all_grps.append(grp[1].tolist())
    f_val, p_val = stats.f_oneway(*all_grps)
    results.append([col_name,f_val,p_val])
results = pd.DataFrame(results,columns = ['col_name','f_val','p_val'])
print(results)


# In[ ]:


# Plotting Results of P Values
results[['col_name','p_val']].sort_values('p_val', ascending = False).plot(kind = 'bar', x = 'col_name')
plt.title("P Value of Grouped Variables")
plt.legend()
plt.show()

# Plotting Results of F Values
results[['col_name','f_val']].sort_values('f_val', ascending = False).plot(kind = 'bar', x = 'col_name')
plt.title("F Value of Grouped Variables")
plt.legend()
plt.show()
# I think that the gender has little impact on Churn alone, we need to dive deeper into gender and drop itself after creating new features
# I think that the PhoneService has little impact on Churn alone, we need to dive deeper into gender and drop itself after creating new features
# From the greater F values,we can conclude that Contract has highest correlation with the churn


# In[ ]:


# Defining KPIs
df['family'] = df.Partner | df.Dependents


# In[ ]:


results


# In[ ]:


# Analyzing the F value of recently added KPI
all_grps = list()
for grp in df.groupby('family')['Churn']:
    all_grps.append(grp[1].tolist())
f_val, p_val = stats.f_oneway(*all_grps)
print('f_val=',f_val,' p_val=',p_val)

# F value is not greater than individual F value
# I am not sure how to interpret the usefulness of this stat

