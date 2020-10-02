#!/usr/bin/env python
# coding: utf-8

# # Using pandas for Better (and Worse) Data Science
# 
# ![1200px-Pandas_logo.svg.png](attachment:1200px-Pandas_logo.svg.png)
# 
# Good data analysis project is all about asking questions, in this notebook we are goinig to answer the following questions:
#     1. Do men or women speed more often?
#     2. Does gender affect who gets searched during a stop?
#     3. During a search, how often is the driver frisked?
#     4. Which year had the least number of stops?
#     5. How does drug activity change by time of day?
#     6. Do most stops occur at night?
# 
# ***
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**
# ***
# 
# ### Instructor: Kevin Markham
# This is a kernel that is perceiving the steps from `PyCon 2019` of `Kevin Markham`.
# 
# - GitHub: https://github.com/justmarkham
# - Twitter: https://twitter.com/justmarkham
# - YouTube: https://www.youtube.com/dataschool
# - Website: http://www.dataschool.io

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# # Dataset: Stanford Open Policing Project  
# 
# ![dataset-thumbnail.jpg](attachment:dataset-thumbnail.jpg)
# 
# [Stanford Open Policing Project ](https://openpolicing.stanford.edu/)

# In[ ]:


df = pd.read_csv("/kaggle/input/police_project.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# - **What does NaN mean?**
# 
# > In computing, NaN, standing for not a number, is a member of a numeric data type that can be interpreted as a value that is undefined or unrepresentable, especially in floating-point arithmetic. 
# 
# 
# - **Why might a value be missing?**
# 
# > There are many causes of missing values, Missing data can occur because of `nonresponse`, `Attrition`, `governments or private entities`, ...
# 
# - **Why mark it as NaN? Why not mark it as a 0 or an empty string or a string saying "Unknown"?**
# 
# > We mark missing values as `NaN` to make them distinguish from the original dtype of the feature.

# `county_name`  All the data is missing, We will `drop` this column.

# # 1. Remove the column that only contains missing values

# In[ ]:


df.dropna(axis=1, how='all').shape


# In[ ]:


df.drop('county_name', axis=1, inplace=True)


# In[ ]:


df.isnull().sum()


# **Lessons:**
# 
# - Pay attention to default arguments
# - Check your work
# - There is more than one way to do everything in pandas

# # 2. Do men or women speed more often?

# In[ ]:


sns.catplot('driver_gender', data=df, kind="count", height=7)


# In[ ]:


df.driver_gender.value_counts()


# Responding to this question, we must take consideration of the non-equivalent distribution of the data or use fraction.

# In[ ]:


print(df[df.violation == 'Speeding'].driver_gender.value_counts(normalize=True))
plt.figure(figsize=(12, 8))
df[df.violation == 'Speeding'].driver_gender.value_counts().plot(kind="bar")


# In[ ]:


df.loc[df.violation == "Speeding", "driver_gender"].value_counts(normalize=True)


# ## 2. 1. When a man is pulled over, How often is it for speeding?

# In[ ]:


df[df.driver_gender == "M"].violation.value_counts(normalize=True)


# ## 2. 2. When a women is pulled over, How often is it for speeding?

# In[ ]:


df[df.driver_gender == "F"].violation.value_counts(normalize=True)


# In[ ]:


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df[df.driver_gender == "F"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Women")

plt.subplot(2, 2, 2)
df[df.driver_gender == "M"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Men")


# In[ ]:


sns.catplot('violation', data=df, hue='driver_gender', kind='count', height=8)


# # 3. Does gender affect who gets searched during a stop?

# In[ ]:


df.search_conducted.value_counts()


# From all `88545` stoping cases the data only `3196` are searched.

# In[ ]:


df.loc[df.search_conducted, 'driver_gender'].value_counts()


# From the stopped cases `2725` are `men` and only `471` are women.

# In[ ]:


df.groupby(['violation', 'driver_gender']).search_conducted.mean()


# In[ ]:


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.search_conducted.value_counts().plot(kind="bar")
plt.title("Searched Cases")

plt.subplot(2, 2, 2)
df.loc[df.search_conducted, 'driver_gender'].value_counts().plot(kind="bar")
plt.title("Searched Men and Women")

plt.subplot(2, 2, 3)
df.groupby(['violation', 'driver_gender']).search_conducted.mean().plot(kind="bar")


# Does this prove causation?
# 
# **Lessons:**
# 
# - Causation is difficult to conclude, so focus on relationships
# - Include all relevant factors when studying a relationship

# # 4. Why is search_type missing so often?

# In[ ]:


df.search_type.isnull().sum()


# In[ ]:


df.search_conducted.value_counts()


# In[ ]:


df[df.search_conducted == False].search_type.value_counts(dropna=False)


# `search_type` is missing every time the police don't conduct a search.

# In[ ]:


df.search_type.value_counts()


# In[ ]:


plt.figure(figsize=(12, 8))
df.search_type.value_counts().plot(kind="bar")


# **Lessons:**
# 
# - Verify your assumptions about your data
# - pandas functions ignore missing values by default

# # 5. During a search, how often is the driver frisked?

# In[ ]:


df.search_type.value_counts()


# In[ ]:


counter = 0
for item in df.search_type:
    if type(item) == str and "Protective Frisk" in item:
        counter += 1
print(counter)


# In[ ]:


df.search_type.str.contains('Protective Frisk').sum()


# In[ ]:


df.search_type.str.contains('Protective Frisk').mean()


# `8.57%` of the time the driver is frisked.

# **Lessons:**
# 
# - Use string methods to find partial matches
# - Use the correct denominator when calculating rates
# - pandas calculations ignore missing values
# - Apply the "smell test" to your results

# # 6. Which year had the least number of stops?

# In[ ]:


df.head()


# In[ ]:


print(df.stop_date.dtype)
print(df.stop_time.dtype)


# In[ ]:


df.stop_date


# In[ ]:


df['stop_date'] = pd.to_datetime(df.stop_date, format="%Y-%M-%d")
df["year"] = df.stop_date.dt.year


# In[ ]:


df.dtypes


# In[ ]:


df.year.value_counts()


# In[ ]:


plt.figure(figsize=(12, 8))
df.year.value_counts().plot(kind="bar")


# **Lessons:**
# 
# - Consider removing chunks of data that may be biased
# - Use the datetime data type for dates and times

# # 7. How does drug activity change by time of day?

# In[ ]:


df.columns


# In[ ]:


df.drugs_related_stop.value_counts()


# In[ ]:


df["stop_time"] = pd.to_datetime(df.stop_time, format="%H:%M").dt.hour
df.head()


# In[ ]:


df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts()


# In[ ]:


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts().sort_index().plot(kind="bar")

plt.subplot(2, 2, 2)
df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts().sort_index().plot()


# **Lessons:**
# 
# - Use plots to help you understand trends
# - Create exploratory plots using pandas one-liners

# # 8. Do most stops occur at night?

# In[ ]:


df.stop_time.sort_index().value_counts().sort_index()


# In[ ]:


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.stop_time.sort_index().value_counts().sort_index().plot()

plt.subplot(2, 2, 2)
df.stop_time.sort_index().value_counts().sort_index().plot(kind="bar")


# **Lessons:**
# 
# - Be conscious of sorting when plotting

# # 9. Find the bad data in the stop_duration column and fix it 

# In[ ]:


df.stop_duration.isnull().sum()


# In[ ]:


df.stop_duration.unique()


# In[ ]:


df.stop_duration.value_counts(dropna=False)


# In[ ]:


# ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)
df.loc[(df.stop_duration == '1')| (df.stop_duration == '2'), 'stop_duration'] = np.nan


# In[ ]:


df.stop_duration.value_counts(dropna=False)


# **Lessons:**
# 
# - Ambiguous data should be marked as missing
# - NaN is not a string

# # 10. What is the mean stop_duration for each violation_raw?

# In[ ]:


df.stop_duration.unique()


# In[ ]:


df.violation_raw.value_counts()


# In[ ]:


df.groupby('stop_duration').violation_raw.value_counts()


# In[ ]:


sns.catplot("stop_duration", data=df, hue="violation_raw", kind="count", height=7)


# In[ ]:


plt.figure(figsize=(12, 12))
df.groupby('stop_duration').violation_raw.value_counts().plot(kind="bar")


# In[ ]:


mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
df['stop_minutes'] = df.stop_duration.map(mapping)


# In[ ]:


df.stop_minutes.value_counts()


# In[ ]:


df.groupby('violation_raw').stop_minutes.mean()


# In[ ]:


df.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


# In[ ]:


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
df.groupby('violation_raw').stop_minutes.mean().plot(rot=45)

plt.subplot(2, 2, 2)
df.groupby('violation_raw').stop_minutes.mean().plot(kind="bar")


# **Lessons:**
# 
# - Convert strings to numbers for analysis
# - Approximate when necessary
# - Use count with mean to looking for meaningless means

# # 11. Compare the age distributions for each violation

# In[ ]:


df.groupby("violation").driver_age.describe()


# In[ ]:


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.driver_age.hist(bins=10)

plt.subplot(2, 2, 2)
df.driver_age.value_counts().sort_index().plot()


# In[ ]:


df.hist('driver_age', by='violation', figsize=(12, 12));


# **Lessons:**
# 
# - Use histograms to show distributions
# - Be conscious of axes when using grouped plots
