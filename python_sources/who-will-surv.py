#!/usr/bin/env python
# coding: utf-8

# ## Problem understanding

# Breast cancer is the most common invasive cancer in women, and the second main cause of cancer death in women, after lung cancer.
# 
# Advances in screening and treatment have improved survival rates dramatically since 1989. There are around 3.1 million breast cancer survivors in the United States (U.S.). The chance of any woman dying from breast cancer is around 1 in 37, or 2.7 percent.
# 
# Analysis the datset can provide insights in identifying and mitigating the risks. 
# 
# Based upon the followinig three factors we have to **classify** whether the patient survived/will survive 5 years and longer or not.
# 
# Features:
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 
# Target variable:
# 1. Survival status (class attribute)
#     - Value = 1: the patient survived 5 years or longer
#     - Value = 2: the patient died within 5 year
#     

# Reference: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 

# **Note**: The kernel is still in progress and your feedback can help to improve it.

# ![image](https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2F3.bp.blogspot.com%2F-3yThenM3P_Y%2FT50A0xVgBeI%2FAAAAAAAAAi8%2F57FTjFJbg3s%2Fs1600%2Fbreastcancer.jpg&f=1)

# ## Import necessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/haberman.csv')
df.columns = ['age', 'op_year', 'axil_nodes', 'surv_status']
df.head()


# In[ ]:


# Since this is a binary classification problem
# replace surv_status value = 2 to 0 for the sake of readability
df['surv_status'].replace(2, 0, inplace=True)


# ## High level Analysis

# In[ ]:


# number of points
print('Shape of data', df.shape, '\nnumber of dimensions', df.ndim)


# In[ ]:


# datapoints per class - is it imbalanced?
sns.countplot(x='surv_status', data=df)
plt.show()


# In[ ]:


survived_count = sum(df['surv_status'] == 1)
died_count = sum(df['surv_status'] == 0)

ratio = survived_count/died_count             if survived_count > died_count                 else died_count/survived_count


# In[ ]:


round(ratio, 2)


# Data points of survivors > 2x Number of demised patients.
# 
# Having an imbalanced dataset can impact the model performance.

# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


# features
X = df.iloc[:, :3]

# classes
y = df.iloc[:, 3]


# In[ ]:


X.shape


# In[ ]:


y.shape


# ## Which individual features are useful for classification? 

# Univariate analysis

# ### Probability Density Function

# In[ ]:


# histogram of age
ax = sns.FacetGrid(df, hue='surv_status', height=8)                 .map(sns.distplot, 'age')                 .add_legend()


# It can be seen that survival status is tightly overlapped with one another, hence we can conclude that `age` wont contribute much in identification of classes correctly.

# In[ ]:


# histogram of operation year
ax = sns.FacetGrid(df, hue='surv_status', height=8)     .map(sns.distplot, 'op_year')     .add_legend()
plt.show()


# Similar to age, operation year cannot be leveraged to classify the survival status.

# In[ ]:


# histogram of axil nodes
ax = sns.FacetGrid(df, hue='surv_status', height=8)     .map(sns.distplot, 'axil_nodes')     .add_legend()


# It can be observed that if the count of positive axiliary nodes < `3` then the person is more likely to survive the disease.

# ### Box plot

# In[ ]:


def box_plot(y_axis, df):
#     sns.boxplot(x='surv_status', y=y_axis, data=df)
    sns.catplot(x='surv_status', y=y_axis, data=df, 
                kind='box', height=8, aspect=.7)
    plt.show()


# In[ ]:


box_plot('age', df)


# For the feature `age`, if 50% of the population lie between 
# - 46 to 61 then survival status is dead
# - 44 to 60 then survival status is alive
# 
# This does not help in clearly separating categories.

# In[ ]:


box_plot('op_year', df)


# Operation year follows the below rule, if the year is
# - Between 1959 to 1965 then the patient might be dead.
# - Between 1960 to 1966 then the patient has chance of survival.

# In[ ]:


box_plot('axil_nodes', df)


# ### Violin plot

# A violin plot combines the benefits of the previous ones (pdf and bar chart) and projects a combined graph.

# In[ ]:


def violin_plot(feature, df):
    sns.catplot(x='surv_status', y=feature, data=df, kind='violin', height=8)
    plt.show()


# In[ ]:


violin_plot('age', df)


# In[ ]:


violin_plot('op_year', df)


# In[ ]:


violin_plot('axil_nodes', df)


# ## Do combination of features help in classification ?
# Bivariate analysis

# By comparing two features we can observe how they are inter dependent and be useful for classification.

# ### Pair plot

# In[ ]:


plt.close();
sns.set_style('whitegrid');
sns.pairplot(df, hue='surv_status', height=4);
plt.show()


# The following observations were made
# - None of the two features combined could give a perfect classifier
# - Axil nodes in conjunction with other features was able to coparatively classify well.

# ### Heatmap

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(X.corr(), linewidths=.5)


# It can be seen that `age` is having low correlation with `op_year`. 
# 
# Whereas, `axil_nodes` shows almost nil correlation with other features.

# ## Conclusion

# Finally, we can conclude that axil_node is the most important to determine if the patient Will Survive Breast Cancer.
