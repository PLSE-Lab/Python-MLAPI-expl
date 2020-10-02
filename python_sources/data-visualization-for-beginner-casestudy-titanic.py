#!/usr/bin/env python
# coding: utf-8

# # The objective of this Notebook is to Introduce Data Visualization through Seaborn.
# * Seaborn is a Statistical Plotting library.
# * It builds on top of matplotlib
# 
# * It Works very well with Pandas dataframes objects
# * Official documentaion website: Seaborn: Statistical Data Visualization (https://seaborn.pydata.org/)
# * You can use the "gallery" in this website as reference
# * You can also use "API" in this website as reference to varous plot type and how to call each plot
# 
# 
# **In this notebook we will learn:**
# * Distribution Plots
# * Categorical Plots
# * Matrix Plots
# 
# 
# 

# # Step1:
# * Call any Library That you Want

# In[ ]:


#Same as any Python code, at first you should call any library that you want:

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Step2:
# * Read the Data 
# * In this notebook I use Titanic Dataset 

# In[ ]:



# read the data:
df =pd.read_csv("../input/python-seaborn-datas/titanic.csv")


# In[ ]:


df.head()


# # Step 3: Data Visualization
# # 1- Histogram

# * To show how Age is distrbuted:
# * in below figure, actually we have a Histogram with KDE (kernel density estimation)
# 

# In[ ]:


sns.distplot(df['Age'])


# * if we don't need KDE, so:

# In[ ]:


sns.distplot(df['Age'], kde=False)


# In[ ]:


* the above histogram shows that:
    Age of most of passengers are between 20 to 40


* If we need more information we can change the number of bins:


# In[ ]:


sns.distplot(df['Age'], kde=False, bins=40)


# # 2-Scatter Plot:
# * use to compare two variable:

# In[ ]:


sns.jointplot(x='Age',y='Fare',data=df)


# # Scatter plot for all numerical data in the dataset:
# #pairplot:
# * paiplot is very useful to quickly visualize your data

# In[ ]:


sns.pairplot(df)


# # 3-Bar Plot:
# 
# * Bar Plot Shows the average of a variable based on a categorical variable
# * below Figure shows that the average (mean) Age of men passengers is slightly higher than women passengers
# 
# 

# In[ ]:


sns.barplot(x='Sex', y='Age', data=df)


# # 4-Count Plot
# * Count plot shows the frequency of a categorical variable

# In[ ]:


sns.countplot(x='Sex',data=df)


# # 5-Box Plot

# In[ ]:


sns.boxplot(x='Sex',y='Age', data=df)


# In[ ]:


sns.boxplot(x='Pclass',y='Age', data=df)


# In[ ]:


sns.boxplot(x='Pclass',y='Age', data=df, hue='Sex')


# # 6-Violin Plot:

# In[ ]:


sns.violinplot(x='Pclass',y='Age', data=df)


# In[ ]:


sns.violinplot(x='Pclass',y='Age', data=df, hue='Sex')


# In[ ]:


sns.violinplot(x='Pclass',y='Age', data=df, hue='Sex', split=True)

