#!/usr/bin/env python
# coding: utf-8

# <a id="top"></a>  
# #  Tutorial - Python SubPlots
# **Subplots** are an excellent tools for *data visualization*.  Data can be plotted side-by-side for comparison and multiple plots can show inter-relationships.  Subplots provide considerable control over how data can be displayed.
# 
# This is quick start tutorial on subplots.  A very short and quick one :-)
# 
# ## Basic Python Code for Subplots
# Following are the key Python components of subplots:
# 
#     -----------------------------------------------
#     fig = plt.figure(figsize=(8,8))   #  figure size
#     
#     fig.add_subplot(a,b,c)            #  subplot 1
#     plt.plot(data_1)                  #  data 1
#     
#     fig.add_subplot(a,b,c)            #  subplot 2
#     plt.plot(data_2)                  #  data 2
#     
#     plt.show()                        #  draw plots
#     -----------------------------------------------
#     where:
#         a  - number of rows
#         b  - number of columns
#         c  - plot counter
#         
#         
# 
# ### Three basic rules for subplots:
# 1. **fig.add_subplot(abc)** and **fig.add_subplot(a,b,c)** are the same commands
# 2. number of **rows** and **columns** are *constant* per subplot group
# 3. **plot count** is *incremented* for each subplot
# 
# 
# 
# ##  Examples
# 1.  [Subplots - 1 x 2](#sub_1)   
# 2.  [Subplots - 2 x 1](#sub_2)   
# 3.  [Subplots - 2 x 3](#sub_3)   
# 4.  [Multiple Subplots with FOR Loop](#sub_for)   
# 5.  [Heatmaps with Subplots](#sub_heat) 
# 6.  [Seaborn PairPlot](#sub_pair)   

# ### Import Libraries and Load Data
# Using the Heart Disease UCI dataset:
# https://www.kaggle.com/ronitf/heart-disease-uci

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)

get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


df = pd.read_csv("../input/heart.csv")


# [go to top of document](#top)     
# 
# ---
# <a id="sub_1"></a>
# ## 1.  Subplots - 1 x 2 
# Generate a basic subplot.
# 
# *  number of Rows = 1   
# *  number of Columns = 2

# In[ ]:


fig = plt.figure(figsize=(10,5))

#  subplot #1
fig.add_subplot(121)
plt.title('subplot(121)', fontsize=14)
sns.countplot(data=df, x='cp')

#  subplot #2
fig.add_subplot(122)
plt.title('subplot(122)', fontsize=14)
sns.scatterplot(data=df,x='age',y='chol',hue='sex')

plt.show()


# [go to top of document](#top)     
# 
# ---
# <a id="sub_2"></a>
# ## 2.  Subplots - 2 x 1 
# Generate and dynamically populate **title** of subplot.
# 
# *  number of Rows = 2   
# *  number of Columns = 1

# In[ ]:


row = 2
col = 1
cnt = 1  # initialize

fig = plt.figure(figsize=(4,10))

#  subplot #1
fig.add_subplot(row,col,cnt)
plt.title('subplot #{}:  row = {}, column = {}, plot count = {}'.format(cnt,row,col,cnt), fontsize=14)
sns.countplot(data=df, x='cp')

cnt = cnt + 1  # increment counter

#  subplot #2
fig.add_subplot(row,col,cnt)
plt.title('subplot #{}:  row = {}, column = {}, plot count = {}'.format(cnt,row,col,cnt), fontsize=14)
sns.scatterplot(data=df,x='age',y='chol',hue='sex')

plt.show()


# [go to top of document](#top)     
# 
# ---
# <a id="sub_3"></a>
# ## 3.  Subplots - 2 x 3 
# Generate multiple subplots.
# 
# *  number of Rows = 2   
# *  number of Columns = 3

# In[ ]:


fig = plt.figure(figsize=(14,12))

#  subplot #1
fig.add_subplot(231)
plt.title('subplot(231) - title', fontsize=14)
sns.countplot(data=df, x='cp',hue='sex')

#  subplot #2
fig.add_subplot(2,3,2)
plt.title('subplot(2,3,2)', fontsize=14)
sns.scatterplot(data=df,x='age',y='chol',hue='sex')

#  subplot #3
fig.add_subplot(233)
plt.title('subplot(233)', fontsize=14)
sns.lineplot(data=df, x=df['age'],y=df['oldpeak'])

#  subplot #4
fig.add_subplot(2,3,4)
plt.title('subplot(2,3,4)', fontsize=14)
sns.boxplot(data=df[['chol','trestbps','thalach']])

#  subplot #5
fig.add_subplot(235)
plt.title('subplot(235)', fontsize=14)
sns.distplot(df.chol)

plt.show()


# [go to top of document](#top)     
# 
# ---
# <a id="sub_for"></a>
# ## 4.  Multiple Subplots with FOR Loop 
# Each attribute will be plotted with:  
#   1. **Overall**
#   2. **No-Disease**
#   3. **Disease**
#   
# *  number of Rows = length(data)   
# *  number of Columns = 3

# In[ ]:


#  Plots: Overall, no disease and disease
df2 = df[['sex','cp','slope','ca']] # select a few attributes

#  select "no disease" and "disease" data
df_target_0 = df[(df['target'] == 0)]
df_target_1 = df[(df['target'] == 1)]


#  SUBPLOTS - FOR Loop
rowCnt = len(df2.columns)
colCnt = 3     # cols:  overall, no disease, disease
subCnt = 1     # initialize plot number

fig = plt.figure(figsize=(12,30))

for i in df2.columns:
    # OVERALL subplots
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.title('OVERALL (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df[i], hue=df.sex)
    subCnt = subCnt + 1

    # NO DISEASE subplots
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.title('NO DISEASE (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df_target_0[i], hue=df.sex)
    subCnt = subCnt + 1

    # DISEASE subplots
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.title('DISEASE (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df_target_1[i], hue=df.sex)
    subCnt = subCnt + 1

plt.show()


# [go to top of document](#top)     
# 
# ---
# <a id="sub_heat"></a>
# ##  5.  Heatmaps with Subplots
# Plot correlations in heatmaps for both the sexes.

# In[ ]:


# correlation - female
dfFemale = df2[(df2['sex'] == 1)]
dfFemaleCorr = dfFemale.drop(["sex"], axis=1).corr()
# correlation - male
dfMale   = df2[(df2['sex'] == 0)]
dfMaleCorr = dfMale.drop(["sex"], axis=1).corr()


#  SUBPLOTS
fig = plt.figure(figsize=(12,6))

#  heatmap - female subplot
fig.add_subplot(121)
plt.title('correlation Heart Disease - FEMALE', fontsize=14)
sns.heatmap(dfFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')

#  heatmap - male subplot
fig.add_subplot(122)
plt.title('correlation Heart Disease - MALE', fontsize=14)
sns.heatmap(dfMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')

plt.show()


# [go to top of document](#top)     
# 
# ---
# <a id="sub_pair"></a>
# ##  6.  Seaborn pairplot()
# Seaborn pairplot() creates a bivariate scatter plot for each attribute in a dataframe and automatically creates the subplots.  The diagonal axes are univariate distribution of the data for the variable in that column.
# 
# Pairplots are a great way of quickly displaying the data without setting up subplots.

# In[ ]:


sns.pairplot(data = df[['age','chol','trestbps','thalach']])


# [go to top of document](#top)     
# 
# ---
# *Please upvote if you found this helpful :-)*
