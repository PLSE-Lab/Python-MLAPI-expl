#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, we explore the user and flights data from a [travel dataset](https://www.kaggle.com/leomauro/argodatathon2019).

# ## Load data
# 
# - Imports
# - Load pandas `DataFrame`
# - Combing `DataFrame`

# In[ ]:


# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting
import seaborn as sns # more plotting

plt.style.use('seaborn-colorblind') # plotting style


# In[ ]:


import os

# Listing all datasets
pathFiles = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pathFile = os.path.join(dirname, filename)
        pathFiles.append(pathFile)
        print(pathFile)


# In[ ]:


# Importing dataset
dfUser = pd.read_csv('/kaggle/input/argodatathon2019/users.csv', delimiter=',')
dfUser.dataframeName = 'users.csv'

dfFlights = pd.read_csv('/kaggle/input/argodatathon2019/flights.csv', delimiter=',')
dfFlights.dataframeName = 'flights.csv'

for df in [dfUser, dfFlights]:
    nRow, nCol = df.shape
    print(f'{df.dataframeName}, there are {nRow} rows and {nCol} columns')


# In[ ]:


dfUser.head(3)


# In[ ]:


dfFlights.head(3)


# In[ ]:


# Merging DataFrame
dfUser.columns = ['userCode', 'company', 'name', 'gender', 'age']
dfMerge = pd.merge(dfUser, dfFlights, on='userCode')
dfMerge.dataframeName = 'flights by user'
dfMerge.head(3)


# ## Correlation
# 
# - Checking all correlations

# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        return
    corr = df.corr()
    plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for "{filename}"', fontsize=10)
    plt.show()


# In[ ]:


plotCorrelationMatrix(dfMerge)


# ## Number of occorrences
# 
# - Histogram of "from place"
# - Histogram of "flight type"

# In[ ]:


# Histogram plot
def plotHistogram(df, column):
    D = df[column].value_counts()
    fig, ax = plt.subplots()
    ax.bar(range(len(D)), D.values, align='center')
    plt.xticks(range(len(D)), D.index, rotation=15)
    ax.set_title(f'Histogram "{column}"')
    ax.set_ylabel('Histogram')
    ax.set_xlabel(f'{column}')
    plt.show()


# In[ ]:


dfMerge.columns


# In[ ]:


plotHistogram(dfMerge, 'from')


# In[ ]:


plotHistogram(dfMerge, 'flightType')


# ## Scatter plot
# 
# - Variables 'price', 'time', 'distance'

# In[ ]:


# Scatter plot
def plotScatter(df, columnA, columnB):
    fig, ax = plt.subplots()
    ax.scatter(df[columnA], df[columnB])
    ax.set_title(f'Scatter "{columnA}" x "{columnB}"')
    ax.set_ylabel(f'{columnB}')
    ax.set_xlabel(f'{columnA}')
    plt.show()


# In[ ]:


plotScatter(dfMerge, 'price', 'time')


# In[ ]:


plotScatter(dfMerge, 'price', 'distance')


# In[ ]:


plotScatter(dfMerge, 'time', 'distance')

