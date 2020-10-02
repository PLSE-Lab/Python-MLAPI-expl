#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input/jamalon-arabic-books-dataset"))

# Any results you write to the current directory are saved as output.


# # IMPORT NECESSARY LIBRARIES TO READING THE DATA

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


jamalon = pd.read_csv('../input/jamalon-arabic-books-dataset/jamalon dataset.csv')
jamalon.head()


# In[ ]:


df1 = jamalon = pd.read_csv('../input/jamalon-arabic-books-dataset/jamalon dataset.csv')
df1.dataframeName = 'jamalon1.csv'


# In[ ]:


df1.shape


# In[ ]:


jamalon.shape


# # Replace 0 with median in Publication Year

# In[ ]:


jamalon['Publication year'] = jamalon['Publication year'].replace(0,jamalon['Publication year'].median())


# In[ ]:


jamalon.head()


# # Check if any null values in the data

# In[ ]:


jamalon.isnull().sum()


# # Data Cleaning

# In[ ]:


# Delete Unwanted Columns Description , Unnamed : 0
jamalon1=jamalon.drop(columns = {'Unnamed: 0' , 'Description'})


# In[ ]:


jamalon1


# # Exploratory Data Analysis (EDA)

# In[ ]:


jamalon1.describe().transpose()


# In[ ]:


# Calculate column count on Categorical Data
# Calculate column count - Title
jamalon1['Title'].value_counts()


# In[ ]:


# Calculate column count - Author
jamalon1['Author'].value_counts()


# In[ ]:


# Calculate column count - Publisher
jamalon1['Publisher'].value_counts()


# In[ ]:


# Calculate column count - Publication year
jamalon1['Publication year'].value_counts()


# In[ ]:


# Calculate column count - Cover
jamalon1['Cover'].value_counts()


# In[ ]:


# Calculate column count - Category
jamalon1['Category'].value_counts()


# In[ ]:


# Calculate column count - Subcategory
jamalon1['Subcategory'].value_counts()


# # Skewness and Kurtosis

# In[ ]:


# Import Libraries
from scipy.stats import skew , kurtosis


# In[ ]:


# Skewness & Kurtosis for Pages
print("Skewness for Pages" , skew(jamalon1['Pages']))
print("Kurtosis for Pages" , kurtosis(jamalon1['Pages']))


# In[ ]:


# Skewness & Kurtosis for Price
print("Skewness for Price" , skew(jamalon1['Price']))
print("Kurtosis for Price" , kurtosis(jamalon1['Price']))


# # Graphical Visualtions

# # Graphical Visualtions

# # 1. Publication Year

# In[ ]:


# Histogram
jamalon1['Publication year'].hist()


# # 2. Price

# In[ ]:


# Histogram
jamalon1['Price'].hist()


# # 3 . Pages

# In[ ]:


# Histogram
jamalon1['Pages'].hist()


# # Distribution plot

# In[ ]:


sns.distplot(jamalon1['Price'])


# In[ ]:


sns.distplot(jamalon1['Pages'])


# In[ ]:


sns.distplot(jamalon1['Publication year'])


# # Correlation

# In[ ]:


plt.figure(figsize=(14,10))
sns.heatmap(jamalon.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()


# # Pair Plot

# In[ ]:


sns.pairplot(data=jamalon)


# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


plotPerColumnDistribution(df1, 5, 5)


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


plotCorrelationMatrix(df1, 8)


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[ ]:


plotScatterMatrix(df1, 10,5)


# In[ ]:




