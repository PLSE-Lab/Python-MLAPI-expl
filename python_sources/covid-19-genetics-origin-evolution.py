#!/usr/bin/env python
# coding: utf-8

# Goals:
# * Investigate SARS-CoV-2 virus genetics, origin,  and evolution.
# * Investigate SARS-CoV-2 management measures at the human-animal interface.

# In[ ]:


# Python 3 environment
# GPU processor
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import shutil


# * Read in csv's nucleotide, protein, and alignment hit (compares nucleotide of virus for identical matches within sequence, percentage of similiar matches, base pairs and its length) 
# * These 3 csv's compare the following taxonomy of viruses:  
# SARS-CoV-2 (genbank)  
# SARS2  
# 2019-nCov  
# COVID-19  
# COVID-19 virus  
# Wuhan corona virus  
# Wuhan seafood market pnuemonia virus

# Define function for distribution graphs (histogram/bar graph) for column data:

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


# Define function for correlation matrix:

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


# Define function for scatter and density plots

# In[ ]:


def plotScatterMatrix(df,plotSize,textSize):
    df=df.select_dtypes(include=[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df=df.dropna('columns')
    df=df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames=list(df)
    if len(columnNames)>10: # reduce number of columns for matrix inversion of kernel density plots
        columnNames=columnNames[:10]
    df=df[columnNames]
    ax=pd.plotting.scatter_matrix(df,figsize=[plotSize,plotSize],diagonal='kde')
    corrs=df.corr().values
    for i,j in zip(*plt.np.triu_indices_from(ax,k=1)):
        ax[i,j].annotate('Corr.coef=%.3f' % corrs[i,j],(0.8,0.2),xycoords='axes fraction',ha='center',va='center',size=textSize)
        plt.suptitle('Scatter and Density Plot')
        plt.show()


# * Read in data and use plotting functions to visualize data 
# * Alignment-HitTable compares nucleotide of virus for identical matches within sequence, percentage of similiar matches, base pairs and its length
# 

# In[ ]:


nRowsRead=1000
df1=pd.read_csv('/kaggle/input/sars-coronavirus-accession/MN997409.1-4NY0T82X016-Alignment-HitTable.csv',delimiter=',',nrows=nRowsRead)
df1.dataframeName='MN997409.1-4NY0T82X016-Alignment-HitTable.csv'
nRow,nCol=df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Get quick view of data:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of quick view data:

# In[ ]:


plotPerColumnDistribution(df1,10,5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1,8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 20, 10)


# Read in SARS_CORONAVIRUS_287BP_MN975263.1_accession_nucleotide:

# In[ ]:


nRowsRead=1000
df2=pd.read_csv('/kaggle/input/sars-coronavirus-accession/SARS_CORONAVIRUS_287BP_MN975263.1_accession_nucleotide.csv',delimiter=',',nrows=nRowsRead)
df2.dataframeName='SARS_CORONAVIRUS_287BP_MN975263.1_accession_nucleotide.csv'
nRow,nCol=df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df2.head(5)


# In[ ]:


plotPerColumnDistribution(df2, 10, 5)


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# SARS_CORONAVIRUS_287BP_MN975263.1_protein.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/sars-coronavirus-accession/SARS_CORONAVIRUS_287BP_MN975263.1_accession_protein.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'SARS_CORONAVIRUS_287BP_MN975263.1_accession_protein.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df3.head()


# In[ ]:


plotPerColumnDistribution(df3, 10, 5)

