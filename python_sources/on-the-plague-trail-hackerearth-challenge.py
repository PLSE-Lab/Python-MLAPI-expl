#!/usr/bin/env python
# coding: utf-8

# # On The Plague Trail (Machine Learning Hackathon) - HackerEarth

# **PROBLEM STATEMENT**
# 
# Predict the total number of people infected by the 7 different pathogens.
# 
# Plague is an epidemic event caused by Bacteria. A group of senior scientists misplaced a package containing fatal plague bacteria during one of their trips. With no means of tracking where the package is, scientists are now trying to come up with a solution to stop the plague. This plague has 7 different strains that are unique for each continent. This strain is expanding rapidly in each continent.
# 
# The dataset contains escalations of the plague for all the seven strains. The dataset is a time series in which the training set contains the number of individuals that are infected by the plague over a defined period of time.
# 
# Your mission, should you choose to accept it, is to defend the world against this plague by building an algorithm that can minimize the damage.

# **COLUMNS DESCRIPTION**
# 
# You have to predict the columns of PA, PB, PC, PD, PE, PF, PG 
# 
# ID:
# A calculated unique ID for each research.
# 
# DateTime:
# Represents the data and time on which the event is recorded
# 
# TempOut:
# Outside Temperature
# 
# HiTemp:
# Highest Temperature
# 
# LowTemp:
# Lowest Temperature
# 
# OutHum:
# Outside Humidity
# 
# DewPt:
# Dew Point
# 
# WindSpeed:
# Wind Speed
# 
# WindDir:
# Wind Direction
# 
# WindRun:
# Wind Run Flow
# 
# HiSpeed:
# Highest Speed of the wind
# 
# HiDir:
# Direction of the wind which has highest speed
# 
# WindChill:
# Chillness of the wind
# 
# HeatIndex:
# Heat Index
# 
# THWIndex:
# THW Index
# 
# Bar:
# Barometer Reading
# 
# Rain:
# Rain
# 
# RainRate:
# Frequency of Rain
# 
# HeatDD:
# Heat DD
# 
# CoolDD:
# Cool DD
# 
# InTemp:
# Temperature Inside
# 
# InHum:
# Humidity Inside
# 
# InDew:
# Dew Inside
# 
# InHeat:
# Heat Inside
# 
# InEMC:
# EMC Inside
# 
# InAirDensity:
# Air Density
# 
# WindSamp:
# Wind - Attribute 1
# 
# WindTx:
# Wind - Attribute 2
# 
# ISSRecpt:
# Reception
# 
# ArcInt:
# Attribute
# 
# PA:
# Total No of People infected by Pathogen A
# 
# PB:
# Total No of People infected by Pathogen B
# 
# PC:
# Total No of People infected by Pathogen C
# 
# PD:
# Total No of People infected by Pathogen D
# 
# PE:
# Total No of People infected by Pathogen E
# 
# PF:
# Total No of People infected by Pathogen F
# 
# PG:
# Total No of People infected by Pathogen G
# 

# **Evaluation Criteria**
# 
# Score = max(0,(100-RMSE))

# In[ ]:


import os # accessing directory structure
print(os.listdir('../input'))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import seaborn as sns
import missingno as msno
from tqdm import tqdm_notebook


# **Exploratory Data Analysis**

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


df = pd.read_csv('../input/train.csv',encoding='latin1')
df.shape


# Divide train.csv into train set and validation set

# In[ ]:


df_trainset = df[0:int(0.7*df.shape[0])]
df_validationset = df[int(0.7*df.shape[0]):]


# In[ ]:


print(df_trainset.shape)
print(df_validationset.shape)


# In[ ]:


df_trainset.dataframeName = 'training dataset'
nRow, nCol = df_trainset.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df_trainset.head()


# In[ ]:


# df_trainset.info() shows the number of non-null values (and hence the number of missing values).
df_trainset.info()


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df_trainset, 10, 2)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df_trainset, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df_trainset, 20, 10)


# In[ ]:


# missingno.matrix function shows missing/non-missing values in two colors.
msno.matrix(df_trainset)


# In[ ]:


# missingno.heatmap function gives nullity correlation
# (how strongly the presence or absence of one variable affects the presence of another).
# For the train dataset given, there is no strong correlation observed for nullity.
try:
    msno.heatmap(df_trainset)
except ValueError:
    pass


# In[ ]:


# Range of Date
#df_trainset['DateTime']


# In[ ]:


# Columns of Non-numeric Type
columns_total = np.array(list(df_trainset))
columns_numeric = np.array(list(df_trainset.select_dtypes(include=np.number)))
columns_non_numeric = np.setdiff1d(columns_total,columns_numeric)
print('---All Columns---')
print(columns_total)
print('---Columns of Numeric Type---')
print(columns_numeric)
print('---Columns of Non-numeric Type---')
print(columns_non_numeric)


# In[ ]:


# Data Visualizations


# **Making ML Model**

# Splitting the train set and validation set into features set and targets set

# In[ ]:


X_train = df_trainset[df_trainset.columns[0:30]]
y_train = df_trainset[df_trainset.columns[30:37]]
X_validate = df_validationset[df_validationset.columns[0:30]]
y_validate = df_validationset[df_validationset.columns[30:37]]


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


X_validate.head()


# In[ ]:


y_validate.head()


# Linear Regression Model

# In[ ]:


def function_preprocess_X(X):
    X_preprocessed = X.copy()
    X_preprocessed = X_preprocessed.select_dtypes(include=np.number)
    #X_train_numeric.head()
    return X_preprocessed


# In[ ]:


model_LinearRegression = sklearn.linear_model.LinearRegression()
X_train_preprocessed = function_preprocess_X(X_train)
model_LinearRegression.fit(X_train_preprocessed,y_train)


# In[ ]:


y_train_prediction = model_LinearRegression.predict(X_train_preprocessed)

rmse = np.sqrt(mean_squared_error(y_train,y_train_prediction))
print(f'RMSE: {rmse}')


# **Making Predictions on Validation Dataset**

# In[ ]:


X_validate_preprocessed = function_preprocess_X(X_validate)
y_validate_prediction = model_LinearRegression.predict(X_validate_preprocessed)

rmse = np.sqrt(mean_squared_error(y_validate,y_validate_prediction))
print(f'RMSE: {rmse}')


# **Making Predictions on Test Dataset**

# In[ ]:


df_sample = pd.read_csv('../input/sample.csv',encoding='latin1')
print(df_sample.shape)
df_sample.head()


# In[ ]:


X_test = pd.read_csv('../input/test.csv',encoding='latin1')
print(X_test.shape)
X_test.head()


# In[ ]:


submission = pd.DataFrame(columns=df_sample.columns)
submission['ID'] = X_test['ID'].values
X_test_preprocessed = function_preprocess_X(X_test)


# **Final Submission**

# In[ ]:


submission[y_train.columns] = model_LinearRegression.predict(X_test_preprocessed)
submission.to_csv('submission.csv',index=False)

