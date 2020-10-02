#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


print(os.listdir('../input'))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

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


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: ../input/Consumo_cerveja.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/Consumo_cerveja.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Consumo_cerveja.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# Scatter and density plots:

# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!

# In[ ]:


import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df1.head()


# In[ ]:


df1.isnull().any()


# In[ ]:


df1.info()


# In[ ]:


df1


# In[ ]:


# major rows of data have NaN values, thus dropping them would be a wiser descision


# In[ ]:


df1.dropna(inplace=True)


# In[ ]:


# this reduces our dataset to great amount


# In[ ]:


df1.info()


# In[ ]:


df1['Temperatura Media (C)'] = df1['Temperatura Media (C)'].apply(lambda x : float(x.replace(',','.')))


# In[ ]:


df1['Temperatura Maxima (C)'] = df1['Temperatura Maxima (C)'].apply(lambda x : float(x.replace(',','.')))


# In[ ]:


df1['Temperatura Minima (C)'] = df1['Temperatura Minima (C)'].apply(lambda x : float(x.replace(',','.')))


# In[ ]:


df1['Precipitacao (mm)'] = df1['Precipitacao (mm)'].apply(lambda x : float(x.replace(',','.')))


# In[ ]:


df1.head()


# In[ ]:


df1.describe()


# In[ ]:


sns.pairplot(df1)


# In[ ]:


# the pairplot gives an idea that our label depends somewhat linearly with most of our features 


# In[ ]:


from sklearn.model_selection import  train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df1.drop(['Consumo de cerveja (litros)','Data'],axis=1),df1['Consumo de cerveja (litros)'])


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.r2_score(y_test,predictions)


# In[ ]:


# the score is not that much bad considering the dataset wasn't large enough


# In[ ]:


metrics.mean_squared_error(y_test,predictions)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


p = sns.distplot(y_test,kde=False,bins=50)


# In[ ]:


p1 = sns.distplot(predictions,kde=False,bins=50)


# In[ ]:


p2 = sns.distplot(y_test-predictions,kde=False,bins=50)


# In[ ]:


lm.coef_


# In[ ]:


lm.intercept_


# In[ ]:


# attempt to try another regressor


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rfr = RandomForestRegressor(n_estimators=50)


# In[ ]:


rfr.fit(X_train,y_train)
pred = rfr.predict(X_test)


# In[ ]:


rfr.score(X_test,y_test)


# In[ ]:


# Randomforestregressor didn't performed well as compared to linear regression which makes sense because our label depends linearly with many of 
# our features


# In[ ]:




