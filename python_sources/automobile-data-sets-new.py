#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

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

# ### Let's check 1st file: ../input/Automobile.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/Automobile.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Automobile.csv'
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

# In[ ]:


plotScatterMatrix(df1, 20, 10)


# In[ ]:


#Define category columns
cat_cols = ['symboling', 'fuel-type', 'aspiration', 'num-of-doors', 
                'body-style', 'drive-wheels', 'engine-location', 'fuel-system',
                'engine-type', 'num-of-cylinders']


# In[ ]:


#Converting type of categorical columns to category
for col in cat_cols:
    df1[col] = df1[col].astype('category')


# In[ ]:


#Converting  numeric to categorical variables
dummied = pd.get_dummies(df1[cat_cols], drop_first = True)


# In[ ]:


plotCorrelationMatrix(dummied, 8)


# In[ ]:


#adding price column with categorical varaibles
df2=pd.concat([df1['price'],dummied],axis =1)


# In[ ]:


# Plotting graph between price and categorical variables
plotCorrelationMatrix(df2, 8)


# In[ ]:


# Coorelation amnong price and categorical variables
df2.corr(method='pearson', min_periods=1)


# In[ ]:


# create X and y
feature_cols = ['symboling']
X = df1[feature_cols]
y = df1.price


# In[ ]:


#Convertng categorical variable to numeric for make 
#dummy_make = pd.get_dummies(df1['make'], drop_first = True)
#df_make=pd.concat([df1['price'],dummy_make],axis =1)
df1.info()


# In[ ]:


# import, instantiate, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)


# In[ ]:


# print the coefficients
print (linreg.intercept_)
print (linreg.coef_)


# In[ ]:


# Plot the graph between symboling and price
df1.plot(kind='scatter', x='height', y='price', alpha=0.2)


# In[ ]:


# Seaborn scatter plot with regression line
import seaborn as sns
sns.lmplot(x='height', y='price', data=df1, aspect=1.5, scatter_kws={'alpha':0.2})


# In[ ]:



feature_cols = ['length', 'width', 'height']
import seaborn as sns
# multiple scatter plots in Seaborn
sns.pairplot(df1, x_vars=feature_cols, y_vars='price', kind='reg')


# In[ ]:



# multiple scatter plots in Pandas
fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
for index, feature in enumerate(feature_cols):
    df1.plot(kind='scatter', x=feature, y='price', ax=axs[index], figsize=(16, 3))


# In[ ]:


#Line plot for price

df1.price.plot()


# In[ ]:


#boxplot fir price group by length
df1.boxplot(column='price', by='length')


# ## Conclusion
# Price is strongly corealted with drive-wheel_rwd, fuel-system_mpfi
# and negative coorelatred to number-of-cylinder-four. 
# Linear regression cleary show that price is hisgher for negative symboling rating and low for positve symboling
