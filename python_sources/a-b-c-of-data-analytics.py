#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. Import the libraries 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import scipy.stats as ss
import missingno as msno

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

#no boundation on number of columns 
pd.set_option('display.max_columns', None)

#no boundation on number of rows
pd.set_option('display.max_rows', None)

# run multiple commands in a single jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # List of Functions used

# In[ ]:


#Heatmap
def heatmap(df):
    
    '''
    this function takes data frame a input and returns the
    heatmap as output.
    
    Arguments
    ====================
    df : Dataframe or Series 
    
    
    Returns
    ===========
    heatmap
    '''
    corr = df.corr()   #create a correlation df
    fig,ax = plt.subplots(figsize = (30,30))   # create a blank canvas
    colormap = sns.diverging_palette(220,10, as_cmap=True)   #Generate colormap
    sns.heatmap(corr, cmap=colormap, annot=True, fmt='.2f')   #generate heatmap with annot(display value) and place floats in map
#    plt.xticks(range(len(corr.columns)), corr.columns);   #apply  xticks(labels of features)
#    plt.yticks(range(len(corr.columns)), corr.columns)   #apply yticks (labels of features)
    plt.show()
    
    

def categorical_feature_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    
    helper function that gives a quick summary of a given column of categorical data
    
    Arguments
    =============
    dataframe: pandas df
    x: str, horizontal axis to plot the label of categorical data
    y: str, vertical xis to plot hte label of the categorical data
    hue: str, if you want to comparer it to any other variable
    palette: array-like, color of the graph/plot
    
    
    Returns
    ==============
    Quick summary of the categorical data
    
    
    '''
    
    if x==None:
        column_interested = y
        
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode', series.mode())
    
    if verbose:
        print('='*80)
        print(series.value_counts())
        
        
    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette = palette)
    plt.show()
    
    '''
    
    categorical_summarized does is it takes in a dataframe, together with some input arguments and 
    outputs the following:
        1. The count, mean, std, min, max, and quartiles for numerical data or
        2. the count, unique, top class, and frequency of the top class for non-numerical data.
        3. Class frequencies of the interested column, if verbose is set to 'True'
        4. Bar graph of the count of each class of the interested column
    
    '''


# # 2. Import the Datasets

# In[ ]:


tr  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
tt = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# # 3.Basic Information

# Let's check out some of the basic information of our datasets

# ## 3.1.Shape and Size

# Lets checkout the Shape and size of our datasets

# In[ ]:


print('train shape:',tr.shape)
print('test shape:', tt.shape)


# In[ ]:


print('train size:', tr.size)
print('test size:', tt.size)


# Since we got the shape and size of the dataset, we now know about the number of instances and number of columns.
# 
# Lets start to explore the dataset

# ## 3.2.Let's have a peek

# Now that we have the shape of our datasets, lets see how our dataset looks like:

# In[ ]:


print('Head:')
tr.head()  # prints first five instances of the dataset
print('Tail:')
tr.tail()  #prints last five instances of the dataset
print('5 Random samples from the train dataset:')
tr.sample(5)  #5 random samples of the dataset


# ## 3.3. Columns/Features of the Dataset:

# In[ ]:


print('Columns of the train dataset:')
tr.columns


# Seeing the differences in the shapes of train and test datasets, its very obvious that the only difference in the number of features of train and test datasets  is the "SalePrice".
# 
# Still if one wants to check the uncommon features in two datasets, here we go:

# In[ ]:


print('columns in train but not in test:')
tr.columns.difference(tt.columns)


# Now lets see the common features in  both the columns

# In[ ]:


print('colummns in common train and test')
(tr.columns).intersection(tt.columns)


# ## 3.4. Info and Describe

# Let us gather some information about the features and indices : Name, Datatypes,value-counts and non-null values: 

# In[ ]:


print('Information on train dataframe:')

tr.info()


# Describe:

# In[ ]:


print('description of the training dataset:')
tr.describe()


# # 4. Missing Values

# Lets check the missing values in the dataframe

# In[ ]:


tr.isnull().sum().sort_values(ascending=False)


# We'll drop the features where the missing data exceeds 50% of the range of indices.

# Grand total of the missing values:

# In[ ]:


print('total null values:', tr.isna().sum().sum(), 
      'out of', tr.size, '(total entries)' )

Let us visualize the pattern in our missing data
# In[ ]:


tr_num = tr._get_numeric_data()


# In[ ]:


missing_bar = msno.bar(tr_num)


# The Bar graph above shows the frequency of missing data.

# In[ ]:


missing_data_matrix = msno.matrix(tr_num, )


# The matrix above shows the distribution of the missing data in the dataset with numerical features. It helps in deciding the strategy to fill up the missing values.

# In[ ]:


train_missing_data_heatmap = msno.heatmap(tr)


# The Heatmap of the missing data in the complete training dataset.

# And finally a dendrogram to show the relations between two variables.
# 

# In[ ]:


missing_dendrogram = msno.dendrogram(tr)


# We'll handle the missing data, but before that we'll split the dataframe based on numerical and categorical entry of the features.

# The degrees of difference in the null values is very high between Fence and SaleCondition.

# # 5. Divide into Categorical and Numerical features

# In[ ]:


print('Dataframe of Features with numerical features')
tr_num = tr._get_numeric_data()
tr_num.head()


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

tr_cat = tr.select_dtypes(exclude = numerics)
print('dataframes with categorical features:')
tr_cat.head()


# Lets check out the columns with numerical and categorical variables

# In[ ]:


print('list of features with numerical values:')
tr_num.columns

print('list of features with categorical values:')
tr_cat.columns


# In[ ]:


print('Shape of the numerical DataFrame:', tr_num.shape)
print('Shape of the Categorical DataFrame:', tr_cat.shape)


# # 5.1. Numerical Data in Train

# Lets dig some deeper and get Statistical Insights 

# Lets start with SalePrice(target variable), plot a distribution of the frequency

# In[ ]:


# Start with an empty canvas
ax,fig = plt.subplots(1,3,figsize=(20,5))

#first plot - disprinbution plot
dist = sns.distplot(tr_num['SalePrice'],bins=50, ax = fig[0])
title = dist.set_title('Distribution Plot');

#second plot - box plot
box = sns.boxplot(x = tr_num['SalePrice'], linewidth=2.5, ax=fig[1])
title_0 = box.set_title('Boxplot')

#third plot - violin plot
violin = sns.violinplot(x=tr_num['SalePrice'], ax=fig[2])
title_1 = violin.set_title('Violin Plot')

plt.show()


# Observations:
# 1. The stars in the boxplot beyond the line (non continuous frequency) indicates that there are only a few houses that are highly priced.
# 2. About 75% of the houses are sold for less than $210000.
# 
# As we see the distribution is Positively Skewed, so lets check out the Measure of the shape
# 
# 

# In[ ]:


print("Average Price of the House:", tr_num['SalePrice'].mean())
print("Most of the houses are close to" ,tr_num['SalePrice'].mode())
print('The median house price is close to'  , tr_num['SalePrice'].median())


# In[ ]:


num_features = tr_num.columns


print('Skewness:', tr.SalePrice.skew())
print('Kurtosis:', tr.SalePrice.kurt())


# Observations (Skewness):
# `
# 1. The positive Skewness indicates that the data is very widely distributed i.e. there is very small number of houses with each price tag(frequency)
# 2. The order of central frequency in SalePrice:
#             Mean>Median>Mode
# 3. Many houses are sold for less than the average price       
# 
# 

# Observations (Kurtosis):
# 
# 1. The plot of the SalePrice is distorted bell-shaped with sharp peak
# 2. Most houses are sold at a price very close to $140000(mode), which is less than the average household price.
# 3. There are many outliers in the Dataset(Keep this in mind while training your model).
# 

# # Correlation

# Before we jump into analysing indiviadual variables, lets decide which variables to pick. 

# In[ ]:


numerical_correlation = tr_num.corr()
numerical_correlation


# This is correlation matrix (ratio of relations between any two variables). It is hard to draw conclusive insights from this matrix.
# 
# Let's visualize it.

# In[ ]:


heatmap(tr_num)


# In[ ]:


tr_num.corr()['SalePrice'].sort_values(ascending=False)


# Quite simpler but still need to simplify it further

# Features Selected for correlation

# In[ ]:


features_for_corr = {'OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath'}


# In[ ]:


corr_some_features = tr[features_for_corr].corr()
ax,fig = plt.subplots(figsize=(10,10))
colormap = sns.diverging_palette(220,10, as_cmap=True)
heatmap_some_features = sns.heatmap(corr_some_features, cmap = colormap, annot=True, annot_kws={'size':20}, fmt='.2f')
plt.show()

Lets check out the distribution of each of the feeatures in features_for_corr list
# In[ ]:


for i in features_for_corr:
    ax,fig = plt.subplots(1,3, figsize=(20,5))
    box = sns.boxplot(y = tr_num[i], linewidth=2.5, ax=fig[2])
    box_title = box.set_title('Box Plot')
    violin = sns.violinplot(y = tr_num[i], linewidth=2.5, ax=fig[0])
    violin_title  = violin.set_title('Violin Plot')
    dist = sns.distplot(tr_num[i], ax=fig[1], vertical=True)
    distplot_title = dist.set_title('Distribution Plot')


# Draw similar conclusions from above features as we had drawn in SalePrice

# In[ ]:


for i in features_for_corr:
    skewness = np.array(tr_num[features_for_corr].skew())
    kurtosis = np.array(tr_num[features_for_corr].kurt())
    mean = np.array(tr_num[features_for_corr].mean())
    median = np.array(tr_num[features_for_corr].median())
    variance = np.array(tr_num[features_for_corr].var())
    
    data_frame = pd.DataFrame({'skewness':skewness,
                               'kurtosis':kurtosis, 
                               'Mean':mean,
                               'Median':median, 
                               'variance':variance},
                              
                              index=features_for_corr,
                              columns={'skewness',
                                       'kurtosis',
                                       'Mean',
                                       'Median',
                                       'variance'})
print(data_frame)


# The above Matrix shows the variance, Skewness, Kurtosis, Mean and Median of the selected numerical features from the train dataset

# # Categorical Features

# In[ ]:


tr_cat = tr.select_dtypes(include = 'object')
tr_cat.head()


# In[ ]:


for i in tr_cat.columns:
    categorical_feature_summarized(tr_cat, x=i)


# # Conclusion

# After the data is extensively analysed we can start follow the following order to train our model:
#     1. Missing Data Handeling
#     2. Data Cleaning
#     3. Data Preprocessing
#     4. Feaature Engineering
#     5. Feature Selection
#     6. Training our model
#     7. Error Analysis
#     8. Optimization of the model
#     9. Submission of the Predictions
