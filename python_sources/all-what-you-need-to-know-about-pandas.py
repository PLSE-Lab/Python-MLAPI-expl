#!/usr/bin/env python
# coding: utf-8

# <h1>All What You Need to Know about Pandas</h1>
# <p>In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license.[2] The name is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.</p>
# <h3>Library Features</h3>
# <ul>
#     <li>DataFrame object for data manipulation with integrated indexing.</li>
#     <li>Tools for reading and writing data between in-memory data structures and different file formats.</li>
#     <li>Data alignment and integrated handling of missing data.</li>
#     <li>Reshaping and pivoting of data sets.</li>
#     <li>Label-based slicing, fancy indexing, and subsetting of large data sets.</li>
#     <li>Data structure column insertion and deletion.</li>
#     <li>Group by engine allowing split-apply-combine operations on data sets.</li>
#     <li>Data set merging and joining.</li>
#     <li>Hierarchical axis indexing to work with high-dimensional data in a lower-dimensional data structure.</li>
#     <li>Time series-functionality: Date range generation[4] and frequency conversion, moving window statistics, moving window linear regressions, date shifting and lagging.</li>
#     <li>Provides data filtration.</li>
# <ul>
# <h3>What is Dataframe</h3>
# <p>Pandas is mainly used for machine learning in form of dataframes. Pandas allow importing data of various file formats such as csv, excel etc. Pandas allows various data manipulation operations such as groupby, join, merge, melt, concatenation as well as data cleaning features such as filling, replacing or imputing null values.</p>

# <p>first we import pandas, numpy and matplotlib packages to use them in our python kernel, and know the path of our datasets</p>

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('fivethirtyeight')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <h3>Save and Read Dataframes</h3>
# <h4>CSV</h4>
# <p>Read: pd.read_csv(),Save:pd.to_csv()</p>
# <h4>Json</h4>
# <p>Read: pd.read_json(),Save:pd.to_json()</p>
# <h4>Excel</h4>
# <p>Read: pd.read_excel(),Save:pd.to_excel()</p>
# <h4>hdf</h4>
# <p>Read: pd.read_hdf(),Save:pd.to_hdf()</p>
# <h4>SQL</h4>
# <p>Read: pd.read_sql(),Save:pd.to_sql()</p>

# In[ ]:


train_df=pd.read_csv('/kaggle/input/my-dataset/credit_train.csv')
test_df=pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')


# <h3>Exploring our Dataset</h3>
# <p>Next, let's use the DataFrame.head() and DataFrame.info() and DataFrame.tail() methods to refamiliarize ourselves with the data.</p>

# In[ ]:


train_df.head()


# <p>the previous code displays the first five rows of the dataframe</p>

# In[ ]:


train_df.tail()


# <p>the previous code displays the last five rows of the dataframe</p>

# In[ ]:


train_df.info()


# <p>the previous code gives you full describtion about your data like the data type of each column, the number of non-null rows and the shape o your data</p>

# <p>let's use DataFrame.columns, DataFrame.shape, DataFrame.dtypes</p>

# In[ ]:


train_df.columns


# <p>the previous code to Find the name of the columns of the dataframe</p>

# In[ ]:


print('(number of rows, number of columns) : ',train_df.shape)


# <h3>Data Types</h3>
# <p>
# Data has a variety of types.<br>
# The main types stored in Pandas dataframes are <b>object</b>, <b>float</b>, <b>int</b>, <b>bool</b> and <b>datetime64</b>. In order to better learn about each attribute, it is always good for us to know the data type of each column. In Pandas:
# </p>

# In[ ]:


train_df.dtypes


# <p>let's use DataFrame.isnull().sum() to see the number of missing-values in each column</p>

# In[ ]:


train_df.isnull().sum()


# <p>let's use DataFrame.notnull().sum() to see the number of non missing-values in each column</p>

# In[ ]:


train_df.notnull().sum()


# <p>let's use DataFrame.select_dtype(include['Data type we want'])to select columns of a specific data type</p> 

# In[ ]:


object_train_df=train_df.select_dtypes(include=['object'])
object_train_df.columns


# <p>these columns are the columns that have object type</p>

# In[ ]:


num_train_df=train_df.select_dtypes(include=['int','float'])
num_train_df.columns


# <p> we use Series.value_counts() method to display the counts of the unique values in a column</p>

# In[ ]:


for i in object_train_df.columns:
    print(train_df[i].value_counts())
    print('-'*40)


# <p>you can use Series.unique() to get unique values in columns</p>

# In[ ]:


for i in object_train_df.columns:
    print(i)
    print(train_df[i].unique())
    print('-'*40)


# <p>Because pandas is designed to operate like NumPy, a lot of concepts and methods from Numpy are supported. Recall that one of the ways NumPy makes working with data easier is with vectorized operations, or operations applied to multiple data points at once</p>

# <p>to access the values of certain column</p>

# In[ ]:


train_df['Annual Income'].head()


# <p>let's use Vectorized Operations like addition>/p>

# In[ ]:


train_df['Annual Income']+train_df['Monthly Debt']


# <p>We can access the values in the dataframe like the way we access the values in list by using DataFrame.iloc[] or DataFrame.loc[]</p>

# In[ ]:


train_df.iloc[2,5]


# <h3>Statistical Functions in Pandas</h3>

# <h3>Descriptive Statistical Analysis</h3>
# <p>Let's first take a look at the variables by utilizing a description method.</p>
# 
# <p>The <b>describe</b> function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.</p>
# 
# This will show:
# <ul>
#     <li>the count of that variable</li>
#     <li>the mean</li>
#     <li>the standard deviation (std)</li> 
#     <li>the minimum value</li>
#     <li>the IQR (Interquartile Range: 25%, 50% and 75%)</li>
#     <li>the maximum value</li>
# <ul>

# In[ ]:


train_df.describe()


# <h3>Correlation</h3>
# <p>for example, we can calculate the correlation between variables  of type "int64" or "float64" using the method "corr"</p>

# In[ ]:


train_df.corr()


# <p>We can produce heatmap for the correlation by using seaborn package</p>

# In[ ]:


sns.heatmap(train_df.corr())


# <h3>Some Statistical Functions</h3>
# <ul>
#     <li>Series.max() and DataFrame.max()</li>
#     <li>Series.min() and DataFrame.min()</li>
#     <li>Series.mean() and DataFrame.mean()</li>
#     <li>Series.median() and DataFrame.median()</li>
#     <li>Series.mode() and DataFrame.mode()</li>
#     <li>Series.sum() and DataFrame.sum()</li>
#     <li>Series.var() and DataFrame.var():variance</li>
# </ul>

# In[ ]:


print('variance of each column')
train_df.var()


# <h3>handling Missing Data and Modify the Dataframe</h3>

# <p>let's drop some columns of our data by using DataFrame.drop()</p>

# In[ ]:


cols_to_drop=['Loan ID','Customer ID']
train_df=train_df.drop(cols_to_drop,axis=1)
train_df.columns


# <p>now let's fill the missing values in  'Monthly Debt' column with the mean</p>

# In[ ]:


col_mean=train_df['Monthly Debt'].mean()
train_df['Monthly Debt']=train_df['Monthly Debt'].fillna(col_mean)
train_df['Monthly Debt'].isnull().sum()


# <p>WoW now we see there are no missing values</p>

# <p>We Can also drop the missing values from the DataFrame</p>

# In[ ]:


train_df=train_df.dropna()
train_df.shape


# <p>use Series.replace() to replace the values in columns and Series.astype() to convert the data-type of columns</p>

# In[ ]:


mapping_dict = {
    "Years in current job": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
train_df=train_df.replace(mapping_dict)
train_df['Years in current job']=train_df['Years in current job'].astype('int')
train_df['Years in current job'].head()


# <p>Yaaaa! We did it</p>

# <p>We can change column names too by using DataFrame.rename()</p>

# In[ ]:


train_df.rename(columns={'Years in current job':'Years_in_current_job'},inplace=True)
train_df.columns


# <p>Series.map() and Series.apply() methods and confirmed that both methods produce the same results.</p>

# <h3>Compining Data With Pandas</h3>

# <p>We Can combine data using the pd.concat() and pd.merge()</p>

# In[ ]:


all_data=pd.concat([object_train_df,num_train_df],axis=1)
all_data.head()


# <h3>Visualization in Pandas</h3>

# In[ ]:


train_df['Purpose'].value_counts().plot.bar()


# <p>This is for frequency distribution</p>

# In[ ]:


train_df['Years of Credit History'].plot.hist()


# In[ ]:


train_df['Years of Credit History'].plot.kde(label='distribution')
plt.axvline(train_df['Years of Credit History'].mean(),color='red',label='mean')
plt.legend(loc='best')
plt.show()


# <p>I hope this tutorial helps anyone want to start coding in pandas there are alot of functions and methods but here there are alot of basic functions and methods good luck</p>
# <h1>Thanks</h1>
