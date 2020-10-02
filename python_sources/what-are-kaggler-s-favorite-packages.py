#!/usr/bin/env python
# coding: utf-8

# ***`Python Implementation`*** of [Kaggler's Favorite Packages](http://www.kaggle.com/rebeccaturner/kaggler-s-favorite-packages) by  **[Rachael Tatman (non-admin)](https://www.kaggle.com/rebeccaturner)**.
# 
# 
# Link for the live coding session video showing ***`R Implementation`***  **[Kaggle Live-Coding: What are Kaggler's favorite packages?](https://www.youtube.com/watch?v=AfQhklPwE6o)**

# This dataset contains information on what packages were imported in a sample of public R and Python kernels. Kernels were selected by looking for notebooks that contained the word "import" (for Python) or "library" (for R). Then the names of the packages were extracted using the following regular expressions:
# 
# `For R: "[library|require]\\(([a-zA-Z\\.0-9']*)\\)"`
# 
# `For Python: "(?`

# In[ ]:


import pandas as pd
pd.set_option('max_rows',100)
pd.set_option('max_columns',400)
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


PYTHON_PATH = '../input/py_packages.json'
R_PATH = '../input/r_packages.json'


# In[ ]:


def load_and_preprocessing(PATH):
    # Load dataframe from PATH.
    df = pd.read_json(PATH)    
    
    library = pd.DataFrame(columns=['Import','Name','NotebookId'])
    for i, row in df.iterrows():
        for j, column in row.iteritems():
            if column != None:
                column.append(f'Notebook {i+1}')
                library = library.append(pd.Series(column, index=['Import','Name','NotebookId'] ), ignore_index=True)
            
    # Using RegEx for data cleaning.
    """
    CASE 1:
        Correcting the importing of package in 'r_packages.json'.
    Case 2:
        Few of the names have quotes around the package names.
    """
    # Case 1
    library.Import.replace(['^y\('],['library('],regex=True,inplace=True)
    
    # Case 2
    library.Name.replace(['^\''],[''],regex=True,inplace=True)
    library.Name.replace(['\'$'],[''],regex=True,inplace=True)
    
    
    # Removing values with length in columns Name and Import
    library[(library.Name.str.len()>1)&(library.Import.str.len()>1)]
    
    library[['Name','Import']] = library[['Name','Import']].astype('category')
    return library


# In[ ]:


PYTHON = load_and_preprocessing(PYTHON_PATH)
PYTHON.name = 'Python' 
R = load_and_preprocessing(R_PATH)
R.name = 'R'


# In[ ]:


PYTHON.info()


# In[ ]:


R.info()


# In[ ]:


PYTHON.head(10)


# In[ ]:


R.head(10)


# In[ ]:


pd.DataFrame(PYTHON.Name.value_counts(sort=True,ascending=False)).head(10)


# In[ ]:


pd.DataFrame(R.Name.value_counts(sort=True,ascending=False)).head(10)


# ## Visualization

# In[ ]:


from mlxtend.frequent_patterns import apriori

def plot_most_frequent(dd):
    # Group Packages based on Notebooks 
    df = pd.DataFrame(dd.groupby('NotebookId')['Name'].apply(list).reset_index(name = "Packages"))
    
    # Creating Sparse matrix to be used in Apriori
    association=df.drop('Packages', 1).join(df.Packages.str.join('|').str.get_dummies())
    association.drop('NotebookId',axis=1,inplace=True)
    
    # Apriori using mlxtend.frequent_patterns
    frequentItems = pd.DataFrame(apriori(association, min_support=0.2, use_colnames=True))
    frequentItems.sort_values(by='support',ascending=False,inplace=True)
    
    # Horizontal Bar Plot showing support and frequent items on X and Y axis respectively    
    frequentItems.plot(kind='barh',x='itemsets',y='support',title=f'Libraries Most Frequently Used in {dd.name}',sort_columns=True,figsize = (10,5),legend=False)

def count_of_libraries(dd):
    # Creating subplots of size 20X10
    fig, axes = plt.subplots(ncols=2,figsize = (20,10))
    
    # Grouping NotebookIds to find count of libaries used in each notebook.
    df = pd.DataFrame(dd.groupby('NotebookId').size().reset_index(name = "Count"))
    
    # Plotting Probability Distribution and Box plot for the count of libraries used.
    df.plot(kind='kde',x='NotebookId',y="Count",sort_columns=True,legend=False,ax=axes[0],title='Count of Libraries used in {dd.name}',xlim=(df['Count'].min()-1,df['Count'].max()+1))
    df.plot(kind='box',y="Count",grid=False,ax=axes[1],title='Count of Libraries used in {dd.name}')


# ## Libraries which are used frequently together :-

# In[ ]:


plot_most_frequent(R)


# In[ ]:


plot_most_frequent(PYTHON)


# ## Analysis of count of packages used per notebook :-

# In[ ]:


count_of_libraries(PYTHON)


# In[ ]:


count_of_libraries(R)


# <center><b><i>You found this notebook helpful or you just liked it , some upvotes would be very much appreciated<br>That will keep me motivated :)</i></b></center>
