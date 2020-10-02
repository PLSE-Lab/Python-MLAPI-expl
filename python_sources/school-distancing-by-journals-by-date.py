#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook we will explore School distancing csv.
# Then we will create Contingency table to summarize data by 2 variables by Journals and by date. 
# 
# We have organized data as number of journals of particular type published on particular date.
# 

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


# # Imports

# In[ ]:


#Import Packages

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'


# # Read data

# In[ ]:


#/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/Effectiveness of school distancing.csv

#read file
fname = '/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/Effectiveness of school distancing.csv'
school_dist_df = pd.read_csv(fname)
school_dist_df


# # Explore data

# In[ ]:


school_dist_df.shape


# In[ ]:


school_dist_df.info()


# In[ ]:


school_dist_df.head(10)


# In[ ]:


school_dist_df.tail(10)


# In[ ]:


school_dist_df.columns


# In[ ]:


school_dist_df.groupby('Journal')['Unnamed: 0'].count()


# In[ ]:


school_dist_df.groupby('Date')['Unnamed: 0'].count()


# In[ ]:


school_dist_df.groupby('Added on')['Unnamed: 0'].count()


# In[ ]:


school_dist_df.groupby('Measure of Evidence')['Unnamed: 0'].count()


# In[ ]:


school_dist_df.groupby('Influential')['Unnamed: 0'].count()


# In[ ]:


school_dist_df.groupby('Study Type')['Unnamed: 0'].count()


# In[ ]:






journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count')
journal_moe_df1
journal_moe_df1.index

journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count').unstack()


# # Summary Table

# In[ ]:




def summary_table(col1, col2, col_index, data_frame):
    var1 = col1 + '_' + col2 + '_df1' 
    print(var1)
    var1 = data_frame.groupby([col1,col2])[col_index].aggregate('count')
    print(var1)
    print('\n')
    print(var1.index)
    print('\n')
    
    var2 = col1 + '_' + col2 + '_df2'
    print(var2)
    var2 = data_frame.groupby([col1,col2])[col_index].aggregate('count').unstack()
    print(var2)
    print('\n')
    var2

    var3 = col1 + '_' + col2 + '_df3'
    print(var3)
    var3 = var2.fillna(0)
    print(var3)
    print('\n')


    arr = var3.values
    print('arr')
    print(arr)
    print('\n')
    
    # Create a new row at bottom to hold COLUMN TOTALS
    new_row = np.zeros(shape=(1,arr.shape[1]))
    new_row

    # Vertically stack new row of zeroes at bottom
    arr1 = np.vstack([arr, new_row])
    print('arr1')
    print(arr1)
    print('\n')

    # Create a new column at right to hold ROW TOTALS
    new_col = np.zeros(shape=(arr1.shape[0],1))
    new_col

    #arr1.shape
    #new_col.shape


    # Horizontally stack new column of zeroes at right
    arr2 = np.hstack([arr1, new_col])
    print('arr2')
    print(arr2)
    print('\n')
    
    arr3 = arr2.copy()
    

    # Fill last row with sum of all values in each column
    arr3[ arr3.shape[0]-1 ]=arr2.sum(axis=0) # column total
    
    

    # Fill last column with sum of all values in each row
    arr3[ :, arr3.shape[1]-1 ] = arr3.sum(axis=1) # row total
    print('arr3')
    print(arr3)
    
summary_table('Journal', 'Date', 'Unnamed: 0', school_dist_df)


# Above output summarizes two variables: Journal and Date.
# 
# 
# A table that summarizes data for two categorical variables in this way is called a Contingency table. 
# Last row contains Column Total.
# Last column contains Row Total.
# 
# Each value in the table represents the number of times a particular combination of variable outcomes occurred. 
# 
# There are 4 rows for Journal types namely ArXiv, Lancet Infect Dis ,MedRxiv and 
# medRxiv
# 
# There are 6 columns for dates on which those journals are dated namely 2020-03-16  2020-03-23  2020-04-14  2020-04-21  2020-04-22 and 2020-05-05.
# 
# For eg, the value 3 in second row of Journal 'Lancet Infect Dis' and under second column of date '2020-03-23' , corresponds to count of journals of by type 'Lancet Infect Dis'  published on date '2020-03-23'.

# In[ ]:





# In[ ]:



journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count')
journal_moe_df1
journal_moe_df1.index

journal_moe_df1 = school_dist_df.groupby(['Journal','Measure of Evidence'])['Unnamed: 0'].aggregate('count').unstack()


# In[ ]:





# In[ ]:





# In[ ]:




