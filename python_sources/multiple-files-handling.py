#!/usr/bin/env python
# coding: utf-8

# ## Challenges:
# Here, I have tried to show some important topics,
# 1. Part A focuses on retriving particular files with certain string patterns in it using Wildcard characters (.CSV files only), but have some added conditions like onle files that start with "start" and ends with "end", on a top of that removing thoes part and saving it with different names in different foder.
# 2. Part B same as above for .XML extension formatting it to the pandas dataframe and saving it to the .XLSX format.
# 3. Part C deals with treaming some part from a dataframe to Dictionary format.
# 
# Please suggest if I can improve it anymore.

# # Part A
# It focuses on retriving particular files with certain string patterns in it using Wildcard characters (.CSV files only), but have some added conditions like onle files that start with "start" and ends with "end", on a top of that removing thoes part and saving it with different names in different foder.

# In[ ]:


#CSV files only
import pandas as pd
import glob
path = r'/kaggle/input/Python_Assignment_1/Part_A/train/'
filenames = glob.glob(path+"/*.csv")
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))


# In[ ]:


for i in range(0, len(dfs)):
    if (dfs[i].columns[0] == 'start' and dfs[i].index[-1][0] == 'end'):
        dfs[i].drop(['end']).to_csv('file_no_{}'.format(i),header= False)


# # Part B
# Same as Part A above for .XML extension formatting it to the pandas dataframe and saving it to the .XLSX format.

# In[ ]:


path = r'/kaggle/input/Python_Assignment_1/Part_B/CDR/'
files = glob.glob(path+'/*.xml')
files


# In[ ]:


import xml.etree.ElementTree as et  
for i in range(0, len(files)):
    xtree = et.parse(files[3])
    xroot = xtree.getroot() 
    df_cols = ['rowkey',
           'latitude',
           'longitude',
           'signalStrength',
           'uploadedbytes',
           'downloadedBytes']
    out_df = pd.DataFrame(columns = df_cols)
    for node in xroot: 
        a = node.find("rowkey").text if node is not None else None
        b = node.find("latitude").text if node is not None else None
        c = node.find("longitude").text if node is not None else None
        d = node.find("signalStrength").text if node is not None else None
        e = node.find("uploadedbytes").text if node is not None else None
        f = node.find("downloadedBytes").text if node is not None else None
        out_df = out_df.append(pd.Series([a, b, c, d, e, f],
                                     index = df_cols), 
                           ignore_index = True)
    out_df.to_excel('converted{}.xlsx'.format(i+1))


# # Part C
# It deals with treaming some part from a dataframe to Dictionary format.

# In[ ]:


File = pd.read_csv(r'/kaggle/input/Python_Assignment_1/Part_C/part_c_input_file.csv', index_col= 0, header=None)
File


# In[ ]:


a = {}
for i in range(0, len(File)):
    a[File.index[i]] = File.iloc[i].tolist()


# In[ ]:


a


#  ## **Please recommend or suggest better way to do it.**
