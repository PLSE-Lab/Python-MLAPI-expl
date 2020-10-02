#!/usr/bin/env python
# coding: utf-8

# # Every column in every table for all of the tables at kaggle.com/covid-19-contributions
# 
# Using data from https://www.kaggle.com/antgoldbloom/aipowered-literature-review-csvs

# In[ ]:


import numpy as np 
import pandas as pd 
import glob
import os

def list_columns_in_folder(file_path):
    '''List out every column for every file in a folder'''
    for dirname, _, filenames in os.walk(file_path):
        for filename in filenames: 
            df = pd.read_csv(os.path.join(dirname, filename)) 
            columns = df.columns.values[1:] # skip row index
            print(filename+'\nColumns: \n',columns,'\n\n') 


# In[ ]:


list_columns_in_folder('/kaggle/input/aipowered-literature-review-csvs/kaggle/working/key_scientific_questions/')


# In[ ]:


list_columns_in_folder('/kaggle/input/aipowered-literature-review-csvs/kaggle/working/risk_factors/')


# In[ ]:




