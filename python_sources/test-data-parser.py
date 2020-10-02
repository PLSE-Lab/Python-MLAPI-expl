# -*- coding: utf-8 -*-
"""
Code to split test data into smaller files to simplify processing
"""
import pandas as pd
import numpy as np

# set path for metadata file
file = 'test_set_metadata.csv'
#create dataframe
df1 = pd.read_csv(file)

# set path for file
file = 'test_set.csv'
#create dataframe
df = pd.read_csv(file)

# Get count of how many objects to split into each dataframe (assuming 20 seperate dataframes ~25M rows each)
#(round(df1.object_id.nunique()/20,0)+1)
#create series of object_id's from metadata
obj = list(df1['object_id'])

# function to chunk through specified number of object in series
def chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# counter - used to name each dataframe in loop below
i = 0
# loop to parse test dataset into smaller chunks (ensuring all observations for each object are in only one dataframe)
for batch in chunk(obj,int(round(df1.object_id.nunique()/20,0)+1)):
    i = i+1
    df2 = df1.loc[df1['object_id'].isin(batch)]
    df2 = df2.merge(df,left_on='object_id',right_on='object_id',how='inner') 
    df_nm = 'df_{}.csv'.format(i)
    df2.to_csv(df_nm)