import numpy as np 
import pandas as pd 
import os
import pickle

print(os.listdir("../input"))

with open('../input/cols4312p/__full__cols.txt', 'r') as f:
    cols = f.read()
    cols = cols.replace('\n','')
    cols = cols.split(',')

cols.remove('OsVer')
cols.append('HasDetections')
with open('use_cols_list.pickle','wb') as f:
    pickle.dump(cols, f)   

with open('../input/s2-dfnancolumnsdropped/df_nancols_dropped.pickle', 'rb') as f:
    df = pickle.load(f)

df = df.loc[:, cols]

df_noNA = df.dropna(how='any', axis=0)

with open('no_nans.pickle','wb') as f:
    pickle.dump(df_noNA, f)