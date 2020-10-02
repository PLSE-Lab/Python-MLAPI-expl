#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:


df_odi=pd.read_csv('../input/master-blaster-sachin-tendulkar-dataset/sachin_odi.csv')
df_odi.shape


# In[ ]:


df_odi.info()


# # Columns with '-'

# * For the matches when Sachin did not bowl, wickets and runs_conceded is '-'
# * For the matches when Sachin did not field, catches and stumps is '-'
# * Convert '-' to np.nan 
# * Save the columns as float

# In[ ]:


col_=['wickets','runs_conceded','catches','stumps']
for col in col_:
    df_odi[col][df_odi[col]=='-']=np.nan
    df_odi[col]=df_odi[col].astype('float')
df_odi.describe()


# # DNB, TDNB and not outs in batting_score

# * not out is denoted by '\*' with the score
# * Not out column where, 0: out, 1: not-out, np.nan: DNB and TDNB
# * batting score where TDNB and DNB to np.nan
# * batting score where '\*' to be saved without '\*'
# * saving the column batting_score as float

# In[ ]:


df_odi["notout"]=0
for i in range(df_odi.shape[0]):
    if df_odi.batting_score[i]=='DNB':
        df_odi.batting_score[i]=np.nan
        df_odi.notout[i]=np.nan
    elif df_odi.batting_score[i]=='TDNB':
        df_odi.batting_score[i]=np.nan
        df_odi.notout[i]=np.nan
    elif df_odi.batting_score[i].endswith('*')==True:
        df_odi.batting_score[i]=df_odi.batting_score[i].replace('*','')
        df_odi.notout[i]=1
        
df_odi.batting_score=df_odi.batting_score.astype('float')


# # Remove 'v ' before opposition name

# In[ ]:


for i in range(df_odi.shape[0]):
    df_odi.opposition[i]=df_odi.opposition[i].replace('v ','')


# # Date formatting

# In[ ]:


df_odi.date=pd.to_datetime(df_odi.date, format='%d %b %Y')


# # Rearranging the columns

# * Placing not out after batting_score

# In[ ]:


df_odi=df_odi[['batting_score', 'notout', 'wickets', 'runs_conceded', 'catches', 'stumps',
       'opposition', 'ground', 'date', 'match_result', 'result_margin', 'toss',
       'batting_innings']]


# In[ ]:


df_odi.info()


# # Now the dataset ready to be used for analysis
