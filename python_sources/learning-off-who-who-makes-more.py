import itertools

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy import stats

columns = ['PUMA', 'ST', 'AGEP', 'SEX', 'WAGP', 'JWTR']

# the exemple is ran for colorado state
ST = 8  # state code
PUMA = None  # 703, 1600, 100, 3302, 300  
SEX = 1 # 1 MALE, 2 FEMALE
#################
# load dataframes
# I need to do this by chunks, my laptop won't hold otherwise..

dfa = pd.DataFrame(columns=columns)
chunka = pd.read_csv('../input/pums//ss13pusa.csv', chunksize=1000, usecols=columns)

while True:
    try:
        sub_df = chunka.get_chunk()

        sub_df = sub_df[(sub_df['ST'] == ST) & (sub_df['SEX'] == SEX)]
        
        if PUMA:
            sub_df = sub_df[sub_df['PUMA'] == PUMA]
            
        sub_df = sub_df.dropna()

        dfa = pd.concat([sub_df, dfa])
    except:
        break
    
dfb = pd.DataFrame(columns=columns)

chunkb = pd.read_csv('../input/pums//ss13pusb.csv', chunksize=1000, usecols=columns)

while True:
    try:
        sub_df = chunkb.get_chunk()

        sub_df = sub_df[(sub_df['ST'] == ST)  & (sub_df['SEX'] == SEX)]
        if PUMA:
            sub_df = sub_df[sub_df['PUMA'] == PUMA]

        sub_df = sub_df.dropna()

        dfb = pd.concat([sub_df, dfb])
    except:
        break
    
df = pd.concat([dfa, dfb])
print("n observations: ", len(df))