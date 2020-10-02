#!/usr/bin/env python
# coding: utf-8

# This notebook is to check and iron out some inconsistencies report in:
# https://www.kaggle.com/c/covid19-global-forecasting-week-1/discussion/137500
# 
# `ConfirmedCases` and `Fatalities` were supposed to be cumulative. But it's been observed not to be always the cases. At such occurrences the value from the previous day is pro-- and iterate over the next (if any) until there is no more day-to-day drop in `ConfirmedCases` and `Fatalities`. The doctored version of train.csv is then written as `trainDoctored.csv`.

# In[ ]:


import os
if 'kid' in os.getcwd():
    HOME = '/home/kid/covid/data'
else:
    HOME = '/kaggle'
GFWPATH = f'{HOME}/input/covid19-global-forecasting-week-1'
import datetime
import pandas as pd


# In[ ]:


def readGFWdata(file):
    df = pd.read_csv(f'{GFWPATH}/{file}.csv', parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    if 'Country_Region' in df.columns:
        countryregion, provincestate = 'Country_Region', 'Province_State'
    else: # for backward compatibility wtih Week 1's data
        countryregion, provincestate = 'Country/Region', 'Province/State'
    df.sort_values(by=[countryregion, provincestate, 'Date'], inplace=True)
    df[provincestate].fillna('', inplace=True)
    return df, countryregion, provincestate
df, countryregion, provincestate = readGFWdata('train')


# In[ ]:


def catchNonCum(df):
    dfDoctored = df.copy()
    okay = True
    for groupID, groupData in df.groupby([countryregion, provincestate]):
        batchA = groupData[['Date', 'ConfirmedCases', 'Fatalities']]
        batchB = batchA[1:].reset_index()
        batchA = batchA[:-1].reset_index()
        assert (batchA['Date'] == batchB['Date'] - datetime.timedelta(1)).all()
        for colname in ['ConfirmedCases', 'Fatalities']:
            checkAB = batchA[colname] > batchB[colname]
            if checkAB.any():
                okay = False
                for idx in checkAB[checkAB].index:
                    print(groupID[1], groupID[0], batchA.loc[idx, 'Date'].strftime("%Y-%m-%d"), 
                                                  batchA.loc[idx, colname], colname)
                    print(groupID[1], groupID[0], batchB.loc[idx, 'Date'].strftime("%Y-%m-%d"), 
                                                  batchB.loc[idx, colname], colname)
                    dfDoctored.loc[batchA['index'].loc[idx]+1, colname] =                     dfDoctored.loc[batchA['index'].loc[idx], colname]
    return dfDoctored, okay

dfDoctored, okay = catchNonCum(df)
if okay:
    print('All clear: no non-cumulative aberration found.')
else:
    nround = 1
    while not okay:
        print('\n*** round #', nround)
        dfDoctored, okay = catchNonCum(dfDoctored)
        nround += 1

if 'Lat' in df.columns: # for backward compatibility wtih Week 1's data
    for colname in ['Lat', 'Long']:
        dfDoctored[colname] = dfDoctored[colname].map(lambda x: '{:.4f}'.format(x))
dfDoctored.to_csv(f'{HOME}/working/trainDoctored.csv', index=False)


# In[ ]:




