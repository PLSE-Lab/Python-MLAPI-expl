#!/usr/bin/env python
# coding: utf-8

# # This notebook is made to expolain how to reduce size of big CSV data and make processing faster

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import glob
listBigCSV = glob.glob("../input/NGS*csv")
import os
import tqdm
import gc
import feather
# Any results you write to the current directory are saved as output.


# ## Downcasting for smaller dataset
# Let's load the first dataset.
# We can us C engine to load faster those big files and take pandas function memory_usage to see what we gain. 
# A conversion from bytes to megabytes is needed. So a division by 1024 ** 2 is need.

# In[ ]:


def memory(df):
    if isinstance(df,pd.DataFrame):
        value = df.memory_usage(deep=True).sum() / 1024 ** 2
    else: # we assume if not a df it's a series
        value = df.memory_usage(deep=True) / 1024 ** 2
    return value, "{:03.2f} MB".format(value)


# In[ ]:


df = pd.read_csv(listBigCSV[2], engine='c')


# In[ ]:


df.describe()


# What kind of types do we have ?

# In[ ]:


df.dtypes


# We can select int64 and make them matching both uint8 which takes 1 byte or uint16 (2 bytes).
# Selecting them and applying a smaller types gives:

# In[ ]:


dfIntSelection = df.select_dtypes(include=['int'])
dfConverted2int = dfIntSelection.apply(pd.to_numeric,downcast='unsigned')
memInt, memIntTxt=  memory(dfIntSelection)
memIntDownCast, memIntDownCastTxt = memory(dfConverted2int)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)


# In[ ]:


dfConverted2int.describe()


# A gain of 400% is observed. Wich really good !
# Integer might also be seen as category. It may be intersting if a convertion to a category is a better choice.

# In[ ]:


dfIntSelection = df.select_dtypes(include=['int'])
dfConverted2int = dfIntSelection.astype('category')
memInt, memIntTxt=  memory(dfIntSelection)
memIntDownCast, memIntDownCastTxt = memory(dfConverted2int)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)


# We gain 600%  ! We can apply this methods on float too and downcast them to float.

# In[ ]:


dfFloatSelection = df.select_dtypes(include=['float'])
dfConverted2float = dfFloatSelection.apply(pd.to_numeric,downcast='float')
memInt, memIntTxt=  memory(dfFloatSelection)
memIntDownCast, memIntDownCastTxt = memory(dfConverted2float)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)


# Again, the gain is about 200% !

# There is two types left: Time and Event.
# First can be converted to date time. The second might be a category. 

# In[ ]:


dfTime = df.Time 
date_format = '%Y-%m-%d %H:%M:%S.%f'
dfTime = pd.to_datetime(dfTime,format=date_format)

mem, memTxt = memory(dfTime)
memConv, memConvTxt = memory(dfTime)

print(memTxt)
print(memConvTxt)
print('Gain: ', mem/memConv *100.0)


# We gain more than 1000% converting it.

# # Do we need to do a special function for this ?
# Pandas provides a way to give data types upfront while reading csv file. So we can use it. 

# In[ ]:


dtypes = df.drop('Time',axis=1).dtypes

dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]

column_types = dict(zip(dtypes_col, dtypes_type))
preview = first2pairs = {key:value for key,value in list(column_types.items())[:10]}
import pprint
pp = pp = pprint.PrettyPrinter(indent=4)
pp.pprint(preview)
dfDownCast =pd.read_csv(listBigCSV[2],dtype=column_types,parse_dates=['Time'], infer_datetime_format=True)


# In[ ]:


dfDownCast


# In[ ]:


memInt, memIntTxt=  memory(df)
memIntDownCast, memIntDownCastTxt = memory(dfDownCast)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)


# In[ ]:


dfDownCast.info()


# In[ ]:





# In[ ]:


def downCast(df):
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    converted_obj = df.select_dtypes(include=['int']).astype('category')
    df[converted_obj.columns] = converted_obj
    converted_obj = df.select_dtypes(include=['float']).apply(pd.to_numeric,downcast='float')
    df[converted_obj.columns] = converted_obj
    if 'Time' in df:
        df.Time = pd.to_datetime(df.Time,format=date_format)
    if 'Event' in df:
        df.Event = df.Event.astype('category')
    return df


# In[ ]:


dfDown = df.copy()
dfDown = downCast(dfDown)

memInt, memIntTxt=  memory(df)
memIntDownCast, memIntDownCastTxt = memory(dfDown)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)


# 270% of space gain is really important. Now we can save it as pkl for the next time. And do that for every big dataset.

# In[ ]:


for csvFile in listBigCSV:
    dataframe = pd.read_csv(csvFile, engine='c')
    dataframe = downCast(dataframe)
    dataframe.to_pickle(os.path.basename(csvFile[:-4]+'.pkl'))
    del dataframe


# In[ ]:


os.path.basename('../input/NGS-2016-reg-wk7-12.pkl')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


def downCast(df):
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    converted_obj = df.select_dtypes(include=['int']).astype('category')
    df[converted_obj.columns] = converted_obj
    converted_obj = df.select_dtypes(include=['float']).apply(pd.to_numeric,downcast='float')
    df[converted_obj.columns] = converted_obj
    if 'Time' in df:
        df.Time = pd.to_datetime(df.Time,format=date_format)
    if 'Event' in df:
        df.Event = df.Event.astype('category')
    return df

dfDown = df.copy()
converted_obj = dfDown.select_dtypes(include=['float']).dropna(how='all').astype(np.float32)
dfDown[converted_obj.columns] = converted_obj

dfDown = dfDown[~dfDown['GSISID'].isna()]
dfDown.Season_Year = dfDown.Season_Year.astype('int32')
dfDown.PlayID = dfDown.PlayID.astype('int32')
dfDown.GSISID = dfDown.GSISID.astype('int32')
dfDown.GameKey = dfDown.GameKey.astype('int32') #7610563 7610563
dfDown.Event = dfDown.Event.astype('category') #7610563 7610563
date_format = '%Y-%m-%d %H:%M:%S.%f'
dfDown.Time = pd.to_datetime(df.Time,format=date_format)

memInt, memIntTxt=  memory(df)
memIntDownCast, memIntDownCastTxt = memory(dfDown)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)


# In[ ]:





# In[ ]:


df_list = []
for i in tqdm.tqdm(listBigCSV):
    dfDown =  pd.read_csv(i, engine='c')
    converted_obj = dfDown.select_dtypes(include=['float']).dropna(how='all').astype(np.float32)
    dfDown[converted_obj.columns] = converted_obj
    dfDown = dfDown[~dfDown['GSISID'].isna()]
    dfDown.Season_Year = dfDown.Season_Year.astype('int32')
    dfDown.PlayID = dfDown.PlayID.astype('int32')
    dfDown.GSISID = dfDown.GSISID.astype('int32')
    dfDown.GameKey = dfDown.GameKey.astype('int32') #7610563 7610563
    dfDown.Event = dfDown.Event.astype('category') #7610563 7610563
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    dfDown.Time = pd.to_datetime(dfDown.Time,format=date_format)
    df_list.append(dfDown)


# In[ ]:


# Merge all dataframes into one dataframe
ngs = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
gc.collect()

# Convert Time to datetime
# ngs['Time'] = pd.to_datetime(ngs['Time'], format='%Y-%m-%d %H:%M:%S')

# See what we have loaded
ngs.info()


# In[ ]:


ngs.Event = ngs.Event.astype('category')
ngs.info()


# In[ ]:


ngs.Time = ngs.Time.astype('datetime64[ms]')
ngs.info()


# In[ ]:


def calculate_speeds_2(df, dt=None, SI=False):
    data_selected = df[['Time', 'x','y']]
    if SI==True:
        data_selected.x = data_selected.x / 1.0936132983
        data_selected.y = data_selected.y / 1.0936132983
    # Might have used shift pd function ?
    data_selected_diff = data_selected.diff()
    if dt==None:
        # Time is now a timedelta and need to be converted
        #data_selected_diff.Time = data_selected_diff.Time.apply(lambda x: (x.total_seconds()))
        data_selected_diff['Speed'] = np.sqrt(data_selected_diff.x **2 + data_selected_diff.y **2).astype(np.float64) / data_selected_diff.Time.values.astype(np.float64)
    else:
        # Need to be sure about the time step...
        data_selected_diff['Speed'] = (data_selected_diff.x **2 + data_selected_diff.y **2).astype(np.float64).apply(np.sqrt) / dt
    #data_selected_diff.rename(columns={'Time':'TimeDelta'}, inplace=True)
    #return data_selected_diff
    df['TimeDelta'] = data_selected_diff.Time
    df['Speed'] = data_selected_diff.Speed
    return df[1:]

def get_speed_2(df):
    df_with_speed = df.copy()
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
    df_with_speed.Time = pd.to_datetime(df_with_speed.Time, format =date_format)
    df_with_speed.sort_values(sortBy, inplace=True)
    df_with_speed = calculate_speeds_2(df_with_speed, SI=True)
    cut_speed=100 / 9.58 # World record 9,857232 m/s for NFL
    df_with_speed = remove_wrong_values(df_with_speed, cutspeed=cut_speed)
    return df_with_speed

def remove_wrong_values(df, tested_columns=['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'TimeDelta'], cutspeed=None):
    dump = df.copy()
    colums = dump.columns
    mask = []
    for col in tested_columns:
        dump['shift_'+col] = dump[col].shift(-1)
        mask.append("( dump['shift_"+col+"'] == dump['"+col+"'])")
    mask =eval(" & ".join(mask))
    # Keep results where next rows is equally space
    dump = dump[mask]
    dump = dump[colums]
    if cutspeed!=None:
        dump = dump[dump.Speed < cutspeed]
    return dump

