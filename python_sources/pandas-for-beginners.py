#!/usr/bin/env python
# coding: utf-8

# <img src="https://i1.wp.com/www.datapluspeople.com/wp-content/uploads/2018/04/pandas_logo-1080x675.jpg?resize=1080%2C675&ssl=1">

# # Creating DataFrame
# <img src="https://i.ibb.co/NrmN8RV/Screenshot-2018-12-23-at-9-39-32-AM.png">
# <img src="https://i.ibb.co/pWTmFGw/Screenshot-2018-12-23-at-9-42-26-AM.png">
# <img src="https://i.ibb.co/7ppNMLm/Screenshot-2018-12-23-at-10-09-07-AM.png">

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.DataFrame(np.random.rand(3, 2))


# In[ ]:


df.head()


# In[ ]:


df = pd.Series(np.random.rand(3))


# In[ ]:


df.head()


# In[ ]:


df = pd.Series(np.random.rand(3), index=["First", "Second", "Third"])


# In[ ]:


df


# In[ ]:


df.index


# In[ ]:


df_2d = pd.DataFrame(np.random.rand(3, 2))


# In[ ]:


df_2d


# In[ ]:


df_2d.columns = ["First", "Second"]


# In[ ]:


df_2d


# In[ ]:


df_2d.rename(index={0:'zero', 1:'one', 2:'Two'}, inplace=True)


# In[ ]:


df_2d


# # Exploring Pandas Data Input Capabilities
# <img src="https://i.ibb.co/nQqnDTW/Screenshot-2018-12-23-at-10-36-44-AM.png">
# <img src="https://i.ibb.co/0DCYdx8/Screenshot-2018-12-23-at-10-37-43-AM.png">
# 
# ## Reading from csv
# <img src="https://i.ibb.co/nQyPDyN/csv-file.png">

# In[ ]:


df = pd.read_csv('../input/artwork/artwork_data.csv')


# In[ ]:


df.head() # 1st 5 rows


# In[ ]:


df.tail() # bottom to 5 rows


# In[ ]:


df.shape


# In[ ]:


df_5rows = pd.read_csv('../input/artwork/artwork_data.csv', nrows=5) # Selecting 1st 5 rows


# In[ ]:


df_5rows


# In[ ]:


df_5rows_ind = pd.read_csv('../input/artwork/artwork_data.csv', nrows=5, index_col='id') # intialized index = id


# In[ ]:


df_5rows_ind


# In[ ]:


df_specific_col = pd.read_csv('../input/artwork/artwork_data.csv', nrows=5, usecols=['id', 'artist']) # all columns that we need


# In[ ]:


df_specific_col


# In[ ]:


cols_to_use = ['id', 'artist', 'title', 'medium', 'year', 'acquisitionYear', 'height', 'width', 'units']


# In[ ]:


df = pd.read_csv('../input/artwork/artwork_data.csv', usecols=cols_to_use, index_col = 'id')


# In[ ]:


df.head()


# In[ ]:


df.to_pickle('Selected_Col.pickle')


# ## Reading from json
# <img src="https://i.ibb.co/1sFdmX5/Example-of-a-webform-application-JSON-data-object-as-submitted-by-a-user.png">

# In[ ]:


import json


# In[ ]:


records = [("Espresso", "5$"),
          ("Flat White", "10$")]


# In[ ]:


pd.DataFrame.from_records(records)


# In[ ]:


pd.DataFrame.from_records(records, columns=["coffee", "Price"])


# In[ ]:


key_to_use = ['id', 'all_artists', 'title', 'medium', 'dateText',
             'acquisitionYear', 'height', 'width', 'units']


# In[ ]:


def get_record_from_file(file_path, keys_to_use):
    '''Process single json file and return a  tuple
    containing specific fields.'''
    with open(file_path) as artwork_file:
        content = json.load(artwork_file)
    
    record = []
    for field in keys_to_use:
        record.append(content[field])
    
    return tuple(record)


# In[ ]:


# Single file processing function
import os
sample_json = os.path.join('../input/data12', 'a00102-1738.json')


# In[ ]:


sample_record = get_record_from_file(sample_json, key_to_use)


# In[ ]:


sample_record


# In[ ]:


def read_artworks_from_json(keys_to_use):
    ''' Traverse the directories with JSON files.
    For first file in each directory call function
    for processing single file and go to the next
    directory.
    '''
    json_root = os.path.join('dataa')
    artworks = []
    for root, _, files in os.walk(json_root):
        for f in files:
            if f.endswith('json'):
                record = get_record_from_file(os.path.join(root, f), keys_to_use)
                artworks.append(record)
    
    df = pd.DataFrame.from_records(artworks, columns=keys_to_use, index='id')
    return df


# In[ ]:


df = read_artworks_from_json(key_to_use)


# In[ ]:


df.head()


# # Indexing and Filtering

# In[ ]:


df = pd.read_pickle('Selected_Col.pickle')


# In[ ]:


artists = df['artist']
x = pd.unique(artists)
len(x)


# In[ ]:


s = df['artist'] == 'Bacon, Francis'
s.value_counts()


# **Note**
# * **loc** by label
# * **iloc** by position
# <img src="https://i.ibb.co/7g7yT1r/Screenshot-2018-12-23-at-1-09-23-PM.png">
# <img src="https://i.ibb.co/zHgCJ7J/Screenshot-2018-12-23-at-1-09-44-PM.png" >
# <img src="https://i.ibb.co/ynD72y4/Screenshot-2018-12-23-at-1-10-13-PM.png">
# <img src="https://i.ibb.co/d5wnGZX/Screenshot-2018-12-23-at-1-11-17-PM.png">

# In[ ]:


df.loc[1035, 'artist']


# In[ ]:


df.iloc[0, 0]


# In[ ]:


df.iloc[0:, ]


# In[ ]:


df.iloc[0:2, 0:2]


# In[ ]:


df['width'].sort_values().head()


# In[ ]:


df['width'].sort_values().tail()


# In[ ]:


# pd.to_numeric(df['width']) # Try to convert to numeric value
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# pandas/_libs/src/inference.pyx in pandas._libs.lib.maybe_convert_numeric()

# ValueError: Unable to parse string "(upper):"

# During handling of the above exception, another exception occurred:

# ValueError                                Traceback (most recent call last)
# <ipython-input-49-6faa34faa68a> in <module>()
# ----> 1 pd.to_numeric(df['width']) # Try to convert to numeric value

# /opt/anaconda3/lib/python3.6/site-packages/pandas/core/tools/numeric.py in to_numeric(arg, errors, downcast)
#     131             coerce_numeric = False if errors in ('ignore', 'raise') else True
#     132             values = lib.maybe_convert_numeric(values, set(),
# --> 133                                                coerce_numeric=coerce_numeric)
#     134 
#     135     except Exception:

# pandas/_libs/src/inference.pyx in pandas._libs.lib.maybe_convert_numeric()

# ValueError: Unable to parse string "(upper):" at position 1839


# In[ ]:


pd.to_numeric(df['width'], errors='coerce')


# In[ ]:


df.loc[:, 'width'] = pd.to_numeric(df['width'], errors='coerece')


# In[ ]:


df.loc[:, 'height'] = pd.to_numeric(df['height'], errors='coerece')


# In[ ]:


df['height'] * df['width']


# In[ ]:


df['units'].value_counts()


# In[ ]:


area = df['height'] * df['width']
df = df.assign(area=area)


# In[ ]:


df['area'].max()  # maximum value


# In[ ]:


df['area'].idxmax() # return index


# In[ ]:


df.loc[df['area'].idxmax(), :]


# # Operations on groups

# In[ ]:


small_df = df.iloc[49980:50019, :].copy()


# In[ ]:


grouped = small_df.groupby('artist')


# In[ ]:


type(grouped)


# In[ ]:


for name, group_df in grouped:
    print(name)
    print(group_df)
    break


# In[ ]:


def fill_values(series):
    values_counted = series.value_counts()
    if values_counted.empty:
        return series
    most_freq = values_counted.index[0]
    new_medium = series.fillna(most_freq)
    return new_medium


# In[ ]:


def transform_df(source_df):
    groups_df = []
    for name, group_df in source_df.groupby('artist'):
        filled_df = group_df.copy()
        filled_df.loc[:, 'medium'] = fill_values(group_df['medium'])
        groups_df.append(filled_df)
    
    new_df = pd.concat(groups_df)
    return new_df


# In[ ]:


filled_df = transform_df(small_df)


# In[ ]:


filled_df.head()


# ## Built-Ins
# ### Transform

# In[ ]:


grouped_mediums = small_df.groupby('artist')['medium']
small_df.loc[:, 'medium'] = grouped_mediums.transform(fill_values)


# ### Min

# In[ ]:


df.groupby('artist').agg(np.min)


# In[ ]:


df.groupby('artist').min()


# ### Filter

# In[ ]:


grouped_titles = df.groupby('title')
title_counts = grouped_titles.size().sort_values(ascending=False)


# In[ ]:


condition = lambda x: len(x.index) > 1
dup_titles_df = grouped_titles.filter(condition)
dup_titles_df.sort_values('title', inplace=True)


# In[ ]:


dup_titles_df.head()


# # Outputting Data

# In[ ]:


# Saving to excel file
small_df.to_excel("basic.xlsx")
small_df.to_excel("no_index.xlsx", index=False)
small_df.to_excel("columns.xlsx", columns=["artist", "title", "year"])


# In[ ]:


# Multiple worksheets
writer = pd.ExcelWriter('multiple_sheets.xlsx', engine='xlsxwriter')
small_df.to_excel(writer, sheet_name="Preview", index=False)
df.to_excel(writer, sheet_name='Complete', index=False)
writer.save()


# In[ ]:


# SQL Format
import sqlite3
with sqlite3.connect('my_database.db') as conn:
    small_df.to_sql('Tate', conn)


# In[ ]:


# Json Format
small_df.to_json('default.json')
small_df.to_json('table.json', orient='table')


# # Plotting

# In[ ]:


df = pd.read_pickle(os.path.join('Selected_Col.pickle'))


# In[ ]:


acquisition_years = df.groupby('acquisitionYear').size()
acquisition_years.plot()


# <img src="https://i.ibb.co/ncY70mQ/Screenshot-2018-12-23-at-2-45-33-PM.png">

# Excercise to practice [click here](https://www.kaggle.com/vj1998/excercise-to-get-started-with-pandas-for-beginners)

# In[ ]:




