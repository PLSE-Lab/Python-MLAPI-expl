#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# # Parsing Water data

# In[ ]:


water = pd.read_csv(
    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Basic and safely managed drinking water services.csv'
)


# In[ ]:


water.head(2)


# In[ ]:


water.shape


# In[ ]:


water.columns


# In[ ]:


water.isnull().sum()


# In[ ]:


def parse_who_data(df):
    df = df.loc[:, [
        'GHO (DISPLAY)',
        'YEAR (DISPLAY)',
        'REGION (DISPLAY)',
        'COUNTRY (CODE)',
        'COUNTRY (DISPLAY)',
        'RESIDENCEAREATYPE (DISPLAY)',
        'Numeric'
    ]]
    
    df['Index'] = df['GHO (DISPLAY)'] + ' - ' + df['RESIDENCEAREATYPE (DISPLAY)']
    
    df = df.drop(columns=[
        'GHO (DISPLAY)',
        'RESIDENCEAREATYPE (DISPLAY)'
    ])

    df = df.rename(columns={
        'YEAR (DISPLAY)':'Year',
        'REGION (DISPLAY)':'Region',
        'COUNTRY (CODE)':'Country code',
        'COUNTRY (DISPLAY)':'Country',
        'Numeric':'Value'
    })
    
    df = df.pivot_table(index = ['Year', 'Region', 'Country code', 'Country'], columns='Index', values='Value').reset_index()
    df.columns = df.columns.rename('')
    
    return df


# In[ ]:


water = parse_who_data(water)


# In[ ]:


water.head()


# # Sanitation

# In[ ]:


sanitation = pd.read_csv(
    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Basic and safely managed sanitation services.csv'
)


# In[ ]:


sanitation.shape


# In[ ]:


sanitation.head(2)


# In[ ]:


sanitation.columns


# In[ ]:


sanitation.isnull().sum()


# In[ ]:


sanitation = parse_who_data(sanitation)


# In[ ]:


sanitation.head()


# # Handwashing

# In[ ]:


hand = pd.read_csv(
    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Handwashing with soap.csv'
)


# In[ ]:


hand.shape


# In[ ]:


hand.head(2)


# In[ ]:


hand.columns


# In[ ]:


hand.isnull().sum()


# In[ ]:


hand = parse_who_data(hand)


# In[ ]:


hand.head()


# # Open defecation

# In[ ]:


defec = pd.read_csv(
    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Open defecation.csv'
)


# In[ ]:


defec.shape


# In[ ]:


defec.head(2)


# In[ ]:


defec.columns


# In[ ]:


defec.isnull().sum()


# In[ ]:


defec = parse_who_data(defec)


# In[ ]:


defec.head()


# # Joining tables

# In[ ]:


water.shape


# In[ ]:


sanitation.shape


# In[ ]:


hand.shape


# In[ ]:


defec.shape


# In[ ]:


join_columns = ['Year', 'Region', 'Country code', 'Country']
metrics = water.merge(sanitation, how='outer', left_on=join_columns, right_on=join_columns)
metrics = metrics.merge(hand, how='outer', left_on=join_columns, right_on=join_columns)
metrics = metrics.merge(defec, how='outer', left_on=join_columns, right_on=join_columns)


# In[ ]:


metrics.shape


# In[ ]:


metrics.head()


# In[ ]:


metrics.to_csv('/kaggle/working/indexes.csv', index=False)

