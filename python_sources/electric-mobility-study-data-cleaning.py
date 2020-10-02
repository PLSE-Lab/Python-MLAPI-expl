#!/usr/bin/env python
# coding: utf-8

# This is part of a [larger project](https://github.com/maxims94/electric-mobility-study).

# # Dataset 1: Cleaning 

# In[ ]:


import pandas as pd
import numpy as np
from pprint import pprint


# Source: Eurostat, dataset [road_eqr_carpda](http://appsso.eurostat.ec.europa.eu/nui/show.do?dataset=road_eqr_carpda&lang=en)

# In[ ]:


df_orig = pd.read_csv("../input/road_eqr_carpda_1_Data.csv", encoding="ISO-8859-1",na_values=[":"])
df = df_orig.copy()
print(df.head())
print(df.tail())
print(df.columns)


# ## Column: Flag and Footnotes

# In[ ]:


print(df["Flag and Footnotes"].unique())
df[df["Flag and Footnotes"] == 'd']


# In[ ]:


# 'd' means "definition differs"
# Assume for simplicity that it is the same

df = df.drop("Flag and Footnotes",axis=1)


# ## Column: UNIT 

# In[ ]:


df.UNIT.unique()


# In[ ]:


df = df.drop("UNIT",axis=1)


# ## Change column names 

# In[ ]:


df.columns = ["time","country","motor","value"]
df.head()


# ## Column: country 

# In[ ]:


df.country.unique()


# In[ ]:


df.country.replace(to_replace='Germany (until 1990 former territory of the FRG)', value='Germany', inplace=True)
df.country.unique()


# ## Column: value 

# In[ ]:


df[[pd.isnull(x) for x in df.value]]


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


df.dtypes


# In[ ]:


def remove_comma(x):
    return str(x).replace(',','')
df.value = df.value.apply(remove_comma).astype(int)


# In[ ]:


df.dtypes


# ## Column: motor type 

# In[ ]:


df['motor'].unique()


# In[ ]:


# Will also remove \xa0 etc.
df.motor = df.motor.str.strip()

df.motor.unique()


# In[ ]:


mot_num = df[['motor','value']].groupby('motor').sum().loc[:,'value'].squeeze()
mot_num.plot.bar()


# ### Sanity test

# In[ ]:


print(mot_num['Diesel'])
print(mot_num['Diesel (excluding hybrids)'] + mot_num['Hybrid diesel-electric'] + mot_num['Plug-in hybrid diesel-electric'])


# The data is inconsistent. 

# In[ ]:


print(df[df.motor == 'Diesel (excluding hybrids)'])


# In[ ]:


df[(df.time==2017) & (df.country=='Germany')]


# In[ ]:


#df.to_csv('../data/road_eqr_carpda_pre.csv',header=True,index=False)


# ## Combine sparse classes in motor column

# In[ ]:


print(len(df.query('motor == "Diesel (excluding hybrids)" and value == 0')))
len(df.query('motor == "Diesel" and value==0'))


# In[ ]:


df2 = df.copy()


# ## Step 1
# 
# * Drop 'X (excluding hybrids)' (due to lack of data)
# * Drop 'Alternative energy' (since it is merely the sum of 'electric' and 'other')

# In[ ]:


df2=df2[~df.motor.isin(['Diesel (excluding hybrids)','Petrol (excluding hybrids)','Alternative Energy'])]
df2.motor.unique()


# ## Step 2
# 
# * Keep 'Diesel', 'Petroleum products' and 'Electric'
# * Combine all hybrids into one
# * Combine all alternative energies into 'other'

# In[ ]:


df2.motor.replace('Petroleum products', 'petroleum', inplace=True)
df2.motor.replace('Diesel', 'diesel', inplace=True)
df2.motor.replace('Electricity', 'electricity', inplace=True)

df2.motor.replace(['Hybrid electric-petrol', 'Plug-in hybrid petrol-electric', 'Hybrid diesel-electric', 'Plug-in hybrid diesel-electric'], 'hybrid', inplace=True)
df2.motor.replace(['Liquefied petroleum gases (LPG)', 'Natural Gas', 'Hydrogen and fuel cells', 'Bioethanol', 'Biodiesel', 'Bi-fuel', 'Other'], 'other', inplace=True)

df2.motor.unique()


# In[ ]:


df2.describe()


# In[ ]:


df2.head()


# In[ ]:


df3 = pd.DataFrame(columns=df2.columns)
for (i,row) in df2.drop('value',axis=1).drop_duplicates().iterrows():
    partial = df2[(df2.time==row.time) & (df2.country==row.country) & (df2.motor==row.motor)]
    row['value'] = partial.value.sum()
    #print(row.motor, len(partial))
    df3 = df3.append(row)
df3.head()


# In[ ]:


df3[df3.country.isin(['Netherlands','Belgium','Norway'])].head()


# ## Find countries without data on electric cars

# In[ ]:


elec_by_country = df3[df3.motor=='electricity'].groupby('country').value.max()
elec_by_country[elec_by_country == 0]


# **This dataset does not contain data on electric cars in Denmark and Italy!**

# ## Write end result 

# In[ ]:


#df3.to_csv('../data/road_eqr_carpda_cleaned.csv',index=False,header=True)

