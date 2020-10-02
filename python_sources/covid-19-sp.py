#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = gpd.read_file('../input/covid19sp')
shp = gpd.read_file('../input/spmunicipios/35MUE250GC_SIR.shp')


# In[ ]:


display(df)
display(shp)


# In[ ]:


df["city"] = df["city"].str.upper()
df = df.drop("geometry", axis = 1)
df.head()


# In[ ]:


df["last_available_confirmed"] = df["last_available_confirmed"].astype("int")


# In[ ]:


df = df.merge(shp[["CD_GEOCODM", "geometry"]], left_on="city_ibge_code", right_on="CD_GEOCODM", how="left")


# In[ ]:


df.plot(column ='last_available_confirmed')

