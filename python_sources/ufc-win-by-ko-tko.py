#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/ufcdata/data.csv")


# **Question: Person who has won more fights by KO/TKO?**

# In[ ]:


#filter the necessary columns from the main DataFrame
df = df[["R_fighter","B_fighter","Winner","B_win_by_KO/TKO",'R_win_by_KO/TKO']]


# In[ ]:


#split the DataFrame into red and blue
#filter each fighter in DataFrames by the highest occurance of 'Win by KO/TKO'
df_blue = pd.DataFrame(df.groupby(['B_fighter'], sort=False)['B_win_by_KO/TKO'].max())


# In[ ]:


df_red = pd.DataFrame(df.groupby(['R_fighter'], sort=False)['R_win_by_KO/TKO'].max())


# In[ ]:


#further cleaning up the DataFrames
df_blue["fighter"] = df_blue.index


# In[ ]:


df_blue.set_index(np.arange(1774),inplace =True)


# In[ ]:


df_blue = df_blue[["fighter",'B_win_by_KO/TKO']]


# In[ ]:


df_blue["KO/TKO"] = df_blue["B_win_by_KO/TKO"] 


# In[ ]:


del df_blue["B_win_by_KO/TKO"]


# In[ ]:


df_red["fighter"] = df_red.index


# In[ ]:


df_red.set_index(np.arange(1334),inplace =True)


# In[ ]:


df_red.rename(columns={
    'R_win_by_KO/TKO':'KO/TKO'
},inplace =True)


# In[ ]:


df_red = df_red[["fighter",'KO/TKO']] 


# In[ ]:


#merge the two DataFrames 
#filter each fighter to the highest occurance of 'Win by KO/TKO'
df_KO1 = pd.concat([df_red,df_blue])


# In[ ]:


df_KO = pd.DataFrame(df_KO1.groupby(['fighter'], sort=False)['KO/TKO'].max())


# In[ ]:


df_KO["fighter1"] = df_KO.index


# In[ ]:


df_KO.set_index(np.arange(1915),inplace = True)


# In[ ]:


df_KO.rename(columns={
    'fighter1':'fighter'
},inplace =True)


# In[ ]:


df_KO = df_KO[['fighter','KO/TKO']]


# In[ ]:


#arrange in descending order
df_KO = df_KO.sort_values(by = 'KO/TKO',ascending =False)


# In[ ]:


df_KO


# In[ ]:




