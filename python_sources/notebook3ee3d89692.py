#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:



df = pd.read_csv('../input/cwurData.csv')
df.head()


# In[ ]:


def return_quality(c):
    return df[df.country == c].quality_of_education.tolist()

us_quality = return_quality('USA')

df_c = df.loc[(df.country == "USA") | (df.country == "United Kingdom"), ['world_rank', 'country', 'quality_of_education']]
print(df_c['quality_of_education'].max())

facet = sns.FacetGrid(df_c, hue="country",aspect=2)
facet.map(sns.kdeplot,'quality_of_education',shade= True)
facet.set(xlim=(0, df_c['quality_of_education'].max()))
facet.add_legend()



# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Quality Dist')
df['quality_of_education'].dropna().astype(int).hist(bins=70, ax=axis1)


# In[ ]:


grouped = df.groupby('country').mean()
grouped.head()

