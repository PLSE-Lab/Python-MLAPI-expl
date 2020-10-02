#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# > ****Now IMPORTING DATA ******

# In[ ]:


pokedata = pd.read_csv('../input/Pokemon.csv')


# In[ ]:


pokedata


# In[ ]:


pokedata.keys()


# > ***Dropping UNNECESSARY Columns***

# In[ ]:


pokedata = pokedata.drop(['Type 1', 'Type 2', 'Generation', 'Legendary'],1)


# In[ ]:


pokedata.head()


# >** Considering TOTAL as the decidable column********** Removing all other columns

# In[ ]:


pokedata = pokedata.drop(['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def'],1)


# In[ ]:


pokedata = pokedata.drop(['Speed'],1)


# In[ ]:


pokedata.head()


# > **> Sorting will decide the BEST and the WORST**********

# In[ ]:


pokedata_sort = pokedata.sort_values('Total')


# In[ ]:


pokedata_sort.head()


# In[ ]:


pokedata_sort.tail()


# > **> Now we get the Sorted Data to decide**********

# In[ ]:


worst_pokemon = pokedata_sort.head(1)


# In[ ]:


worst_pokemon


# In[ ]:


best_pokemon = pokedata_sort.tail(1)


# In[ ]:


best_pokemon


# > **We get the best and worst POKEMONS****

# In[ ]:




