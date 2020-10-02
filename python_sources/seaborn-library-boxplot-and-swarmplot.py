#!/usr/bin/env python
# coding: utf-8

# Hi everyone first we will import required libraries. After then we will read 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")


# In this tutorial we will use boxpolt and swarm plot. Let's begin

# In[ ]:


kill.head()


# In[ ]:


kill.manner_of_death.unique()


# In[ ]:


kill.gender.unique()


# In[ ]:


# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
# BOXPLOT
# Plot the orbital period with horizontal boxes
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")
plt.show()


# SWARMPLOT

# In[ ]:


kill.head()


# In[ ]:


# swarm plot
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)
plt.show()

