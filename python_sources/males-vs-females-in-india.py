#!/usr/bin/env python
# coding: utf-8

# Sex Ratio and other important aspects derived from the dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/all.csv",sep=",")


plt.figure(figsize= (20, 30))
states = df.groupby('State').sum()
states=states.sort_values(['Persons'],ascending=[0])
states = states[['Males', 'Females']]

states.plot(kind  = 'bar')
plt.xlabel("State name", size = 20)
plt.ylabel("Sex Ratio per city", size  = 20)
plt.show()



# Number of literate males vs females
import matplotlib.pyplot as plt
plt.figure(figsize= (20,15))
states = df.groupby('State').sum()
states = states.sort_values(['Persons..literate'], ascending=[0])
states = states[['Females..Literate', 'Males..Literate']]
print(states)
states.plot(kind  = 'bar', stacked = True , label = ["Males literacy rate", "Females literacy rate"] )
plt.legend()
plt.title("Literacy rate for both the sexes")
plt.xlabel('State name')
plt.ylabel('Number in thousands')
plt.show()


# In[ ]:


# 

