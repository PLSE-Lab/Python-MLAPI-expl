#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Add a reader -- checkout pandas

df = pd.read_csv("../input/train.csv")
df.head(5)


# Lets start by histograming the age of survivors and those that died. 

# In[ ]:


#Get max and min
maxAge = df.Age.max()
minAge = df.Age.min()
nbins = math.ceil( maxAge-minAge )
#print(maxAge, minAge, nbins)
ax.hist(hamdists, bins=bins, align='left')
ax.set_xticks(bins[:-1])
df[df.Survived==1].Age.hist(bins=nbins, normed=True, align='left')

