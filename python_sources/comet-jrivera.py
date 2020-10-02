#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


data1 = pd.read_csv('../input/responses.csv',usecols=['Getting up', 'Energy levels'])
data1.corr()


# In[ ]:


data1 = pd.read_csv('../input/responses.csv',usecols=['Getting up', 'Energy levels'])
data1.cumsum()
plt.figure()
data1.plot()


# In[ ]:


data2 = pd.read_csv('../input/responses.csv',usecols=['Fun with friends', 'Self-criticism'])
data2.corr()


# In[ ]:


data3 = pd.read_csv('../input/responses.csv',usecols=['Number of siblings', 'Cheating in school'])
data3.corr()


# In[ ]:


Questions;
1. Is there a correlation between the fear of getting up and the energy levels of young people
2. Is there a correlation between the interest to have fun with friends and Self-Criticism

3. Is there a correlation between the young person's view on cheating in school and the person's number of siblings

