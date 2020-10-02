#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# When I found such an interesting dataset, I was surprised to see that there was no kernel at all about it. But there is a good reason for that: you can not load it directly in a dataframe. To be more specific, if you try to run:
# 
# *df = pd.read_csv('/kaggle/input/betfair-sports/betfair_140901.csv')*
# 
# This will return a ParseError:
# 
# *ParserError: Error tokenizing data. C error: Expected 16 fields in line 292284, saw 22*
# 
# Telling explicitly that there are too many columns in at least one line. Let's try to look into it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


f = open('/kaggle/input/betfair-sports/betfair_140901.csv', 'r')
T = f.read().split('\n')
f.close()
headers = T[0].split(',')
bug_lines = []
for t in T:
    c = t.count(',')
    if c > 16:
        bug_lines.append(t)
        print('--------------')
        print(t)
        print('--------------')
        for i in range(len(headers)):
            print(headers[i]+' : '+t.split(',')[i])
        print(t.split(',')[16:])


# So there are to crashing lines. In the first one, it seems that two events are mixed, a Japanese soccer match and a Youth Czech soccer match. 
# The second one looks misformated. There are two void fields, but still 5 values to put in. Maybe with some knowledge, it is fixable.
# But, for the sake of simplicity, we will only remove these two bugged lines.

# In[ ]:


for bl in bug_lines:
    T.remove(bl)

f = open('fixed_betfair_dataset.csv', 'w')
f.write('')
f.close()
f = open('fixed_betfair_dataset.csv', 'a')
for t in T:
    f.write(t + '\n')
f.close()

df = pd.read_csv('fixed_betfair_dataset.csv')


# In[ ]:


df.head()


# Now, enjoy ! :-)
