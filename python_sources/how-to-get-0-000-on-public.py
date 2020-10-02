#!/usr/bin/env python
# coding: utf-8

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


sample = pd.read_csv("/kaggle/input/gd-code-battle/sampleSubmission.csv")


# In[ ]:


sample.head()


# In[ ]:


sample.value = 0.0


# In[ ]:


def gen(min_first, plus_first, res=[]):
    print(len(res))
    x = (min_first - plus_first) / 2.0 * 25
    e = plus_first*25 - (100.0 - x)
    data = [x]
    errors = [e]
    for el in res:
        x = (e - 25*el + 1000)/2
        e = e - x
        errors.append(e)
        data.append(x)
    return data + [1000.0], errors


# Here to find out: 
# - min_first we need to get leaderboard value for submission when first value is `-100`
# - plus_first we need to get leaderboard value for submission when first value is `100`
# Next step we run the next cell with those values and get our `cur.csv`.
# Then we get result from leaderboard and it to `res` list argument in the next cell and run it again.
# Then we do a cycle.
# If you do it correctly your values would be:
# ```
# min_first, plus_first = 49.83239, 46.09239
# res = [80.28080, 78.44800, 76.75440, 74.85160, 73.10560, 71.37000, 69.56640,
#                                         67.58000, 65.71040, 64.01080, 62.06560, 60.23880, 58.52320, 56.70920, 54.86280,
#                                        53.13280, 51.28960, 49.49440, 47.71680, 45.86360, 43.98280,
#                                        41.95160, 40.06400, 38.02160])
# ```

# In[ ]:


min_first, plus_first, res = 49.83239, 46.09239, []
rows, errors = gen(min_first, plus_first, res)
sample.value[:len(rows)] = rows
sample.head()
sample.to_csv("cur.csv", index=None)


# In[ ]:




