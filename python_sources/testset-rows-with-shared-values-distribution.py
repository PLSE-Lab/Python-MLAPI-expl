#!/usr/bin/env python
# coding: utf-8

# This kernel shows distribution of rows sharing different percent of values with other rows.
# 
# We can see that ~50% of rows in test dataset does not have "paired" rows which share even 30% of values
# 
# Also I have checked that these rows mostly conatins non-round numbers, so they are not included in public part of testset and possibly could be a part of private testset
# 
# Datasource for this kernel computed here:
# https://www.kaggle.com/upitersoft/test-common-rows

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

data_path = "../input/test-common-rows/"
print(os.listdir("../input"))
print(os.listdir(data_path))


# Any results you write to the current directory are saved as output.


# In[ ]:


from six.moves import cPickle as pickle
import bz2
import numpy as np
import os

def loadPickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickle(pickle_file, data):
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return

def loadPickleBZ(pickle_file):
    with bz2.BZ2File(pickle_file, 'r') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickleBZ(pickle_file, data):
    with bz2.BZ2File(pickle_file, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt

rowNumMatches = loadPickleBZ(data_path+'rowNumMatches.pbz')

matches30 = rowNumMatches[:, 4]
matches50 = rowNumMatches[:, 2]
matches80 = rowNumMatches[:, 0]
print('Percent of common values between rows:\t\t\t   30% common\t   50% common\t   80% common')
for lim in [1, 3, 10]:
    print('Rows sharing same values count <= {} :  > {}  '.format(lim, lim), '\t '
          '\t', (matches30 <= lim).sum(), ':', (matches30 > lim).sum(),
          '\t', (matches50 <= lim).sum(), ':', (matches50 > lim).sum(),
          '\t', (matches80 <= lim).sum(), ':', (matches80 > lim).sum(),
          )

fig = plt.figure(figsize=(18, 8))
ax = plt.subplot(111)
ax.set_yscale('log')
plt.plot(rowNumMatches)
plt.show()


# "Rows sharing same values count <= 1" means that these values are present only in one row
# 
# "Rows sharing same values count > 1" means that these values are present in two or more rows
# 
# By "same values" I mean that two rows share at lest 30/50/80% of values (depending on column in this table)
