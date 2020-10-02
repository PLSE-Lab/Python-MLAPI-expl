#!/usr/bin/env python
# coding: utf-8

# <h1>Vesta: does zeros in Cxx features mean NaN?</h1>
# Hi Everyone!<br>
# In this notebook I'd like to show that <b>maybe</b> zeros in Cxx features mean NaN.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns

import os


# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


df_transactions = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',index_col=0)
df_transactions.head(2)


# In[ ]:


Cxx = ['C{}'.format(x) for x in range(1,15)]
nans = np.isnan(df_transactions[Cxx]).sum(axis=0)
print('Quantity of NaNs in features:')
print(nans)


# In[ ]:


print('Distribution of values of Cxx and the fraud probability for different values.')
fig, ax1 = plt.subplots(7,2,figsize=(16,40))
l = list()
for i in range(7):
    l.append(list())
    for j in range(2):
        l[i].append(ax1[i,j].twinx())
ax2 = np.array(l)
for x in range(1,15):
    dfA = pd.DataFrame(df_transactions['C{}'.format(x)].value_counts())
    dfA = dfA.merge(df_transactions[['isFraud','C{}'.format(x)]].groupby('C{}'.format(x)).mean(),how='left',left_index=True,
                    right_index=True)
    dfA.sort_index(inplace=True)
    i = (x-1)//2
    j = x%2 - 1
    ax1[i,j].set_ylim((0,400000))
    ax1[i,j].bar(dfA.index.values[:16],dfA.loc[0:15,'C{}'.format(x)])
    ax2[i,j].set_ylim((0,0.4))
    ax2[i,j].plot(dfA.index.values[:16],dfA.loc[0:15,'isFraud'],color='r')
    ax1[i,j].set_xlabel('C{}'.format(x))
    ax1[i,j].set_ylabel('Frequency')
    ax2[i,j].set_ylabel('Fraud propability')
    ax1[i,j].set_title('C{}'.format(x),fontdict={'fontsize':20})
plt.subplots_adjust(wspace=0.3,hspace=0.3)


# <b>The reasons to suggest that zero means NaN:</b>
# <ol>
#     <li>There is no NaN values in these features.</li>
#     <li>Despite distribution of all Cxx features looks more or less similar with fast fading from 1 to max value, the probability of zero (in comparison with other values) is different for different Cxx.</li>
#     <li>Assuming that values in Cxx features are consecutive we can see that fraud probability changes smoothly with increasing any Cxx feature value staring from 1, we can see the big fluctuation for bigger values, but it can be explained by small sample size for that values. At the same time the fraud probability for zero sometimes differs a lot from the main pattern. In some cases (like C1, C2) it also can be explained by small sample size but there are cases (like C5, C9, C13) where the sample size is big enough but the fraud probability dramatically differs.</li>
# </ol>

# <b>Can it be helpful?</b><br>
# That's not evident for me but I think it can improve performance of some models. For instance, if we change 0 to NaN in lightGBM, model will never combine 0 and 1 into one bin. 
