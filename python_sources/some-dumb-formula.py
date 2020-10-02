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


test = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t').rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})


# In[ ]:


test.head()


# In[ ]:


test['length'] = [len(w) for w in test['Text']]


# In[ ]:


subA = np.array(abs(test["Pronoun-offset"] - test["A-offset"]) / test['length'] , dtype=float)


# In[ ]:


subB = np.array((abs(test["Pronoun-offset"] - test["B-offset"]) / test['length']), dtype=float)


# In[ ]:


subN = [w if w >= 0 else 0.0 for w in (1- (subA + subB))]


# In[ ]:


submit = pd.DataFrame({'ID': test['ID'], 'A': subA, 'B':subB, 'NEITHER' : subN}) 


# In[ ]:


submit['B'][1304]


# In[ ]:


submit.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




