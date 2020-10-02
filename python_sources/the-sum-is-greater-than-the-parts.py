#!/usr/bin/env python
# coding: utf-8

# # The Sum is Greater than the Parts
# 
# Not really planning on releasing this kernel but here we go anyway. All credit goes to the man whom I borrowed the result from.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


submission_1 = pd.read_csv("../input/simple-blend/submission_1.csv") # https://www.kaggle.com/kunwar31/simple-lstm-with-identity-parameters-fastai
submission_2 = pd.read_csv("../input/simple-blend/submission_2.csv") # https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution
submission_3 = pd.read_csv("../input/simple-blend/submission_3.csv") # https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
submission_4 = pd.read_csv("../input/simple-blend/submission_4.csv") # https://www.kaggle.com/kunwar31/simple-lstm-fastai


# In[ ]:


submission = pd.DataFrame.from_dict({
    'id': submission_1['id'],
    'prediction': (submission_1.prediction.values * 0.5) + (submission_2.prediction.values * 0.25) + (submission_3.prediction.values * 0.125) + (submission_4.prediction.values * 0.125)
})


# It's clear that weighted average will only get you so far. Perhaps it's best to take this kernel as a reminder that ensembling technique is really impactful and you should try it!
# 
# There are already tons of great ensembling tutorial available online!
# 
# Well, here goes nothing:

# In[ ]:


submission.to_csv('submission.csv', index=False)

