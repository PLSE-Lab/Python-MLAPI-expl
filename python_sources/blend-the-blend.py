#!/usr/bin/env python
# coding: utf-8

# # Growing on the shoulders of giants
#  
# Using the result of other open kernels I made a simple blend. All credits for the giants. Inspired by https://www.kaggle.com/ilhamfp31/the-sum-is-greater-than-the-parts

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


submission_1 = pd.read_csv("../input/sub19042/submission_19042.csv") # https://www.kaggle.com/ilhamfp31/the-sum-is-greater-than-the-parts
submission_2 = pd.read_csv("../input/submission-v3/submission.csv")  # https://www.kaggle.com/kunwar31/simple-lstm-with-identity-parameters-fastai
submission_3 = pd.read_csv("../input/blend2/submission.csv") # https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution
submission_4 = pd.read_csv("../input/blend3/submission.csv") # https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
submission_5 = pd.read_csv("../input/blend4/submission.csv") # https://www.kaggle.com/kunwar31/simple-lstm-fastai


# In[ ]:


submission = pd.DataFrame.from_dict({
    'id': submission_1['id'],
    'prediction': (submission_1.prediction.values * 0.5) + (submission_2.prediction.values * 0.3) + (submission_3.prediction.values * 0.125) + (submission_4.prediction.values * 0.0375) + (submission_5.prediction.values * 0.0375)
})


# In[ ]:


submission.to_csv('submission.csv', index=False)

