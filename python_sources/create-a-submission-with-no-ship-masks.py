#!/usr/bin/env python
# coding: utf-8

# This is a simple code to create an empty submission i.e. predicting that all images have no ships on them. 
# 
# Credit to https://www.kaggle.com/osciiart/no-mask-prediction

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


# Read the sample submission and display the first lines
sub = pd.read_csv("../input/sample_submission_v2.csv")
sub.head()


# In[ ]:


# Set EncodedPixels to an empty string
sub['EncodedPixels'] = ""

# Create a CSV file with this data
sub.to_csv("no_mask_prediction.csv", index=None)


# Now, you just have to commit and run this kernel.  Then, jump to the Output tab and submit this file to the competition. This will score 0.520 on the public leaderboard.
