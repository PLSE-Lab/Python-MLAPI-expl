#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Read data and make it into DataFrame
dummy_data = pd.read_csv("../input/scl-dummy/Dummy data.csv")

# Rename column in dummy_data from number to new_number
dummy_data = dummy_data.rename(columns = {"number": "new_number"})

# Add 1 to each row in new_number column
dummy_data = dummy_data.apply(lambda row: row.new_number + 1, axis = 1).reset_index()

# Export DataFrame into csv file
dummy_data.to_csv("submission.csv", header  = True, index = False)
