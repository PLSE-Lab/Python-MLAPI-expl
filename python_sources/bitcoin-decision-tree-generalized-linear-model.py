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


# [Bitcion Price Prediction- Decision Tree & Generalized Linear Model](https://drive.google.com/file/d/1ZzFJSudf-F8q5lFxCSe-GnnuzzIIoTN9/view?usp=sharing)
# 
# [Printed version](https://drive.google.com/file/d/1ZFQKnUYlfm1QFUm1j6OHIOnIb_ooj_p0/view?usp=sharing)
