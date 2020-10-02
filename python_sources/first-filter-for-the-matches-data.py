# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

## This kernel filters the data in two ways:

# 1. Matches where all the ratings are 0 are walk-overs:
HT = pd.read_csv('../input/Datos_HT.csv',sep=';')
HT = HT[HT['Home midfield']!=0 & (HT['Home midfield']!=0)]

# 2. Non readable matches
HT = HT[~np.isnan(HT['Home midfield'])]

# 3. Defining if the Home team won
HT['Home Winner'] = HT['Home Goals']>HT['Away Goals']

# Write into a new file
HT.to_csv('Filtered_Data_HT.csv')
