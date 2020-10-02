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


import Levenshtein as lv   #Used to compute Levenshtein distance between two strings


df_train = pd.read_csv('../input/train.csv')  #Loading training data
df_test = pd.read_csv('../input/test.csv')   #Loading test data

#Creating a copy of df_test, it will be used to generate the submission file
df_sub = df_test.copy()

#Computing the Levenshtein distance between each pair of texts and storing it in 'same_security' column.
df_sub['same_security'] = df_sub.apply(lambda x: lv.ratio(x['description_x'], x['description_y']), axis=1)

#Generating the submission file. Only using id and same_security columns
df_sub.loc[:, ['same_security']].to_csv('benchmark_Levenshtein.csv', index=True, index_label='id')


