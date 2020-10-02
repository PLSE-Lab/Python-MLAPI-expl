# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/accident.csv')
print(df.head(0))

comb = df[['VE_FORMS','PVH_INVL','PEDS']]
print(comb)

test = df['VE_FORMS'],df['PVH_INVL']
print(test[0:10])