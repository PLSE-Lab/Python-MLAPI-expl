# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df=pd.read_csv("../input/testset.csv",sep='\s*,\s*')
#print(df.columns.tolist())
#def_value=1
#df.setdefault(key,def_value)
print(df['_tempm'])

df.plot(kind='bar',y='_tempm')
# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.