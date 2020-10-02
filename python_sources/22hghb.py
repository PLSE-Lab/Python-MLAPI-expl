# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_d=pd.read_csv('../input/train_date.csv',nrows=180000)
train_d=train_d.dropna(axis=1,thresh=50000)
train_d=train_d.dropna(thresh=140)
train_n=pd.read_csv('../input/train_numeric.csv',nrows=180000)
train_n=train_n.dropna(axis=1,thresh=50000)
train_n=train_n.dropna(thresh=140)
train=pd.merge(train_n,train_d,on='Id')
train=train.dropna(axis=1)
print(train)
print(list(train.columns))
import matplotlib.pyplot as plt
plt.scatter(train['Response'],train.ix[:,1])
plt.savefig('fig.png')

