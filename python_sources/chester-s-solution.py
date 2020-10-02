# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import linear_model, svm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

dat =pd.read_csv('../input/train.csv')
x = dat.as_matrix(['Sex'])
y = dat.as_matrix(['Survived'])[0].T

print (dat.columns.values)
dat.info
print ('_'*40)
print (dat.info)