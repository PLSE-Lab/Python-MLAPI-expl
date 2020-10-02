# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

# Any results you write to the current directory are saved as output.
rowdata=pd.read_csv("../input/winequality-red.csv")
print(rowdata.describe())

high=rowdata[rowdata['quality']>6]
low=rowdata[rowdata['quality']<=6]
p=ttest_ind(high,low,equal_var=False)
print(p)

sns.distplot(high['quality'],color='red')
sns.distplot(high['quality'],color='green')
plt.show()
