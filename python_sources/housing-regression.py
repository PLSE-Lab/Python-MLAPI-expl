# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')

print (df_train.dtypes)
print (df_train.isnull().sum())
print (df_train.corr().sort_values(['SalePrice'], ascending = False)['SalePrice'])
print(df_train.SalePrice.describe())
df_train.SalePrice.hist()
plt.show()

# Any results you write to the current directory are saved as output.