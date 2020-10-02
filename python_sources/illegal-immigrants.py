# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv('../input/arrests.csv')
print(df.head())


#choose only total for certain regions
df2 = df.loc[[4,13,23,24]]
df2

df2.index = list(df2['Border'])#need to convert to list since there was an extra Border row.

df2 = df2.drop(['Border', 'Sector', 'State/Territory'], axis = 1)#remove these columns

cols = df2.columns

print(df2)

#split into mexican immigrants and all immigrants
cols_mex=[]
cols_all=[]
for col in list(cols):
    if 'Mexicans' in col:
        cols_mex.append(col)
    else:
        cols_all.append(col)


df_mex = df2[cols_mex]
df_all = df2[cols_all]

#%matplotlib inline
df_mex.T.plot(title='Mexicans caught')
df_all.T.plot(title = 'All Immigrants caught')
plt.show()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.