# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

df_init = pd.read_csv('../input/BenefitsCostSharing.csv', nrows=1, low_memory=False)
df_2 = pd.read_csv('../input/BenefitsCostSharing.csv')
columns_list = df_init.columns.values.tolist()
trunc_columns_list = df_2.columns.values.tolist()

new_column_dtype = dict()

for column in columns_list:
    print(column)
    if column not in trunc_columns_list:
        new_column_dtype[str(column)] = np.object

print(new_column_dtype)

# Any results you write to the current directory are saved as output.