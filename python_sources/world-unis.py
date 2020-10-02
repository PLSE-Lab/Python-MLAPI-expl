# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#df1=pd.read_csv('../input/school_and_country_table.csv')
df2=pd.read_csv('../input/timesData.csv')
#print(df1.columns)
print(df2.columns)


print(df2.country.unique())
ind= df2.country.str.contains('India')

print(sum(ind))

print(df2[ind])
