# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='iso-8859-1')
data.head()
print(data.head(5))
data.describe
df=data.groupby(by=['state']).sum()
df.sort_values(by=['number'],ascending=False)
df.sort_values(by=['number']).plot.barh()
# Mato Grosso, Paraiba, Sao Paulo have lots of forest fire


print(data.head())
df2=data.groupby(by=['year']).sum()
df2.sort_values(by=['number'],ascending=False)
df2.sort_values(by=['number']).plot.barh()
# 2003 2016 2015 are the top three for having forest fire


df3=data.groupby(by=['month']).sum()
df3.sort_values(by=['number'],ascending=False)
df3.sort_values(by=['number']).plot.barh()
# Julho Outubro Agosto are the top three