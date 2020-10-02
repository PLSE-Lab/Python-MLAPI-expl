# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/vgsales.csv')

dff = df.groupby('Platform').sum()
dff.sort(columns='Global_Sales', inplace=True)
dff.plot(y='Global_Sales', kind='bar')
plt.show()

df.groupby(['Year', 'Platform']).sum().unstack().plot(y='Global_Sales', kind='bar', stacked=True, colormap='Paired')
plt.show()
