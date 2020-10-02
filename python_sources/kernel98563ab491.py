# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
file='../input/battledeath.xlsx'
xls=pd.ExcelFile(file)
df1 =xls.parse('2004') #loading sheet
print(df1.head())
df2=xls.parse('2002')
print(df2.head())
df3 =xls.parse(0, skiprows=[0],names=['Country','AAM due to war (2002)'])
print(df3)
df4 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country'])
print(df4.head())
