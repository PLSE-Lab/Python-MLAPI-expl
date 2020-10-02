# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
mall = pd.read_csv('../input/Mall_Customers.csv')
Gender = mall.Gender.map(lambda t: len(t.replace("F",""))-4)
PricePoints = mall.xs('Spending Score (1-100)', axis=1)
print(PricePoints)
print(Gender)
#print(mall)
# Any results you write to the current directory are saved as output.