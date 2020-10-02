# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
inputData = pd.read_csv(r"../input/BreadBasket_DMS.csv")

print(inputData.dtypes)
print(inputData.columns)
print("Data shape:",inputData.shape)
print(inputData.head())
print("Unique items count:",len(inputData.Item.unique()))
cat_columns = inputData["Item"].astype('category')
inputData["Category"] = cat_columns.cat.codes
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
sns.countplot(x='Category', data=inputData)
plt.show()
inputData.hist()
plt.show()
