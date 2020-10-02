# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
small_df = train_df.head(20)
print(small_df.shape)
# print(train_df.columns)
# print(train_df.shape)
# print(train_df.dtypes.tolist())
# print(train_df.describe())

# first_train = train_df.iloc[21]
# print(first_train.dtypes)
# print(type(first_train["Alley"]))
# print(first_train["Alley"])
# print(type(first_train["LotArea"]))
# print(first_train["LotArea"])

print(small_df.plot.bar(stacked=True, color=['y','r']))