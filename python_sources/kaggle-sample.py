# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# coding=utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

'''
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
'''

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

data_train.sample(3)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
plt.xticks(rotation=90)
plt.show()