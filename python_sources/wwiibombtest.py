# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#I set index_col = 0 because the data is indexed with an Auto Inc. Int. 
data = pd.read_csv("../input/operations.csv", index_col=0, low_memory=False)

print(data.head(n=5))

print("stats:\n")
print(data.describe())

#print(data.index())