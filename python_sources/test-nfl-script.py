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

import os
import matplotlib.pyplot as plt
import pandas as pd
basepath = '../input/'
tmp = os.listdir(basepath)
tmp
file = basepath + tmp[0]
file

data = pd.read_csv(file)
data.head()
firstdownyrdtogo = data[data.down == 1].ydstogo
fourthdownyrdtogo = data[data.down == 4].ydstogo
firstdownyrdtogo.shape
fourthdownyrdtogo.shape

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

firstdownyrdtogo.plot.bar(ax=axes[0])
axes[0].set_xlabel("yards to go first down")
fourthdownyrdtogo.plot.bar(ax=axes[1])
axes[1].set_xlabel("yards to go fourth down")

fig = plt.figure(figsize=(12,8))






